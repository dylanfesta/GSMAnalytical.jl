

struct NormalMixer{R} <: GSMMixer{R}
  μ::R
  σ::R
end

"""
    NormalMixer(X::Matrix{<:Real})
Defines a normal mixer by moment-matching.
# inputs
- `X` : rows are filter outputs, columns are datapoints (different img patches)
# output
- `mixer::NormalMixer`
"""
function NormalMixer(X::Matrix{<:Real})
  # empirical distribution
  mus = mapslices(mean,X;dims=1)
  return NormalMixer(mean_and_std(mus)...)
end

struct GSuM{Mx,R}
    covariance::AbstractMatrix{R}
    covariance_noise::AbstractMatrix{R}
    mixer::Mx
end
n_dims(g::GSuM) = size(g.covariance,1)
Base.Broadcast.broadcastable(g::GSuM) = Ref(g)

@inline function hasnoise(gsum::GSuM ; tol::Real=1E-4)
    return tr(gsum.covariance_noise) > tol
end

function add_xnoise(gsum::GSuM{NormalMixer,R},xs::AbstractArray{R}) where R
  Σn = gsum.covariance_noise
  if tr(Σn) < 1E-4 # no noise, really
    return copy(xs)
  else
    return xs .+ rand(MultivariateNormal(Σn),size(xs,2))
  end
end

function get_samples(gsum::GSuM{NormalMixer{R},R}, nsamples::Integer) where R
    Σg=gsum.covariance
    Σn=gsum.covariance_noise
    mixers = rand(Normal(gsum.mixer.μ, gsum.mixer.σ),nsamples)
    gs=rand(MvNormal(Σg),nsamples)
    hasnoise = tr(Σn) > 1E-5
    if hasnoise
        noises=rand(MvNormal(Σn),nsamples)
    else
        noises = zero(gs)
    end
    xs = broadcast(+,mixers',gs,noises)
    return (gs=gs,xs=xs,mixers=mixers,noise=noises)
end

function Random.rand(gsum::GSuM{NormalMixer{R},R},n::Integer) where R
  s=get_samples(gsum,n)
  return (s.mixers,s.gs,s.xs)
end

"""
  fit_given_mixer_and_noise(::Type{GSuM},
      x_input::Matrix{R},Σnoise::Matrix{R},mixer::NormalMixer{R}) where R

Fits an additive (and not multiplicative)
GSM model with known mixer distrbution and known noise.

# Arguments
+ `::Type{GSuM}` : this should be `GSuM`
+ `x_input` : Matrix where columns are observations, and rows are dimensions.
+ `Σnoise::Matrix{R}` : the noise covariance matrix
+ `mixer::NormalMixer` : the mixer of the GSM , for example `RayleighMixer(1.0)`

# Output
+ gsm::GSuM : the gsm fit on data
"""
function fit_given_mixer_and_noise(::Type{GSuM},
    x_input::Matrix{R},Σnoise::Matrix{R},mixer::NormalMixer{R}) where R
  #remove mean
  # x_input_zero = broadcast(-,x_input,mean(x_input;dims=1))
  x_input_zero = x_input .- mixer.μ
  Σg = cov(x_input_zero;dims=2)
  if !isposdef(Σg)
    min_eig = minimum(eigvals(Σg))
    @warn """ Covariance matrix not positive definite.
     Added a diagonal of 1E-4 to fix the issue
    """
    Σg += 1E-4I
  end
  @assert isposdef(Σg)
  return GSuM(Σg,Σnoise,mixer)
end

"""
  fit_given_mixer_and_noise(::Type{GSM},
      x_input::Matrix{R},Σnoise::Matrix{R},mixer::RayleighMixer{R}) where R

Fits a GSM model with known mixer distrbution and known noise.

# Arguments
+ `::Type{GSM}` : this should be `GSM`
+ `x_input` : Matrix where columns are observations, and rows are dimensions.
+ `Σnoise::Matrix{R}` : the noise covariance matrix
+ `mixer::RayleighMixer` : the mixer of the GSM , for example `RayleighMixer(1.0)`

# Output
+ gsm::GSM : the gsm fit on data
"""
function fit_given_mixer_and_noise(::Type{GSM},
    x_input::Matrix{R},Σnoise::Matrix{R},mixer::RayleighMixer{R}) where R
  mixer_snd = second_moment(mixer)
  Σx = cov(x_input;dims=2)
  Σg = Σx ./ mixer_snd
  return GSM(Σg,Σnoise,mixer)
end



function scale_covariance_noise!(gsum::GSuM,noise_level::Real)
  scale_covariance_noise!(gsum.covariance_noise,
            gsum.covariance,noise_level)
  return nothing
end

##### distributions down here

function distr_p_x(gsum::GSuM{NormalMixer{R},R}) where R
  n=n_dims(gsum)
  muu = fill(gsum.mixer.μ,n)
  S = broadcast(+,gsum.mixer.σ^2, gsum.covariance_matrix, gsum.covariance_matrix_noise)
  return MultivariateNormal(muu,S)
end

function distr_p_xi(i::Integer,gsum::GSuM{NormalMixer{R},R}) where R
  n=n_dims(gsum)
  muu = fill(gsum.mixer.μ,n)
  S = broadcast(+,gsum.mixer.σ^2, gsum.covariance_matrix, gsum.covariance_matrix_noise)
  return Normal(muu[i],sqrt(S[i,i]))
end

function pars_p_nuGx(x::AbstractVector{R},gsum::GSuM{NormalMixer{R},R}) where R
  iσ_sq = inv(gsum.mixer.σ^2)
  Ssum_inv = inv(gsum.covariance+gsum.covariance_noise)
  σstar_sq = inv( iσ_sq + sum( Ssum_inv) )
  mu_star = σstar_sq * (sum(Ssum_inv*x) + gsum.mixer.μ * iσ_sq)
  return mu_star,sqrt(σstar_sq)
end
function pars_p_nuGx(xs::AbstractMatrix{R},gsum::GSuM{NormalMixer{R},R}) where R
  iσ_sq = inv(gsum.mixer.σ^2)
  Ssum_inv = inv(gsum.covariance+gsum.covariance_noise)
  σstar_sq = inv( iσ_sq + sum( Ssum_inv) )
  μstar = map(eachcol(xs)) do x
    σstar_sq * (sum(Ssum_inv*x) + gsum.mixer.μ * iσ_sq)
  end
  σstar  = fill(sqrt(σstar_sq),length(μstar))
  return μstar,σstar
end

function distr_p_mixGx(x::AbstractVector{R},gsum::GSuM{NormalMixer{R},R}) where R
  μstar,σstar=pars_p_nuGx(x,gsum)
  return Normal(μstar,σstar)
end

function distr_p_giGx(i::Integer, x::AbstractVector{R},
    gsum::GSuM{NormalMixer{R},R}) where R
  Σn = gsum.covariance_noise
  mixvar = gsum.mixer.σ^2
  n=n_dims(gsum)
  if tr(Σn) < 1E-5 # no noise
    S = inv(inv(gsum.covariance) .+ inv(mixvar))
    muu_boh = vec(sum(S;dims=2)) .*(sum(x .- gsum.mixer.μ)/mixvar)
    allones = ones((n,n))
    muu = (S*allones *(x .- gsum.mixer.μ) ) ./ mixvar
  else
    Ssum = mixvar .+ Σn
    S = inv(inv(gsum.covariance_matrix) + inv(Ssum))
    muu = (S/Ssum)*(x .- gsum.mixer.μ)
  end
  return Normal(muu[i],sqrt(S[i,i]))
end

function p_nu(nu::R,gsum::GSuM{NormalMixer{R},R}) where R
  return pdf(Normal(gsum.mixer.μ,gsum.mixer.σ),nu)
end

function p_xGnu(x::AbstractVector{R},nu::R,gsum::GSuM{NormalMixer{R},R}) where R
  n=n_dims(gsum)
  nu_all = fill(nu,n)
  d=MultivariateNormal(nu_all,gsum.covariance .+ gsum.covariance_noise)
  return pdf(d,x)
end

function p_x_nu(x::AbstractVector{R},nu::R,gsum::GSuM{NormalMixer{R},R}) where R
  return p_xGnu(x,nu,gsum) * p_nu(nu,gsum)
end
function p_nuGx(nu::R,x::AbstractVector{R},gsum::GSuM{NormalMixer{R},R}) where R
  return pdf(distr_p_mixGx(x,gsum),nu)
end




function sigmastarsqprime(x::AbstractArray{<:Real},
    gsum::GSuM{NormalMixer{R},R}) where R
  @assert !hasnoise(gsum) "must have no noise"
  μstar,σstar=pars_p_nuGx(x,gsum)
  Σg = gsum.covariance
  id=fill(1.0,size(Σg,1))
  dL = dotinvmatprodprime(id,Σg,id)
  dL .*= -σstar^4
  return dL
end
function sumstarprime(x::AbstractArray{<:Real},
    gsum::GSuM{NormalMixer{R},R}) where R
  μstar,σstar=pars_p_nuGx(x,gsum)
  dL1 = sigmastarsqprime(x,gsum)
  Σg = gsum.covariance
  id=fill(1.0,size(Σg,1))
  dL2 = dotinvmatprodprime(id,Σg,x)
  return @. 2.0 * (μstar^2/σstar^2 * dL1 + σstar^2*μstar*dL2)+dL1
end

function mustarprime(x::AbstractArray{<:Real},
    gsum::GSuM{NormalMixer{R},R}) where R
  dL1 = sigmastarsqprime(x,gsum)
  μstar,σstar=pars_p_nuGx(x,gsum)
  Σg = gsum.covariance
  id=fill(1.0,size(Σg,1))
  dL2 = dotinvmatprodprime(id,Σg,x)
  return @. (μstar/σstar^2) * dL1 + σstar^2 * dL2
end



"""
    conditional_expectation_thingy(x::AbstractVector{R},
        gsum::GSuM{NormalMixer{R},R}) where R

This is the numerical version of the M step for the EM of the
additive GSM model (GSuM). Returns the expectation of log(P(x,mixer)) over P(mixer|x)
"""
function conditional_expectation_EM_num(x::AbstractVector{R},
    gsum::GSuM{NormalMixer{R},R}) where R
  f_integrate = function (_nu::R)
    val=p_x_nu(x,_nu,gsum)
    val < 1E-10 && return 0.0
    return log(val) * p_nuGx(_nu,x,gsum)
  end
  return quadgk(f_integrate,-Inf,Inf)[1]
end

"""
    conditional_expectation_thingy(x::AbstractVector{R},
        gsum::GSuM{NormalMixer{R},R}) where R

This is the analytic version of the M step for the EM of the
additive GSM model (GSuM). Returns the expectation of log(P(x,mixer)) over P(mixer|x)
"""
function conditional_expectation_EM(x::AbstractVector{R},
    gsum::GSuM{NormalMixer{R},R}) where R
  @assert !hasnoise(gsum) "Case with noise not covered yet!"
  n=n_dims(gsum)
  icov=inv(gsum.covariance)
  μstar,σstar=pars_p_nuGx(x,gsum)
  μmix,σmix=gsum.mixer.μ,gsum.mixer.σ
  sumstar = μstar^2+σstar^2
  iv=fill(1.,n_dims(gsum))
  ret = 0.0
  ret -= 0.5dot(x,icov,x)
  ret += μstar*dot(x,icov,iv)
  ret -=  0.5dot(iv,icov,iv) * sumstar
   ret -= 0.5*n*log(2π)
   ret -= 0.5*log(det(gsum.covariance))
   ret -= 0.5sumstar/σmix^2
   ret += μmix*μstar/σmix^2
   ret -= 0.5*μmix^2/σmix^2
   ret -= 0.5log(2π*σmix^2)
   return ret
end


function conditional_expectation_EM_prime(x::AbstractVector{R},
    gsum::GSuM{NormalMixer{R},R}) where R
  @assert !hasnoise(gsum) "Case with noise not covered yet!"
  n=n_dims(gsum)
  Σ=gsum.covariance
  iΣ=inv(Σ)
  μstar,σstar=pars_p_nuGx(x,gsum)
  μmix,σmix=gsum.mixer.μ,gsum.mixer.σ
  sumstar = μstar^2+σstar^2
  iv=fill(1.,n_dims(gsum))
  _sumstarprime = sumstarprime(x,gsum)
  _mustarprime = mustarprime(x,gsum)
  fill!(dest,0.0)
  dest = -0.5.*dotinvmatprodprime(x,Σ,x)
  dest .+= μstar.*dotinvmatprodprime(x,Σ,iv)
  dest .+= _mustarprime.*dot(x,iΣ,iv)
  dest .-= 0.5*sumstar.*dotinvmatprodprime(iv,Σ,iv)
  dest .-= 0.5*dot(iv,iΣ,iv) .* _sumstarprime
  dest .-=  0.5.*logdetprime(Σ)
  dest .-=  (0.5/σmix^2) .* _sumstarprime
  dest .+= (μmix/σmix^2).*_mustarprime
  return dest
end





#=





mutable struct GSuM_Model
  patch_size::Integer
  nsurround::Integer # 4 or 8
  bank::OnePyrBank
  #
  noise_level::Real
  x_natural::Matrix
  gsum::GSuM
  #
  stim_pars::Dict # contrast, shift form center, size, etc
  #
  views::DataFrame  # collects all the features used , has idx element
  samples::DataFrame # for each view, vs and gs samples, idx matches the above
end


function get_x_natural(nsamples::Integer,bank::OnePyrBank,
      natural_images,size_patch::Integer)
  natpatches = sampling_tiles(nsamples, natural_images, size_patch)
  return apply_bank(natpatches, bank, Xstd(2.0),false)
end



function GSuM_Model(x_natural::Matrix{<:Real},noise_level::Real,
    bank::OnePyrBank,patch_size::Integer)
  mixer = NormalMixer(x_natural)
  Σnoise = get_covariance_noise(size(x_natural,2),patch_size,bank)
  gsum = fit_given_mixer_and_noise(GSuM,x_natural,Σnoise,mixer)
  scale_covariance_noise!(gsum,noise_level)
  views=DataFrame()
  samples=DataFrame()
  stim_pars=Dict()
  nsurround=get_nsurround(bank)
  GSuM_Model(patch_size,nsurround,bank,noise_level,x_natural,gsum,
    stim_pars, views,samples)
end



=#
