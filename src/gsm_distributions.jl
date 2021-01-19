
abstract type GSMMixer{R} end

struct RayleighMixer{R} <: GSMMixer{R}
  alpha::R
end
struct LogNormalMixer{R} <: GSMMixer{R}
  μ::R
  σ::R
end

function fun_p_nu(mx::RayleighMixer{R}) where R
  distr = Rayleigh(mx.α)
  return function (x::R)
    return pdf(distr,x)
  end
end
function fun_log_p_nu(mx::RayleighMixer{R}) where R
  distr = Rayleigh(mx.α)
  return function (x::R)
    return logpdf(distr,x)
  end
end

function fun_p_nu(mx::LogNormalMixer{R}) where R
  distr = LogNormal(mx.μ,mx.σ)
  return function (x::R)
      return pdf(distr,x)
  end
end
function fun_log_p_nu(mx::LogNormalMixer{R}) where R
  distr = LogNormal(mx.μ,mx.σ)
  return function (x::R)
      return logpdf(distr,x)
  end
end


struct GSM{Mx,R}
    covariance::Matrix{R}
    covariance_noise::Matrix{R}
    mixer::Mx
end
Base.Broadcast.broadcastable(gsm::GSM) = Ref(gsm)
Base.copy(gsm::GSM) = GSM(copy(gsm.covariance),copy(gsm.covariance_noise),gsm.mixer)
Base.ndims(g::GSM) = size(g.covariance,1)
# for now all functions are defined for GSM{RayleighMixer}
# _nn = no noise  _wn = with noise
fun_p_nu(gsm::GSM) = fun_p_nu(gsm.mixer)
fun_logp_nu(gsm::GSM) = fun_logp_nu(gsm.mixer)

@inline function hasnoise(gsm::GSM ; tol::Real=1E-3)
    Sn = gsm.covariance_noise
    n=size(Sn,1)
    return tr(Sn)/n > tol
end
"""
    Random.rand(gsm::GSM{RayleighMixer,R},n::Integer) where R
Generates random samples from the specified GSM. Returns the vector of mixers,
the latent features, and the xs. If noise is absent, the x samples are the product of
the gs with the mixers.
"""
function Random.rand(gsm::GSM{RayleighMixer{R},R},n::Integer) where R
  hasnoise(gsm) && return _rand_gsmray_wn(gsm,n)
  return _rand_gsmray_nn(gsm,n)
end

function _rand_gsmray_nn(gsm::GSM{RayleighMixer{R},R},n::Integer) where R
  mixers = rand(Rayleigh(gsm.mixer.alpha),n)
  gs = rand( MultivariateNormal(gsm.covariance),n)
  xs = broadcast(*,Transpose(mixers),gs)
  return (mixers,gs,xs)
end
function _rand_gsmray_wn(gsm::GSM{RayleighMixer{R},R},n::Integer) where R
  (mixers,gs,xs) = _rand_gsmray_nn(gsm,n)
  xs .+= rand(MultivariateNormal(gsm.covariance_noise),n)
  return (mixers,gs,xs)
end

"""
        p_xGnu(x,nu,gsm::GSM) -> p::Float64
Probability of input `x` given the mixer `nu`
"""
function p_xGnu(x::AbstractVector{R},nu::Real,gsm::GSM{Mx,R}) where {Mx,R}
    hasnoise(gsm) && return p_xGnu_wn(x,nu,gsm)
    return p_xGnu_nn(x,nu,gsm)
end
function p_xGnu_nn(x::AbstractVector{R},nu::R,p::GSM{Mx,R}) where {Mx,R}
    S=(nu*nu+eps(100.)) .* p.covariance #numerical stability!
    return pdf(MultivariateNormal(S),x)
 end
function p_xGnu_wn(x::AbstractVector{R},nu::R,p::GSM{Mx,R}) where {Mx,R}
    S= @. (nu*nu)*p.covariance + p.covariance_noise
    return pdf(MultivariateNormal(S),x)
end


# aux functios for the no noise case, analytic
@inline function lambdasq(x::AbstractVector{R},gsm::GSM{Mx,R}) where {Mx,R}
    Sg = gsm.covariance
    return dot(x,Sg\x)
end
@inline function lambda(x::AbstractVector{R},gsm::GSM{Mx,R}) where {Mx,R}
  return sqrt(lambdasq(x,gsm))
end

function psibig(k::Integer,x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
   α = gsm.mixer.alpha
   λ = lambda(x,gsm)
   nd = ndims(gsm)
   nn = 1+0.5(k-nd)
   return  (α*λ)^nn *
     besselk( nn , (λ+eps(100.))/α ) / #numerical stability...
     (α*α*sqrt( (2pi)^nd *det(gsm.covariance)))
end

function psibig_ratio(k1::Integer,k2::Integer,
    x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
  α = gsm.mixer.alpha
  λ = lambda(x,gsm)
  nd = ndims(gsm)
  bb = (λ+eps(100.0))/α
  return (α*λ)^(0.5(k1-k2)) *
    besselk(1+0.5(k1-nd),bb) /
    besselk(1+0.5(k2-nd),bb)
end

function psibig_ratio_approx(k::Integer,
    x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R},oneterm::Bool=true) where R
  α = gsm.mixer.alpha
  λ = lambda(x,gsm)
  n = ndims(gsm)
  onet = (α*λ)^(0.5k)
  oneterm && return onet
  return onet * ( 1. + ( (n-k)*(n-k-4)+3 )/(8λ)*α )
end

# aux function for the case with nise, semi-analytic
function psibigtilde(k::Integer,
    x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
  α = gsm.mixer.alpha
  f_integrate(nu::R) = (nu^k)*pdf(Rayleigh(α),nu)*p_xGnu_wn(x,nu,gsm)
  return quadgk(f_integrate,0,Inf)[1]
end

function meancovar_p_gGxnu(nu::R,x::AbstractVector{R},gsm::GSM{Mx,R}) where {Mx,R}
  @assert hasnoise(gsm)
  n = ndims(gsm)
  Sn_inv = gsm.covariance_noise |> inv
  Sg_inv = gsm.covariance |> inv
  return meancovar_p_gGxnu(nu,x,Sg_inv,Sn_inv)
end

# saves a couple of matrix inversions
function meancovar_p_gGxnu(nu::R,x::AbstractVector{R},
      Sg_inv::Matrix{R},Sn_inv::Matrix{R}) where R
  S3_inv = @. Sg_inv + Sn_inv * (nu*nu)
  S3 = inv(S3_inv)
  mu3 = nu .*(S3*(Sn_inv*x))
  return (mu3,S3)
end


#########
# probabilities and moments etc
"""
    function p_x(x::AbstractVector,gsm::GSM) -> p::Float64
Probability of input `x` for the gsm model
"""
function p_x(x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
  hasnoise(gsm) && return p_x_wn(x,gsm)
  return p_x_nn(x,gsm)
end
@inline function p_x_wn(x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
  return psibigtilde(0,x,gsm)
end
@inline function p_x_nn(x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
  return psibig(0,x,gsm)
end

function p_x(x::AbstractVector{R},gsm::GSM{LogNormalMixer{R},R}) where R
    p_nu = fun_p_nu(gsm.mixer)
    f_integrate(nu) = p_nu(nu)*p_xGnu(x,nu,gsm)
    return quadgk(f_integrate,0,Inf)[1]
end

##

function p_x_idx(x,idxs,gsm)
    hasnoise(gsm) && return p_x_idx_wn(x,idxs,gsm)
    return p_x_idx_nn(x,idxs,gsm)
end
function p_x_idx_nn(x,idxs,gsm)
    @assert length(idxs) == 1 "sorry, size too large!"
    function f_integrate(xi)
        n=ndims(gsm)
        xfull = fill(xi,n)
        xfull[idxs] = x
        return p_x_nn(xfull,gsm)
    end
    return quadgk(f_integrate,-Inf,Inf)[1]
end
function p_x_idx_wn(x,idxs,gsm)
    function p_xGnu_wn_idx(x,nu,p::GSM,idx)
        S= @. (nu*nu)*p.covariance + p.covariance_noise
        Sless=S[idx,idx]
        pdf(MultivariateNormal(Sless),x)
    end
    α = gsm.mixer.alpha
    f_integrate(nu) = pdf(Rayleigh(α),nu)*p_xGnu_wn_idx(x,nu,gsm,idxs)
    return quadgk(f_integrate,0,Inf)[1]
end


"""
Log-likelyhood test of datapoint for a given model.
Returns the log-probability that the data has been
generated by a certain gsm model.
"""
function loglik_data(x::AbstractMatrix{R},gsm::GSM{RayleighMixer{R},R}) where R
    out = 0.0
    nx=size(x,2)
    for i in 1:nx
        xv=x[:,i]
        out += log(p_x(x,gsm))
    end
    out
end

## Now the moments of the mixer

"""
        p_nuGx(nu::Float64,x::Vector,gsm::GSM{RayleighMixer}) -> p::Float64
Probability curve for the mixer given the stimulus
"""
function p_nuGx(nu::R,x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
    hasnoise(gsm) && return p_nuGx_wn(nu,x,gsm)
    return p_nuGx_nn(nu,x,gsm)
end

function p_nuGx_nn(nu::R,x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
  Ψ = psibig(0,x,gsm)
  α = gsm.mixer.alpha
  return  p_xGnu_nn(x,nu,gsm) * pdf(Rayleigh(α),nu) / Ψ
end
function p_nuGx_wn(nu::R,x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
  Ψ = psibigtilde(0,x,gsm)
  α = gsm.mixer.alpha
  return p_xGnu_wn(x,nu,gsm) * pdf(Rayleigh(α),nu) / Ψ
end

# conditional mean and variance for mixer
"""
    EnuGx(x::Vector,gsm::GSM{RayleighMixer})

Expectation of the mixer for a given input `x`
"""
function EnuGx(x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
    hasnoise(gsm) && return EnuGx_wn(x,gsm)
    return EnuGx_nn(x,gsm)
end

@inline function EnuGx_nn(x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
    return psibig_ratio(1,0,x,gsm)
end
@inline function EnuGx_wn(x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
    return psibigtilde(1,x,gsm)/psibigtilde(0,x,gsm)
end
# high-values approximation
@inline function EnuGx_nn_approx(x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R};
    oneterm::Bool=true) where R
    return psibig_ratio_approx(1,x,gsm,oneterm)
end

"""
    Var_nuGx(x::Vector,gsm::GSM{RayleighMixer})

Variance of the mixer for a given input `x`
"""
function Var_nuGx(x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
    hasnoise(gsm) && return Var_nuGx_wn(x,gsm)
    return Var_nuGx_nn(x,gsm)
end
function Var_nuGx_nn(x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
    n = ndims(gsm)
    nn = n-1
    return psibig_ratio(2,0,x,gsm) - (psibig_ratio(1,0,x,gsm))^2
end
function Var_nuGx_wn(x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
    n = ndims(gsm)
    psi0 = psibigtilde(0,x,gsm)
    psi1 = psibigtilde(1,x,gsm)
    psi2 = psibigtilde(2,x,gsm)
    return psi2/psi0 - (psi1/psi0)^2
end

# now, conditional probability for single feature

"""
        p_giGx(gi::Float64,i::Integer,x::Vector,gsm::GSM{RayleighMixer}) -> p::Float64
Probability curve for the `i`th feature (the `ith` component of the full `g`)
given the input `x`. Usually k=1
"""
function p_giGx(gk::R,i::Integer,x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
  hasnoise(gsm) && return p_gkGx_wn(gk,i,x,gsm)
  return p_gkGx_nn(gk,i,x,gsm)
end
function p_gkGx_nn(gk::R,k::Integer,x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
  n = ndims(gsm)
  α = gsm.mixer.alpha
  S = gsm.covariance
  Ψ = psibig(0,x,gsm)
  nu = x[k]/gk
  g = x ./ nu # this replaces gk also
  return pdf(Rayleigh(α),nu) / ( Ψ * (nu^(n-1)*abs(gk)) ) *
      pdf(MultivariateNormal(S),g)
end
function p_gkGx_wn(gk::R,k::Integer,x::AbstractVector{R},
      gsm::GSM{RayleighMixer{R},R}) where R
  n = ndims(gsm)
  α = gsm.mixer.alpha
  Sg = gsm.covariance
  Sn = gsm.covariance_noise
  Sg_inv = inv(Sg)
  Sn_inv = inv(Sn)
  Ψ = psibigtilde(0,x,gsm)
  function f(nu::R)
    (mu3,S3) = meancovar_p_gGxnu(nu,x,Sg_inv,Sn_inv)
    sigma3k = sqrt(S3[k,k])
    return pdf(Rayleigh(α),nu) * pdf(Normal(mu3[k],sigma3k),gk) *
      p_xGnu_wn(x,nu,gsm)
  end
  return quadgk(f,0,Inf)[1] / Ψ
end

# expectation and variance, still single feature !

"""
        EgiGx(i::Integer,x::Vector,gsm::GSM{RayleighMixer}) -> mu_gi::Float64
Expectation for the `i`th feature (the `ith` component of the full `g`)
given the input `x`. Usually k=1
"""
function EgiGx(i::Integer,x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
    hasnoise(gsm) && return EgiGx_wn(i,x,gsm)
    return EgiGx_nn(i,x,gsm)
end

# expect_g1(x,gsm) = x[1]*psi_ratio(1,x,gsm)
# variance_g1(x,gsm) =  (x[1]^2)*(psi_ratio(2,x,gsm)-psi_ratio(1,x,gsm)^2)
function EgiGx_nn(i::Integer,x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
    return x[i] * psibig_ratio(-1,0,x,gsm)
end
function EgiGx_wn(i::Integer,x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
  n = ndims(gsm)
  α = gsm.mixer.alpha
  Sg = gsm.covariance
  Sn = gsm.covariance_noise
  Sg_inv = inv(Sg)
  Sn_inv = inv(Sn)
  Ψ = psibigtilde(0,x,gsm)
  function f(nu::R)
    (mu3,S3) = meancovar_p_gGxnu(nu,x,Sg_inv,Sn_inv)
    return pdf(Rayleigh(α),nu) * p_xGnu_wn(x,nu,gsm) * mu3[i]
  end
  return quadgk(f,0,Inf)[1] / Ψ
end

@inline function EgiGx_nn_approx(i::Integer,x::AbstractVector{R},
    gsm::GSM{RayleighMixer{R},R};oneterm::Bool=true) where R
  return x[i]*psibig_ratio_approx(-1,x,gsm,oneterm)
end
# expectation for g square !

@inline function Egi_sqGx_nn(i::Integer,x::AbstractVector{R},
    gsm::GSM{RayleighMixer{R},R}) where R
  return (x[i])^2 * psibig_ratio(-2,0,x,gsm)
end
function Egi_sqGx_wn(i::Integer,x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
  n = ndims(gsm)
  α = gsm.mixer.alpha
  Sg = gsm.covariance
  Sn = gsm.covariance_noise
  Sg_inv = inv(Sg)
  Sn_inv = inv(Sn)
  Ψ = psibigtilde(0,x,gsm)
  function f(nu::R)
    (mu3,S3) = meancovar_p_gGxnu(nu,x,Sg_inv,Sn_inv)
    return pdf(Rayleigh(α),nu) * p_xGnu_wn(x,nu,gsm) * (mu3[i]^2 + S3[i,i])
  end
  return quadgk(f,0,Inf)[1] / Ψ
end

@inline function Egi_sqGx_nn_approx(i::Integer,x::AbstractVector{R},
    gsm::GSM{RayleighMixer{R},R};oneterm::Bool=true) where R
  return (x[i])^2*psibig_ratio_approx(-2,x,gsm,oneterm)
end

"""
        Var_giGx(i::Integer,x::Vector,gsm::GSM{RayleighMixer}) -> mu_gi::Float64
Variance for the `i`th feature (the `ith` component of the full `g`)
given the input `x`. Usually k=1
"""
function Var_giGx(i::Integer,x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
  hasnoise(gsm) && return Var_giGx_wn(i,x,gsm)
  return Var_giGx_nn(i,x,gsm)
end
function Var_giGx_nn(i::Integer,x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
    return  (x[i]^2)*(psibig_ratio(-2,0,x,gsm)-psibig_ratio(-1,0,x,gsm)^2)
end
function Var_giGx_wn(i::Integer,x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
  E = EgiGx_wn(i,x,gsm)
  Esq = Egi_sqGx_wn(i,x,gsm)
  return Esq - E^2
end
function Var_giGx_nn_approx_old(i::Integer,x::AbstractVector{R},
  gsm::GSM{RayleighMixer{R},R}) where R
  n = ndims(gsm)
  α = gsm.mixer.alpha
  λ = lambda(x,gsm)
  return  (x[i]/λ)^2 *(4n-n*n-1)/8
end
function Var_giGx_nn_approx(i::Integer,x::AbstractVector{R},
    gsm::GSM{RayleighMixer{R},R}) where R
  n = ndims(gsm)
  α = gsm.mixer.alpha
  λ = lambda(x,gsm)
  return  0.25*(x[i]/λ)^2
end

function Var_giGx_nn_approx_bis(i::Integer,x::AbstractVector{R},
    gsm::GSM{RayleighMixer{R},R}) where R
  E = EgiGx_nn_approx(i,x,gsm; oneterm=false)
  Esq = Egi_sqGx_nn_approx(i,x,gsm; oneterm=false)
  return Esq - E^2
end

# Fano, too !
"""
        FFgiGx(i::Integer,x::Vector,gsm::GSM{RayleighMixer}) -> mu_gi::Float64
Fano factor for the `i`th feature (the `ith` component of the full `g`)
given the input `x`. Usually k=1
"""
function FFgiGx(i::Integer,x::AbstractVector{R},
    gsm::GSM{RayleighMixer{R},R}) where R
  hasnoise(gsm) && return FFgiGx_wn(i,x,gsm)
  return FFgiGx_nn(i,x,gsm)
end
function FFgiGx_nn(i::Integer,x::AbstractVector{R},
    gsm::GSM{RayleighMixer{R},R}) where R
  psi1 = psibig_ratio(-1,0,x,gsm)
  psi2 = psibig_ratio(-2,0,x,gsm)
  return x[i]*(psi2/psi1-psi1)
end
function FFgiGx_wn(i::Integer,x::AbstractVector{R},
    gsm::GSM{RayleighMixer{R},R}) where R
  E = EgiGx_wn(i,x,gsm)
  Esq = Egi_sqGx_wn(i,x,gsm)
  return Esq/E - E
end

function FFgiGx_nn_approx(i::Integer,x::AbstractVector{R},
    gsm::GSM{RayleighMixer{R},R}) where R
  n = ndims(gsm)
  α = gsm.mixer.alpha
  λ = lambda(x,gsm)
  return  x[i]*sqrt(α)/(4.0*λ*sqrt(λ))
end

function FFgiGx_nn_approx_alt(i::Integer,x::AbstractVector{R},
    gsm::GSM{RayleighMixer{R},R};oneterm::Bool=true) where R
  psi1 = psibig_ratio_approx(-1,x,gsm,oneterm)
  psi2 = psibig_ratio_approx(-2,x,gsm,oneterm)
  return x[i]*(psi2/psi1-psi1)
end


# Now the same, but for multiple elements of g.
# must be the case with noise only

"""
    p_gGx(g::Vector,idxs::Vector,x::Vector,gsm::GSM{RayleighMixer})

"""
function p_gGx(g::AbstractVector{R},
    idxs::Vector{<:Integer},x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
  @assert hasnoise(gsm)
  @assert length(g) == length(idxs)
  n = ndims(gsm)
  α = gsm.mixer.alpha
  Sg = gsm.covariance
  Sn = gsm.covariance_noise
  Sg_inv = inv(Sg)
  Sn_inv = inv(Sn)
  Ψ = psibigtilde(0,x,gsm)
  function f(nu::R)
    (mu3,S3) = meancovar_p_gGxnu(nu,x,Sg_inv,Sn_inv)
    mu3less = mu3[idxs]
    S3less = S3[idxs,idxs]
    multnorm = MultivariateNormal(mu3less,S3less)
    return pdf(Rayleigh(α),nu) * pdf(multnorm,g) *
      p_xGnu_wn(x,nu,gsm)
  end
  return quadgk(f,0,Inf)[1] / Ψ
end

"""
        EgiGx(i::Integer,x::Vector,gsm::GSM{RayleighMixer}) -> mu_gi::Float64
Expectation for the `i`th feature (the `ith` component of the full `g`)
given the input `x`. Usually k=1
"""
function EgGx(idxs::Vector{<:Integer},x::AbstractVector{R},
    gsm::GSM{RayleighMixer{R},R}) where R
  @assert hasnoise(gsm)
   n = ndims(gsm)
   α = gsm.mixer.alpha
   Sg = gsm.covariance
   Sn = gsm.covariance_noise
   Sg_inv = inv(Sg)
   Sn_inv = inv(Sn)
   Ψ = psibigtilde(0,x,gsm)
   return map(idxs) do idx
       function f(nu)
           (mu3,_) = meancovar_p_gGxnu(nu,x,Sg_inv,Sn_inv)
           return pdf(Rayleigh(α),nu) * p_xGnu_wn(x,nu,gsm) * mu3[idx]
       end
       return quadgk(f,0,Inf)[1] / Ψ
   end
end

function EgigjGx(idx_i::Integer,idx_j::Integer,
    x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
  @assert hasnoise(gsm)
  n = ndims(gsm)
  α = gsm.mixer.alpha
  Sg = gsm.covariance
  Sn = gsm.covariance_noise
  Sg_inv = inv(Sg)
  Sn_inv = inv(Sn)
  Ψ = psibigtilde(0,x,gsm)
  function f(nu::R)
      (mu3,S3) = meancovar_p_gGxnu(nu,x,Sg_inv,Sn_inv)
      factor_ij = S3[idx_i,idx_j] + (mu3[idx_i]*mu3[idx_j])
      return pdf(Rayleigh(α),nu) * p_xGnu_wn(x,nu,gsm) * factor_ij
  end
  return quadgk(f,0,Inf)[1] / Ψ
end

function Pearson_gigjGx(idx_i::Integer,idx_j::Integer,
    x::AbstractVector{R},gsm::GSM{RayleighMixer{R},R}) where R
  @assert hasnoise(gsm)
  Egi,Egj = EgGx([idx_i,idx_j],x,gsm)
  EgijGx = EgigjGx(idx_i,idx_j,x,gsm)
  vgi = Var_giGx(idx_i,x,gsm)
  vgj = Var_giGx(idx_j,x,gsm)
  return (EgijGx - Egi*Egj)/sqrt(vgi*vgj)
end

"""
    fit_gsm(xs::AbstractMatrix{R},mixer::GSMMixer{R}) where R

Fits a GSM model with known mixer distrbution and no noise.

# Arguments
+ xs : Matrix where columns are observations, and rows are dimensions.
+ mixer::GSMMixer : the mixer of the GSM , for example `RayleighMixer(1.0)`

# Output
+ gsm::GSMModel : the gsm fit on data
"""
function fit_gsm(xs::AbstractMatrix{R},mixer::GSMMixer{R}) where R
    mixer_snd = second_moment(mixer)
    Σx = cov(xs;dims=2)
    Σg = Σx ./ mixer_snd
    Σnoise = zeros(Σg)
    return GSM(Σg,Σnoise,mixer)
end

# Mixer normalization factor for Rayleigh
function second_moment(mixer::RayleighMixer{R}) where R
    return (2.0mixer.alpha*mixer.alpha)
end

function fit_gsm_addnoise!(gsm::GSM,noise_scal::R,
                xs_noise::AbstractMatrix{R};correct_Sg::Bool=false) where R
    Σnoise = cov(xs_noise;dims=2)
    Σnoise ./=  noise_scal*tr(gsm.covariance)/tr(Σnoise)
    gsm.covariance_noise .= Σnoise
    if correct_Sg
        mixer_snd = second_moment(gsm.mixer)
        @. gsm.covariance -= Σnoise /mixer_snd
        @assert isposdef(gsm.covariace) "could not correct GSM variance to account for the noise... not enough noise in train data? Reduce noise scaling or add noise to gsm training inputs!"
    end
    return nothing
end

# Laplace approximation !

# find most likely mixer

function mle_nuGx(x::AbstractVector{R},gsm::GSM) where R
    log_p_nu = fun_log_p_nu(gsm)
    function log_p_xGnu(nu::R)
        distr = MultivariateNormal(@. gsm.covariance .+ nu*nu*gsm.covariance_noise)
        return logpdf(distr,x)
    end
    objfun(nu::Real) = - (log_p_nu(nu) + log_p_xGnu(nu))
    res=optimize(objfun, 1E-6,100.)
    return Optim.minimizer(res)
end

function laplace_samples_p_gGx(nsampl::Integer,
            x::AbstractVector{R},gsm::GSM) where R
    nu_tilde = mle_nuGx(x,gsm)
    mu3,S3=meancovar_p_gGxnu(nu_tilde,x,gsm)
    distr=MultivariateNormal(mu3,S3)
    return rand(distr,nsampl)
end
