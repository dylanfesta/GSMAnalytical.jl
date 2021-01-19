
#=

struct NormalMixer <: MixerType
    μ::Real
    σ::Real
end

struct GSuM{Mx,R}
    covariance_matrix::Matrix{R}
    covariance_matrix_noise::Matrix{R}
    mixer::Mx
end
n_dims(g::GSuM) = size(g.covariance_matrix,1)
Base.Broadcast.broadcastable(g::GSuM) = Ref(g)

function get_samples(gsum::GSuM{<:NormalMixer}, nsamples)
  Σg=gsum.covariance_matrix
  Σn=gsum.covariance_matrix_noise
  mixers = rand(Normal(gsum.mixer.μ, gsum.mixer.σ),nsamples)
  gs=rand(MvNormal(Σg),nsamples)
  hasnoise = tr(Σn) > 1E-5
  if hasnoise
    noises=rand(MvNormal(Σn),nsamples)
  else
    noises = zero(gs)
  end
  xs = broadcast(+,mixers',gs,noises)
  (gs=gs,xs=xs,mixers=mixers,noise=noises)
end


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

function NormalMixer(X::Matrix{<:Real})
  # empirical distribution
  mus = mapslices(mean,X;dims=1)
  return NormalMixer(mean_and_std(mus)...)
end

function get_x_natural(nsamples::Integer,bank::OnePyrBank,
      natural_images,size_patch::Integer)
  natpatches = sampling_tiles(nsamples, natural_images, size_patch)
  return apply_bank(natpatches, bank, Xstd(2.0),false)
end

function fit_given_mixer_and_noise(::Type{GSuM},
    x_natural::Matrix{<:Real},Σnoise::Matrix{<:Real},mixer)
  x_natural_zero = broadcast(-,x_natural,mean(x_natural;dims=1))
  Σg = cov(x_natural_zero;dims=2)
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

function scale_covariance_noise!(gsum::GSuM,noise_level::Real)
  scale_covariance_noise!(gsum.covariance_matrix_noise,
            gsum.covariance_matrix,noise_level)
  return nothing
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



function add_xnoise(gsum::GSuM,xs::AbstractArray{<:Real})
  Σn = gsum.covariance_matrix_noise
  if tr(Σn) < 1E-4
    return copy(xs)
  else
    return xs .+ rand(MultivariateNormal(Σn),size(xs,2))
  end
end
=#
