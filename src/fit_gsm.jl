

"""
    make_rand_cov_mat( dims , diag_val::Float64 ; k_dims=5)

Returns a random covariance matrix that is positive definite
and has off-diagonal elements.
# Arguments
- `d`: dimensions
- `diag_val`: scaling of the diagonal
- `k-dims`: tuning of off diagonal elements
"""
function make_rand_cov_mat(dims::Integer,diag_val::Real,k_dims::Integer=5)
  W = randn(dims,k_dims)
  S = W*W'+ Diagonal(rand(dims))
  temp_diag = Diagonal(inv.(sqrt.(diag(S))))
  S = temp_diag * S * temp_diag
  S .*= diag_val
  # perfectly symmetric
  for i in 1:dims
    for j in i+1:dims
      S[i,j]=S[j,i]
    end
  end
  return S
end

"""
    make_noise(nsamples::Integer,gb::GaborBank)
White noise filtered by bank, normalized so that
mean is 0 and std is 1
"""
function make_noise(nsamples::Integer,gb::GaborBank)
    sz = gb.frame_size
    noise=randn(sz,sz,nsamples)
    xs_noise =  gb(noise)
    # normalize
    xs_noise ./= mean_std(cov(xs_noise;dims=2))
    return xs_noise
end


function gsm_fit_factor(mixer::RayleighMixer)
  return inv(2.0mixer.alpha*mixer.alpha)
end

struct GSM_Neuron{Mx,R}
  gsm::GSM{Mx,R}
  filter_bank::GaborBank{R}
end

function mean_std(covmat::Matrix{<:Real})
  n=size(covmat,1)
  return sqrt(tr(covmat)/n)
end

function GSM_Neuron(x_train::Matrix{R},x_noise::Matrix{R},
      mixer::GSMMixer{R}, bank::GaborBank{R};
      train_noise::Bool=false,test_bank::Bool=true,
      normalize_noise_cov::Bool=true) where R
  @assert !train_noise "Can only train on noiseless data!"
  Σx = cov(x_train;dims=2)
  nsamples=size(x_train,2)
  Σnoise = cov(x_noise;dims=2)
  Σg = rmul!(Σx, gsm_fit_factor(mixer))
  gsm = GSM(Σg,Σnoise,mixer)
  if test_bank
    @assert ndims(bank) == size(Σg,1) "Filter bank has the wrong dimensionality!"
  end
  return GSM_Neuron(gsm,bank)
end

function GSM_Neuron(x_train::Matrix{R},noise_level::Real,
    mixer::GSMMixer{R}, bank::GaborBank{R};
    train_noise::Bool=false,test_bank::Bool=true,
    normalize_noise_cov::Bool=true) where R
  @assert !train_noise "Can only train on noiseless data!"
  Σx = cov(x_train;dims=2)
  nsamples=size(x_train,2)
  x_noise = make_noise(nsamples,bank)
  Σnoise = cov(x_noise;dims=2)
  Σg = rmul!(Σx, gsm_fit_factor(mixer))
  if normalize_noise_cov
    stdg = mean_std(Σg)
    Σnoise .*= (noise_level*stdg)^2
  end
  gsm = GSM(Σg,Σnoise,mixer)
  if test_bank
    @assert ndims(bank) == size(Σg,1) "Filter bank has the wrong dimensionality!"
  end
  return GSM_Neuron(gsm,bank)
end

function GSM_Neuron(train_patches::Array{R,3},
    noise_level::R,mixer::GSMMixer{R},
    bank::GaborBank{R}; train_noise::Bool=false) where R
  @assert bank.frame_size == size(train_patches,1) == size(train_patches,2) "Error in patch sizes!"
  x_train=bank(train_patches)
  return GSM_Neuron(x_train,noise_level,mixer,bank;
    train_noise=train_noise,normalize_noise_cov=true)
end
