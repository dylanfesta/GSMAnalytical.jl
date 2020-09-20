

"""
    make_rand_cov_mat( dims , diag_val::Float64 ; k_dims=5)

Returns a random covariance matrix that is positive definite
and has off-diagonal elements.
# Arguments
- `d`: dimensions
- `diag_val`: scaling of the diagonal
- `k-dims`: tuning of off diagonal elements
"""
function make_rand_cov_mat( dims::Integer , diag_val::Real , (k_dims::Integer)=5)
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

# TODO
function make_noise(gb::GaborBank,nsamples::Integer)
    sz = gb.frame_size
    noise=randn(sz,sz,nsamples)
    xs_noise =  gb(noise)
    # normalize
    xs_noise .*= inv(sqrt(mean(diag(cov(xs;dims=2)))))
    return
end


function gsm_fit_factor(mixer::RayleighMixer)
  return inv(2.0mixer.alpha*mixer.alpha)
end

struct GSM_Neuron
  gsm::GSM
  filter_bank::GaborBank
end

function GSM_Neuron(x_train::Matrix{<:Real},x_noise::Matrix{<:Real},
    mixer::GSMMixer, bank::GaborBank; train_noise::Bool=false)
  @assert !train_noise "other condition not covered, yet"
  Σx = cov(x_train;dims=2)
  Σnoise = cov(x_noise;dims=2)
  Σg = rmul!(Σx, gsm_fit_factor(mixer) )
  gsm = GSM(Σg,Σnoise,mixer)
  if ndims(bank) != size(Σg,1)
    @error "Filter bank has the wrong dimensionality! This is a test?"
  end
  return GSM_Neuron(gsm,bank)
end

function GSM_Neuron(x_train::Matrix{<:Real},mixer::GSMMixer,
    bank::GaborBank; train_noise::Bool=false)
  nsamples = size(x_train,2)
  x_noise = make_noise(bank,nsamples)
  return GSM_Neuron(x_train,x_noise,mixer,bank;
    train_noise=train_noise)
end
