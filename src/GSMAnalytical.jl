module GSMAnalytical

export rand

using Statistics, LinearAlgebra, StatsBase
using SpecialFunctions
using Distributions, Random
using QuadGK


# function tridot(x,M,y)
#   ret = 0.0
#   n,m=size(M)
#   for i in 1:n,j in 1:m
#     ret += x[i]*M[i,j]*y[j]
#   end
#   return ret
# end
tridot(x,M,y)=dot(x,M*y)

mat1d(x::Real) = let m = Matrix{Float64}(undef,1,1) ; m[1,1] = x ; m ; end

abstract type MixerType end
struct RayleighMixer <: MixerType
    alpha
end
struct LogNormalMixer <: MixerType
    μ
    σ
end

function fun_p_nu(mx::RayleighMixer)
    distr = Rayleigh(mx.α)
    function ret(x::R) where R<:Real
        pdf(distr,x)
    end
    return ret
end
function fun_p_nu(mx::LogNormalMixer)
    distr = LogNormal(mx.μ,mx.σ)
    function ret(x::R) where R<:Real
        pdf(distr,x)
    end
    return ret
end

struct GSM{Mx}
    covariance::AbstractMatrix
    covariance_noise::AbstractMatrix
    mixer::Mx
end
Base.Broadcast.broadcastable(gsm::GSM) = Ref(gsm)
Base.ndims(g::GSM) = size(g.covariance,1)
# for now all functions are defined for GSM{RayleighMixer}
# _nn = no noise  _wn = with noise
fun_p_nu(gsm::GSM) = fun_p_nu(gsm.mixer)

@inline function hasnoise(gsm::GSM ; tol=1E-3)
    Sn = gsm.covariance_noise
    n=size(Sn,1)
    return tr(Sn)/n > tol
end
"""
        Random.rand(gsm::GSM{RayleighMixer},n::Integer) -> (mixers,gs,xs)
Generates random samples from the specified GSM. Returns the vector of mixers,
the latent features, and the xs. If noise is abasent, the x samples are the product of
the gs with the mixers.
"""
function Random.rand(gsm::GSM{RayleighMixer},n::Integer)
  hasnoise(gsm) && return _rand_gsmray_wn(gsm,n)
  return _rand_gsmray_nn(gsm,n)
end

function _rand_gsmray_nn(gsm,n)
  mixers = rand(Rayleigh(gsm.mixer.alpha),n)
  gs = rand( MultivariateNormal(gsm.covariance),n)
  xs = broadcast(*,Transpose(mixers),gs)
  return (mixers,gs,xs)
end
function _rand_gsmray_wn(gsm,n)
  (mixers,gs,xs) = _rand_gsmray_nn(gsm,n)
  xs .+= rand(MultivariateNormal(gsm.covariance_noise),n)
  return (mixers,gs,xs)
end

"""
        p_xGnu(x,nu,gsm::GSM) -> p::Float64
Probability of input `x` given the mixer `nu`
"""
function p_xGnu(x,nu,gsm::GSM)
    hasnoise(gsm) && return p_xGnu_wn(x,nu,gsm)
    return p_xGnu_nn(x,nu,gsm)
end
function p_xGnu_nn(x,nu,p::GSM)
    S=(nu*nu+eps(100.)) .* p.covariance #numerical stability!
    pdf(MultivariateNormal(S),x)
 end
function p_xGnu_wn(x,nu,p::GSM)
    S= @. (nu*nu)*p.covariance + p.covariance_noise
    pdf(MultivariateNormal(S),x)
end


# aux functios for the no noise case, analytic
function lambdasq(x,gsm)
    Sg = gsm.covariance
    return dot(x,Sg\x)
end

lambda(x,gsm) = sqrt(lambdasq(x,gsm))

function psibig(k,x,gsm::GSM{RayleighMixer})
   α = gsm.mixer.alpha
   λ = lambda(x,gsm)
   nd = ndims(gsm)
   nn = 1+0.5(k-nd)
   return  (α*λ)^nn *
     besselk( nn , (λ+eps(100.))/α ) / #numerical stability...
     (α*α*sqrt( (2pi)^nd *det(gsm.covariance)))
end

function psibig_ratio(k1,k2,x,gsm::GSM{RayleighMixer})
  α = gsm.mixer.alpha
  λ = lambda(x,gsm)
  nd = ndims(gsm)
  bb = (λ+eps(100.0))/α
  return (α*λ)^(0.5(k1-k2)) *
    besselk(1+0.5(k1-nd),bb) /
    besselk(1+0.5(k2-nd),bb)
end

function psibig_ratio_approx(k,x,gsm::GSM{RayleighMixer},oneterm=true)
  α = gsm.mixer.alpha
  λ = lambda(x,gsm)
  n = ndims(gsm)
  onet = (α*λ)^(0.5k)
  oneterm && return onet
  return onet * ( 1. + ( (n-k)*(n-k-4)+3 )/(8λ)*α )
end

# aux function for the case with nise, semi-analytic
function psibigtilde(k,x,gsm)
  α = gsm.mixer.alpha
  f_integrate(nu) = (nu^k)*pdf(Rayleigh(α),nu)*p_xGnu_wn(x,nu,gsm)
  return quadgk(f_integrate,0,Inf)[1]
end

function meancovar_p_gGxnu(nu::Float64,x::Vector,gsm::GSM)
    @assert hasnoise(gsm)
    n = ndims(gsm)
    Sn_inv = gsm.covariance_noise |> inv
    Sg_inv = gsm.covariance |> inv
    return meancovar_p_gGxnu(nu,x,Sg_inv,Sn_inv)
end

# saves a couple of matrix inversions
function meancovar_p_gGxnu(nu::Float64,x::Vector,Sg_inv,Sn_inv)
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
function p_x(x::AbstractVector,gsm::GSM{RayleighMixer})
  hasnoise(gsm) && return p_x_wn(x,gsm)
  return p_x_nn(x,gsm)
end
p_x_wn(x,gsm) =psibigtilde(0,x,gsm)
p_x_nn(x,gsm) = psibig(0,x,gsm)

function p_x(x::AbstractVector,gsm::GSM{LogNormalMixer})
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
function loglik_data(x::AbstractMatrix,gsm::GSM{RayleighMixer})
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
function p_nuGx(nu::Float64,x::Vector,gsm::GSM{RayleighMixer})
    hasnoise(gsm) && return p_nuGx_wn(nu,x,gsm)
    return p_nuGx_nn(nu,x,gsm)
end

function p_nuGx_nn(nu::Float64,x::Vector,gsm::GSM{RayleighMixer})
  Ψ = psibig(0,x,gsm)
  α = gsm.mixer.alpha
  return  p_xGnu_nn(x,nu,gsm) * pdf(Rayleigh(α),nu) / Ψ
end
function p_nuGx_wn(nu::Float64,x::Vector,gsm::GSM{RayleighMixer})
  Ψ = psibigtilde(0,x,gsm)
  α = gsm.mixer.alpha
  return p_xGnu_wn(x,nu,gsm) * pdf(Rayleigh(α),nu) / Ψ
end

# conditional mean and variance for mixer
"""
    EnuGx(x::Vector,gsm::GSM{RayleighMixer})

Expectation of the mixer for a given input `x`
"""
function EnuGx(x::Vector,gsm::GSM{RayleighMixer})
    hasnoise(gsm) && return EnuGx_wn(x,gsm)
    return EnuGx_nn(x,gsm)
end

function EnuGx_nn(x::Vector,gsm::GSM{RayleighMixer})
    return psibig_ratio(1,0,x,gsm)
end
function EnuGx_wn(x::Vector,gsm::GSM{RayleighMixer})
    return psibigtilde(1,x,gsm)/psibigtilde(0,x,gsm)
end
# high-values approximation
function EnuGx_nn_approx(x::Vector,gsm::GSM{RayleighMixer} ; oneterm=true)
    return psibig_ratio_approx(1,x,gsm,oneterm)
end

"""
    Var_nuGx(x::Vector,gsm::GSM{RayleighMixer})

Variance of the mixer for a given input `x`
"""
function Var_nuGx(x::Vector,gsm::GSM{RayleighMixer})
    hasnoise(gsm) && return Var_nuGx_wn(x,gsm)
    return Var_nuGx_nn(x,gsm)
end
function Var_nuGx_nn(x::Vector,gsm::GSM{RayleighMixer})
    n = ndims(gsm)
    nn = n-1
    return psibig_ratio(2,0,x,gsm) - (psibig_ratio(1,0,x,gsm))^2
end
function Var_nuGx_wn(x::Vector,gsm::GSM{RayleighMixer})
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
function p_giGx(gk::Float64,i::Integer,x::Vector,gsm::GSM{RayleighMixer})
    hasnoise(gsm) && return p_gkGx_wn(gk,i,x,gsm)
    return p_gkGx_nn(gk,i,x,gsm)
end
function p_gkGx_nn(gk,k,x,gsm)
    n = ndims(gsm)
    α = gsm.mixer.alpha
    S = gsm.covariance
    Ψ = psibig(0,x,gsm)
    nu = x[k]/gk
    g = x ./ nu # this replaces gk also
    return pdf(Rayleigh(α),nu) / ( Ψ * (nu^(n-1)*abs(gk)) ) *
        pdf(MultivariateNormal(S),g)
end
function p_gkGx_wn(gk,k,x,gsm)
    n = ndims(gsm)
    α = gsm.mixer.alpha
    Sg = gsm.covariance
    Sn = gsm.covariance_noise
    Sg_inv = inv(Sg)
    Sn_inv = inv(Sn)
    Ψ = psibigtilde(0,x,gsm)
    function f(nu)
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
function EgiGx(i::Integer,x::Vector,gsm::GSM{RayleighMixer})
    hasnoise(gsm) && return EgiGx_wn(i,x,gsm)
    return EgiGx_nn(i,x,gsm)
end

# expect_g1(x,gsm) = x[1]*psi_ratio(1,x,gsm)
# variance_g1(x,gsm) =  (x[1]^2)*(psi_ratio(2,x,gsm)-psi_ratio(1,x,gsm)^2)
function EgiGx_nn(i,x,gsm)
    return x[i] * psibig_ratio(-1,0,x,gsm)
end
function EgiGx_wn(i,x,gsm)
    n = ndims(gsm)
    α = gsm.mixer.alpha
    Sg = gsm.covariance
    Sn = gsm.covariance_noise
    Sg_inv = inv(Sg)
    Sn_inv = inv(Sn)
    Ψ = psibigtilde(0,x,gsm)
    function f(nu)
      (mu3,S3) = meancovar_p_gGxnu(nu,x,Sg_inv,Sn_inv)
      return pdf(Rayleigh(α),nu) * p_xGnu_wn(x,nu,gsm) * mu3[i]
    end
    return quadgk(f,0,Inf)[1] / Ψ
end

function EgiGx_nn_approx(i,x::Vector,gsm::GSM{RayleighMixer} ; oneterm=true)
    return x[i]*psibig_ratio_approx(-1,x,gsm,oneterm)
end
# expectation for g square !

function Egi_sqGx_nn(i,x,gsm)
    return (x[i])^2 * psibig_ratio(-2,0,x,gsm)
end
function Egi_sqGx_wn(i,x,gsm)
    n = ndims(gsm)
    α = gsm.mixer.alpha
    Sg = gsm.covariance
    Sn = gsm.covariance_noise
    Sg_inv = inv(Sg)
    Sn_inv = inv(Sn)
    Ψ = psibigtilde(0,x,gsm)
    function f(nu)
      (mu3,S3) = meancovar_p_gGxnu(nu,x,Sg_inv,Sn_inv)
      return pdf(Rayleigh(α),nu) * p_xGnu_wn(x,nu,gsm) * (mu3[i]^2 + S3[i,i])
    end
    return quadgk(f,0,Inf)[1] / Ψ
end

function Egi_sqGx_nn_approx(i,x::Vector,gsm::GSM{RayleighMixer} ; oneterm=true)
    return (x[i])^2*psibig_ratio_approx(-2,x,gsm,oneterm)
end

"""
        Var_giGx(i::Integer,x::Vector,gsm::GSM{RayleighMixer}) -> mu_gi::Float64
Variance for the `i`th feature (the `ith` component of the full `g`)
given the input `x`. Usually k=1
"""
function Var_giGx(i::Integer,x::Vector,gsm::GSM{RayleighMixer})
    hasnoise(gsm) && return Var_giGx_wn(i,x,gsm)
    return Var_giGx_nn(i,x,gsm)
end
function Var_giGx_nn(i,x,gsm)
    return  (x[i]^2)*(psibig_ratio(-2,0,x,gsm)-psibig_ratio(-1,0,x,gsm)^2)
end
function Var_giGx_wn(i,x,gsm)
    E = EgiGx_wn(i,x,gsm)
    Esq = Egi_sqGx_wn(i,x,gsm)
    return Esq - E^2
end
function Var_giGx_nn_approx_old(i,x,gsm::GSM{RayleighMixer})
    n = ndims(gsm)
    α = gsm.mixer.alpha
    λ = lambda(x,gsm)
    return  (x[i]/λ)^2 *(4n-n*n-1)/8
    # E = EgiGx_nn_approx(i,x,gsm; oneterm=false)
    # Esq = Egi_sqGx_nn_approx(i,x,gsm; oneterm=false)
    # return Esq - E^2
end
function Var_giGx_nn_approx(i,x,gsm::GSM{RayleighMixer})
    n = ndims(gsm)
    α = gsm.mixer.alpha
    λ = lambda(x,gsm)
    return  0.25*(x[i]/λ)^2
end

function Var_giGx_nn_approx_bis(i,x,gsm::GSM{RayleighMixer})
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
function FFgiGx(i::Integer,x::Vector,gsm::GSM{RayleighMixer})
    hasnoise(gsm) && return FFgiGx_wn(i,x,gsm)
    return FFgiGx_nn(i,x,gsm)
end
function FFgiGx_nn(i,x,gsm)
  psi1 = psibig_ratio(-1,0,x,gsm)
  psi2 = psibig_ratio(-2,0,x,gsm)
  return x[i]*(psi2/psi1-psi1)
end
function FFgiGx_wn(i,x,gsm)
    E = EgiGx_wn(i,x,gsm)
    Esq = Egi_sqGx_wn(i,x,gsm)
    return Esq/E - E
end

function FFgiGx_nn_approx(i,x,gsm)
    n = ndims(gsm)
    α = gsm.mixer.alpha
    λ = lambda(x,gsm)
    return  x[i]*sqrt(α)/(4.0*λ*sqrt(λ))
end

function FFgiGx_nn_approx_alt(i,x,gsm;oneterm=true)
  psi1 = psibig_ratio_approx(-1,x,gsm,oneterm)
  psi2 = psibig_ratio_approx(-2,x,gsm,oneterm)
  return x[i]*(psi2/psi1-psi1)
end


# Now the same, but for multiple elements of g.
# must be the case with noise only

"""
    p_gGx(g::Vector,idxs::Vector,x::Vector,gsm::GSM{RayleighMixer})

"""
function p_gGx(g::Vector,idxs::Vector,x::Vector,gsm::GSM{RayleighMixer})
   @assert hasnoise(gsm)
   @assert length(g) == length(idxs)
   n = ndims(gsm)
   α = gsm.mixer.alpha
   Sg = gsm.covariance
   Sn = gsm.covariance_noise
   Sg_inv = inv(Sg)
   Sn_inv = inv(Sn)
   Ψ = psibigtilde(0,x,gsm)
   function f(nu)
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
function EgGx(idxs::Vector,x::Vector,gsm::GSM{RayleighMixer})
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

function EgigjGx(idx_i::Integer,idx_j::Integer,x::Vector,gsm::GSM{RayleighMixer})
    @assert hasnoise(gsm)
    n = ndims(gsm)
    α = gsm.mixer.alpha
    Sg = gsm.covariance
    Sn = gsm.covariance_noise
    Sg_inv = inv(Sg)
    Sn_inv = inv(Sn)
    Ψ = psibigtilde(0,x,gsm)
    function f(nu)
        (mu3,S3) = meancovar_p_gGxnu(nu,x,Sg_inv,Sn_inv)
        factor_ij = S3[idx_i,idx_j] + (mu3[idx_i]*mu3[idx_j])
        return pdf(Rayleigh(α),nu) * p_xGnu_wn(x,nu,gsm) * factor_ij
    end
    return quadgk(f,0,Inf)[1] / Ψ
end

function Pearson_gigjGx(idx_i::Integer,idx_j::Integer,x::Vector,gsm::GSM{RayleighMixer})
    @assert hasnoise(gsm)
    Egi,Egj = EgGx([idx_i,idx_j],x,gsm)
    EgijGx = EgigjGx(idx_i,idx_j,x,gsm)
    vgi = Var_giGx(idx_i,x,gsm)
    vgj = Var_giGx(idx_j,x,gsm)
    return (EgijGx - Egi*Egj)/sqrt(vgi*vgj)
end

end #module

    # function p_nuGx(nus::Vector,x,gsm;normalized=true)
    #     mixer_pdf=get_mixer_pdf(gsm)
    #     n=ndims(gsm)
    #     px= normalized ? p_x(x,gsm) : 1.0
    #     map(nu-> p_xGnu(nu,x,gsm)*mixer_pdf(nu)/px , nus)
    # end
    # function p_nuGx(nus::Real,x,gsm;normalized=true)
    #     p_nuGx([nus],x,gsm;normalized=normalized)[1]
    # end

    # expect_nu(x,gsm) = psi_ratio(-1,x,gsm)
    # variance_nu(x,gsm) = psi_ratio(-2,x,gsm)-psi_ratio(-1,x,gsm)^2
##################################
# ### old stuff to update
#
#
#
# function lambda_from_x(x::Vector{Float64},gsm::GSM)
#   #@assert length(x)==size(gsm.convariance_matrix,1) "the vector has the wrong length!"
#   dot(gsm.covariance_matrix\x,x)
# end
# tolambda(x::Vector,gsm) = lambda_from_x(x,gsm)
#
# function psi_ray(n,lam_orx,gsm)
#   lam = tolambda(lam_orx)
#   alpha_ray_sq=gsm.mixer_prior_gsms[1]^2
#   (alpha_ray_sq*lam)^((2-n)/4.0) *
#    besselk((2-n)/2,sqrt(lam/alpha_ray_sq)) /
#    (alpha_ray_sq * sqrt((2pi)^gsm.ndims *det(gsm.covariance_matrix)))
# end
#
# function psi_ray_ratio(k,lam_orx,gsm)
#     lam=tolambda(lam_orx)
#     n=gsm.ndims
#     alpha_ray_sq=gsm.mixer_prior_gsms^2
#     bb = sqrt(lam/alpha_ray_sq)
#     (alpha_ray_sq*lam)^(-k/4.0) *
#     besselk((2-n-k)/2.0,bb) /
#     besselk((2-n)/2.0,bb)
# end
#
# function psi_ray_ratio(lam::Float64,gsm)
#         psi_ray_ratio(1,lam,par)
# end
#
# function psi_halfG(n,lam_orx,par)
#   lam=tolambda(lam_orx)
#   sigma_noise=par.mixer_prior_pars^2
#   (sigma_noise*lam)^((1-n)/4.0) *
#    besselk((1-n)/2,sqrt(lam/sigma_noise))*2.0/
#    (sqrt(sigma_noise*(2pi)^(par.n+1) *det(par.covariance_matrix)))
# end
#
# function psi_halfG_ratio(k,lam_orx,par)
#     lam=tolambda(lam_orx)
#     n=par.ndims
#     sigma_noise=par.mixer_prior_pars^2
#     bb = sqrt(lam/sigma_noise)
#     (sigma_noise*lam)^(-k/4.0) *
#     besselk((1-n-k)/2.0,bb) /
#     besselk((1-n)/2.0,bb)
# end
#
# psi(nn,lam_orx,gsm::GSM{RayleighMixer}) = psi_ray(nn,lam_orx,gsm)
# psi_ratio(nn,lam_orx,gsm::GSM{RayleighMixer}) = psi_ray_ratio(nn,lam_orx,gsm)
#
# expect_nu(x,gsm) = psi_ratio(-1,x,gsm)
# variance_nu(x,gsm) = psi_ratio(-2,x,gsm)-psi_ratio(-1,x,gsm)^2
#
# function checknoise(gsm::GSM)
#     _noise = gsm.covariance_matrix_noise
#     n=size(_noise,1)
#     _tr = tr(_noise)/n
#     return (_tr > 1E-4)
# end
#
# """
# p_xGnu(nu::Float64,x::Vector,gsm_model)
#
# Probability of x given the mixer P(x | nu )
# """
# function p_xGnu(nu,x,gsm)
#     checknoise(gsm) ? _p_xGnu_noise(nu,x,gsm) : _p_xGnu_nonoise(nu,x,gsm)
# end
#
# function _p_xGnu_nonoise(nu,x,gsm)
#   n=ndims(gsm)
#   exppartt=exp(-0.5dot(gsm.covariance_matrix\x,x)/nu^2)
#   denom=sqrt((2pi)^n*det(gsm.covariance_matrix)) * nu^n
#   exppartt/denom
# end
# function _p_xGnu_noise(nu,x,gsm)
#     Sg=gsm.covariance_matrix
#     Sn=gsm.covariance_matrix_noise
#     d=MultivariateNormal(nu^2*Sg+Sn)
#     pdf(d,x)
# end
#
# function p_x1Gnu(nu,x1,gsm)
#     s_g=gsm.covariance_matrix[1,1]
#     s_n= checknoise(gsm) ? gsm.covariance_matrix_noise[1,1] : 0.0
#     s = sqrt( nu^2 * s_g + s_n )
#     pdf(Normal(0.0,s),x1)
# end
# function p_x1Gg1nu(nu,x1,g1,gsm)
#     @assert checknoise(gsm) "a model with no noise should not use this!"
#     s_n= sqrt(gsm.covariance_matrix_noise[1,1])
#     mu=g1*nu
#     pdf(Normal(mu,s_n),x1)
# end
#
# function p_nuGx(nus::Vector,x,gsm;normalized=true)
#     mixer_pdf=get_mixer_pdf(gsm)
#     n=ndims(gsm)
#     px= normalized ? p_x(x,gsm) : 1.0
#     map(nu-> p_xGnu(nu,x,gsm)*mixer_pdf(nu)/px , nus)
# end
# function p_nuGx(nus::Real,x,gsm;normalized=true)
#     p_nuGx([nus],x,gsm;normalized=normalized)[1]
# end
#
# """
#  p_g1Gx(g1,x,model; normalized=true)
#
# """
# function p_g1Gx(g1,x,gsm; normalized=true)
#     if checknoise(gsm)
#        _p_g1Gx_noise(g1,x,gsm)
#     else
#        _p_g1Gx_nonoise(g1,x,gsm;normalized=normalized)
#     end
# end
# function _p_g1Gx_nonoise(g1::Float64,x,gsm; normalized=true)
#   mixer_pdf=get_mixer_pdf(gsm)
#   n=ndims(gsm)
#   px= normalized ? p_x(x,gsm) : 1.0
#   my_mult=MultivariateNormal(gsm.covariance_matrix)
#   feat_pdf(g)=pdf(my_mult,g)
#   x1=x[1]
#   nu=x1/g1
#   g_all=x./nu
#   g_all[1]=g1
#   abs(g1^(n-2)/x1^(n-1)) * mixer_pdf(nu) * feat_pdf(g_all) / px
# end
# function _p_g1Gx_nonoise(g1s::Vector,x,gsm; normalized=true)
#     println("Warning: the function p_g1Gx is not really optimized for vectors")
#     map(g1-> _p_g1Gx_nonoise(g1,x,gsm;normalized=normalized), g1s)
# end
#
# function _p_g1Gx_noise_old(g1s,x,gsm)
#     sigma_g11=sqrt(par.covariance_matrix[1,1])
#     dist_g1=Normal(0.0,sigma_g11)
#     x1=x[1]
#     px=p_x(x,par)
#     map(g1s) do g1
#         to_integrate(nu)=p_nuGx(nu,x,par;normalized=false)*p_x1Gg1nu(nu,x1,g1,par) / p_x1Gnu(nu,x1,par)
#         _integral=hquadrature(to_integrate,1E-5,10)[1]
#         if _integral > 0
#             _integral * pdf.(dist_g1,g1) / px
#         else
#             0
#         end
#     end
# end
#
# function _p_g1Gx_noise(g1s,x,par)
#     px=p_x(x,par)
#     map(g1s) do g1
#         to_integrate(nu)=p_nuGx(nu,x,par;normalized=false)/px*
#                 p_g1Gnux(g1,nu,x,par)
#         _integral=quadgk(to_integrate,1E-5,10)[1]
#         @assert isfinite(_integral)
#         @assert _integral > 0
#         _integral
#     end
#     #     if _integral > 0
#     #         _integral
#     #     else
#     #         _integral=pquadrature(to_integrate,1E-5,10)[1]
#     #         if _integral > 0
#     #         _integral / px
#     #         else
#     #         0
#     #     end
#     #     end
#     # end
# end
# """
# p_g1nuGx(nu,g1,x::Vector,model)
#
# this is the  P(g1,nu | x ) with noise present
# for now nu and g1 are scalars
# """
# function p_g1nuGx(nu,g1,x,par)
#     @assert par.with_noise "without noise, it's just a delta!"
#     sigma_g11=sqrt(par.covariance_matrix[1,1])
#     dist_g1=Normal(0.0,sigma_g11)
#     x1=x[1]
#     p_nuGx(nu,x,par)*p_x1Gg1nu(nu,x1,g1,par) /
#         p_x1Gnu(nu,x1,par)*pdf(dist_g1,g1)
# end
#
# """
# Using my analytix results on mu3 S3
# """
# function mean_var_gGnux(nu,x,par)
#     S_g=par.covariance_matrix
#     S_n=par.covariance_matrix_noise
#     S_n_inv=inv(S_n)
#     nusq=nu*nu
#     S3 = inv(inv(S_g) + S_n_inv*nusq ) # I should probably make this positive definite!
#     mu3 = ( (S3*S_n_inv)*x ) .*nu
#     mu3,S3
# end
# function p_gGnux(g1,nu,x,par)
#     mu3,S3 = mean_var_gGnux(nu,x,par)
#     distr3=MultivariateNormal(mu3,S3)
#     pdf.(distr3,g1)
# end
# function p_g1Gnux(g1,nu,x,par)
#     mu3,S3 = mean_var_gGnux(nu,x,par)
#     distr3_11=Normal(mu3[1],sqrt(S3[1,1]))
#     pdf.(distr3_11,g1)
# end
