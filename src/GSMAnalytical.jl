module GSMAnalytical

export rand

using Statistics, LinearAlgebra, StatsBase
using SpecialFunctions
using Distributions, Random
using QuadGK

include("gsm_distributions.jl")
include("gabor_banks.jl")
include("fit_gsm.jl")

mat1d(x::Real) = let m = Matrix{Float64}(undef,1,1) ; m[1,1] = x ; m ; end



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
