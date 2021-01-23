using Pkg
Pkg.activate(".")
using GSMAnalytical ; const G = GSMAnalytical
using Test

using Random,LinearAlgebra,Statistics,StatsBase,Distributions
using Calculus
using Plots ; theme(:dark)
using QuadGK

using Optim


function do_gsm1D(cov,noise,mx)
  cov = G.mat1d(cov)
  noise = G.mat1d(noise)
  mx = G.RayleighMixer(mx)
  return G.GSM(cov,noise,mx)
end

function do_gsm2D(cov::R,noise::R,mx::R) where R<:Real
  Σg = [ 1. cov
          cov 1.]
  Σnoise =zeros(2,2) + noise*I
  mx = G.RayleighMixer(mx)
  G.GSM(Σg,Σnoise,mx)
end

function plotvs(x::AbstractArray{<:Real},y::AbstractArray{<:Real})
  x,y=x[:],y[:]
  @info """
  The max differences between the two are $(extrema(x .-y ))
  """
  plt=plot()
  scatter!(plt,x,y;leg=false,ratio=1,color=:white)
  lm=xlims()
  plot!(plt,identity,range(lm...;length=3);linestyle=:dash,color=:yellow)
  return plt
end

function tinydotgrad(x,y,Σ)
  n=size(Σ,1)
  ch=cholesky(Σ)
  L=ch.L
  fun_grad=function(Lij::Real,i::Integer,j::Integer)
    Lpre = L[i,j]
    L[i,j]=Lij
    iΣnew = inv(L*L')
    _ret = dot(x,iΣnew,y)
    L[i,j]=Lpre
    return _ret
  end
  ret = zeros(size(Σ)...)
  for j in 1:n, i in j:n
    ret[i,j] = Calculus.gradient( Lij->fun_grad(Lij,i,j),L[i,j])
  end
  return ret
end


function tinydotgradprime(x::Vector{R},y::Vector{R},Σ::AbstractMatrix{R}) where R
  n=size(Σ,1)
  iΣ=inv(Σ)
  ch=cholesky(Σ)
  L=ch.L
  ret = zeros(size(Σ)...)
  lf, rg = x'*iΣ, iΣ*y
  ijgrad=zero(ret)
  for j in 1:n, i in j:n
    fill!(ijgrad,0.0)
    ijgrad[i,:] .= L[:,j]
    ijgrad[:,i] .+= L[:,j]
    ret[i,j] = - lf * ijgrad * rg
  end
  return ret
end

function gradient_for_em(x::AbstractArray{<:Real},
    gsum::G.GSuM{G.NormalMixer{R},R}) where R
  n=G.n_dims(gsum)
  ch=cholesky(gsum.covariance)
  L=ch.L
  fun_grad=function(Lij::Real,i::Integer,j::Integer)
    Lpre = L[i,j]
    L[i,j]=Lij
    gsum.covariance .= L*L'
    _ret = G.conditional_expectation_EM_test(x,gsum)
    L[i,j]=Lpre
    gsum.covariance .= L*L'
    return _ret
  end
  ret_num = zeros(size(gsum.covariance)...)
  for j in 1:n, i in j:n
    ret_num[i,j] = Calculus.gradient( Lij->fun_grad(Lij,i,j),L[i,j])
  end
  return ret_num
end

function grad_stars_single_num(x::AbstractArray{<:Real},
    gsum::G.GSuM{G.NormalMixer{R},R},dosigma::Bool=true) where R
  n=G.n_dims(gsum)
  ch=cholesky(gsum.covariance)
  L=ch.L
  fun_grad=function(Lij::Real,i::Integer,j::Integer)
    Lpre = L[i,j]
    L[i,j]=Lij
    gsum.covariance .= L*L'
    μstar,σstar = G.pars_p_nuGx(x,gsum)
    _ret = dosigma ? σstar^2 : μstar^2+σstar^2
    L[i,j]=Lpre
    gsum.covariance .= L*L'
    return _ret
  end
  ret_num = zeros(size(gsum.covariance)...)
  for j in 1:n, i in j:n
    ret_num[i,j] = Calculus.gradient( Lij->fun_grad(Lij,i,j),L[i,j])
  end
  return ret_num
end

function grad_sigmastarsq_ana(x::AbstractArray{<:Real},
    gsum::G.GSuM{G.NormalMixer{R},R}) where R
  @assert !G.hasnoise(gsum) "must have no noise"
  μstar,σstar=G.pars_p_nuGx(x,gsum)
  Σg = gsum.covariance
  id=fill(1.0,size(Σg,1))
  dL = tinydotgradprime(id,id,Σg)
  dL .*= -σstar^4
  return dL
end
function grad_starsqsums_ana(x::AbstractArray{<:Real},
    gsum::G.GSuM{G.NormalMixer{R},R}) where R
  μstar,σstar=G.pars_p_nuGx(x,gsum)
  dL1 = grad_sigmastarsq_ana(x,gsum)
  Σg = gsum.covariance
  id=fill(1.0,size(Σg,1))
  dL2 = tinydotgradprime(id,x,Σg)
  return @. 2.0 * (μstar^2/σstar^2 * dL1 + σstar^2*μstar*dL2)+dL1
end



function EMfit_test(x::AbstractArray{<:Real},gsum::G.GSuM{G.NormalMixer{R},R}) where R
  μstar,σstar=G.EMfit_Estep(x,gsum)
  # analytic
  grad_an=G.EMfit_Mstep_costprime(μstar,σstar,x,gsum)
  # numeric
  Σ=gsum.covariance
  Σ0=copy(Σ)
  n=size(Σ,1)
  L=cholesky(Σ).L
  μstar,σstar = G.pars_p_nuGx(x,gsum)
  fun_grad=function(Lij::Real,i::Integer,j::Integer)
    Lpre = L[i,j]
    L[i,j]=Lij
    copy!(Σ,L*L')
    _ret = G.EMfit_Mstep_cost(μstar,σstar,x,gsum)
    L[i,j]=Lpre
    copy!(Σ,Σ0)
    return _ret
  end
  grad_num = zeros(size(Σ)...)
  for j in 1:n, i in j:n
    grad_num[i,j] = Calculus.gradient( Lij->fun_grad(Lij,i,j),L[i,j])
  end
  return grad_num[:],grad_an[:]
end
function EMfit_test_many(xs::AbstractMatrix{<:Real},
    gsum::G.GSuM{G.NormalMixer{R},R}) where R
  μstar,σstar=G.EMfit_Estep(xs,gsum)
  # analytic
  grad_an=G.EMfit_Mstep_costprime(μstar,σstar,xs,gsum)
  # numeric
  Σ=gsum.covariance
  Σ0=copy(Σ)
  n=size(Σ,1)
  L=cholesky(Σ).L
  μstar,σstar = G.pars_p_nuGx(xs,gsum)
  fun_grad=function(Lij::Real,i::Integer,j::Integer)
    Lpre = L[i,j]
    L[i,j]=Lij
    copy!(Σ,L*L')
    _ret = G.EMfit_Mstep_cost(μstar,σstar,xs,gsum)
    L[i,j]=Lpre
    copy!(Σ,Σ0)
    return _ret
  end
  grad_num = zeros(size(Σ)...)
  for j in 1:n, i in j:n
    grad_num[i,j] = Calculus.gradient( Lij->fun_grad(Lij,i,j),L[i,j])
  end
  return grad_num[:],grad_an[:]
end



function EMFit_Mstep_optim(μstar::Vector{R},σstar::Vector{R},
    xs::AbstractMatrix{<:Real},gsum::G.GSuM{G.NormalMixer{R},R}) where R
  Σ=gsum.covariance
  L=cholesky(Σ).L
  Lv0 = L[:]
  costfun = function (Lv::Vector{R})
    L=reshape(Lv,n,n)
    copy!(Σ,L*L')
    return - G.EMfit_Mstep_cost(μstar,σstar,xs,gsum)
  end
  gradfun! = function (grad::Vector{R},Lv::Vector{R})
    L=reshape(Lv,n,n)
    copy!(Σ,L*L')
    gradMat= G.EMfit_Mstep_costprime(μstar,σstar,xs,gsum)
    copy!(grad,.- gradMat[:])
    return  grad
  end
  alg=ConjugateGradient() # BFGS()
  res=optimize(costfun, gradfun!, Lv0, alg,
    Optim.Options(show_every=5))
  Lout=reshape(Optim.minimizer(res),n,n)
  return Lout*Lout',res
end

##
# make random GSuM
n=12
Σg = convert(Matrix{Float64},G.random_covariance_matrix(n,3.))
Σnoise = zero(Σg)
mixer=G.NormalMixer(0.44,0.890)
gsum = G.GSuM(Σg,Σnoise,mixer)

xstry = let n=300,
  (mx,g,x)=G.rand(gsum,n)
  x
end
##

μstar,σstar=G.EMfit_Estep(xstry,gsum)
plotvs(EMfit_test_many(xstry,gsum)...)
plotvs(EMfit_test(xstry[:,1],gsum)...)

Σfit,result=EMFit_Mstep_optim(μstar,σstar,xstry,gsum)

plotvs(Σfit,Σg)

##
# make random GSuM
n=10
Σg = convert(Matrix{Float64},G.random_covariance_matrix(n,3.))
Σg0=copy(Σg)
Σg2 = convert(Matrix{Float64},G.random_covariance_matrix(n,3.))
Σnoise = zero(Σg)
mixer=G.NormalMixer(0.44,0.890)
gsum = G.GSuM(Σg,Σnoise,mixer)

xstry = let n=1_000,
  (mx,g,x)=G.rand(gsum,n)
  x
end
copy!(gsum.covariance,cov(xstry;dims=2))
for i in 1:10
  @show i
  μstar,σstar=G.EMfit_Estep(xstry,gsum)
  Σfit,result=EMFit_Mstep_optim(μstar,σstar,xstry,gsum)
  copy!(Σg,Σfit)
end
##
plotvs(Σfit,Σg0)


##
##
G.conditional_expectation_EM_prime(xtry,gsum)

gradient_for_em(xtry,gsum)

plotvs(gradient_for_em(xtry,gsum), G.conditional_expectation_EM_prime(xtry,gsum) )



logdetprime_num(Σg)
logdetprime(Σg)

plotvs(logdetprime_num(Σg),logdetprime(Σg))

##
# compare analytic with numberic
G.conditional_expectation_thingy(xtry,gsum)

# repeat for lots of xs
G.conditional_expectation_EM(xtry,gsum)

buh=gradient_for_em(xtry,gsum)
heatmap(buh)


Σtry=copy(gsum.covariance)
xtry=randn(n)
ytry=randn(n)
plotvs(tinydotgrad(xtry,ytry,Σtry),tinydotgradprime(xtry,ytry,Σtry))
tinydotgrad(xtry,ytry,Σtry)
tinydotgradprime(xtry,ytry,Σtry)

plotvs(tinydotgrad(xtry,ytry,Σtry),tinydotgradprime(xtry,ytry,Σtry))

##
plotvs(grad_stars_single_num(xtry,gsum),grad_sigmastarsq_ana(xtry,gsum))

grad_stars_single_num(xtry,gsum,false)
grad_starsqsums_ana(xtry,gsum)
fill!(xtry,0.0)
plotvs(grad_stars_single_num(xtry,gsum,false),grad_starsqsums_ana(xtry,gsum))



##
# fit GSiM 1D

cov,noise = 3.0 , 0.0
cov = G.mat1d(cov)
noise = G.mat1d(noise)
mx = G.NormalMixer(0.5,0.3)
gsum_true = G.GSuM(cov,noise,mx)

x_train,x_noise = let (mx,gs,xs) = rand(gsum_true,15_000)
  x_nn = broadcast(+,mx',gs)
  noise = xs-x_nn
  @assert all(isapprox.(x_nn+noise,xs))
  (x_nn,noise)
end

gsum_train=G.fit_given_mixer_and_noise(G.GSuM,x_train,noise,mx)

gsum_train.covariance_noise

plotvs(gsum_true.covariance,gsum_train.covariance)
plotvs(gsum_true.covariance_noise,gsum_train.covariance_noise)

##

cov_mat = [ 1.0  -0.22  0.2
           -0.22 1.456  0.11
            0.2  0.11   0.7778 ]
cov_noise = [ 0.5  0.01  -0.07
              0.01  0.4   -0.08
             -0.07 -0.08  0.234 ]
mx = G.RayleighMixer(1.345)
gsm_true = G.GSM(cov_mat,cov_noise,mx)
x_train,x_noise = let all = rand(gsm_true,15_000)
  gs = all[2]
  x_nn = broadcast(*,all[1]',gs)
  noise = all[3]-x_nn
  (x_nn,noise)
end
bank_test = G.GaborBank(G.SameSurround(1,2), 121,20,5,4)
gsm_train = G.GSM_Neuron(x_train,x_noise,mx,bank_test;test_bank=false).gsm
@test all(
 isapprox.(gsm_train.covariance, gsm_true.covariance;atol=0.033))
@test all(
  isapprox.(gsm_train.covariance_noise, gsm_true.covariance_noise;atol=0.033))

gsm_train2=G.fit_given_mixer_and_noise(G.GSM,x_train,gsm_train.covariance_noise,mx)
@test all(isapprox.(gsm_train.covariance,gsm_train2.covariance))
@test all(isapprox.(gsm_train.covariance_noise,gsm_train2.covariance_noise))

# now the GSuM
mx = G.NormalMixer(0.345,0.2)
gsum_true = G.GSuM(cov_mat,cov_noise,mx)
x_train,x_noise = let (mx,gs,xs) = rand(gsum_true,15_000)
  x_nn = broadcast(+,mx',gs)
  noise = xs-x_nn
  @assert all(isapprox.(x_nn+noise,xs))
  (x_nn,noise)
end

gsum_train=G.fit_given_mixer_and_noise(G.GSuM,x_train,cov(x_noise;dims=1),mx)

gsum_train.covariance_noise

plotvs(gsum_true.covariance,gsum_train.covariance)
plotvs(gsum_true.covariance_noise,gsum_train.covariance_noise)

@test all(
 isapprox.(gsum_train.covariance, gsum_true.covariance;atol=0.033))
@test all(
  isapprox.(gsum_train.covariance_noise, gsum_true.covariance_noise;atol=0.033))

@test all(isapprox.(gsum_train.covariance,gsum_train2.covariance))
@test all(isapprox.(gsum_train.covariance_noise,gsum_train2.covariance_noise))




##

bank_test = G.GaborBank(G.SameSurround(1,8), 91,12,5,10)
bank_view = G.show_bank(bank_test; indexes=[1,(3:2:ndims(bank_test))...])
heatmap(bank_view;ratio=1,c=:grays)

# train just with noise
ntrain=10_000
noise_patches = let n = bank_test.frame_size
  rand(n,n,ntrain)
end

gsm_neu = G.GSM_Neuron(noise_patches,0.01,G.RayleighMixer(0.4876),bank_test)

G.mean_std(gsm_neu.gsm.covariance_noise)/G.mean_std(gsm_neu.gsm.covariance)

heatmap(gsm_neu.gsm.covariance; ratio=1)
heatmap(gsm_neu.gsm.covariance_noise; ratio=1)
mean(gsm_neu.gsm.covariance ./ gsm_neu.gsm.covariance_noise)
scatter(gsm_neu.gsm.covariance[:],gsm_neu.gsm.covariance_noise[:]; leg=false  )
cor(gsm_neu.gsm.covariance[:],gsm_neu.gsm.covariance_noise[:])
##

cov_mat = [ 1.0  -0.22  0.2
            -0.22 1.456  0.11
             0.2  0.11   0.7778 ]
cov_noise = [ 0.5  0.01  -0.07
             0.01  0.4   -0.08
            -0.07 -0.08  0.234 ]

##
#  first test, generate from random 1D gsm, and take a look at x


gsm = do_gsm(0.333,0.0,0.2334)
mx,gs,xs =  rand(gsm,5_000)
histogram(xs[1,:] ; nbins=80, normed=true, leg=false)

# second test p(x) integrates to 1
_ = let f(x) = G.p_x([x,],gsm)
   r = quadgk(f,-Inf,Inf)[1]
   @info "integral is $r"
 end

# third test: samples Vs curve

gsm = do_gsm(0.333,0.0,0.2334)
_ =  let (mx,gs,xs) =  rand(gsm,10_000),
  f(x) = G.p_x([x,],gsm),
  h = normalize(fit(Histogram,xs[1,:]; nbins=101))
  binsc = midpoints(collect(h.edges[1]))
  plot(h ; label="samples")
  plot!(binsc, f ; linewidth=5,opacity=0.8, color="red", label="analytic")
end

## all good ! now p( nu | x), still 1 D

gsm = do_gsm(0.333,0.987,0.2334)
# must integrate to 1 in nu for a given x
_ = let x=0.9 ,
   f(nu) = G.p_nuGx(nu,[x,],gsm)
   r = quadgk(f,0,Inf)[1]
   @info "integral is $r"
 end

#
##
# test approximate moments Vs analytic ones

gsm = do_gsm(0.333,0,0.2334)

moment_an(x) = G.EnuGx_nn([x,],gsm)
moment_approx(x) = G.EnuGx_nn_approx([x,],gsm)
moment_approx_more(x) =  G.EnuGx_nn_approx([x,],gsm ; oneterm=false)

xplot = range(0.01,1 ; length=300) |> collect


plot(xplot, [moment_an, moment_approx,moment_approx_more] ;
 label=["full" "approx" "approx 2 terms"], legend=:bottom , linewidth=3  )

# repeat for g1

gsm = do_gsm(0.833,0,1.333)

moment_an(x) = G.EgiGx_nn(1,[x,],gsm)
moment_approx(x) = G.EgiGx_nn_approx(1,[x,],gsm)
moment_approx_more(x) =  G.EgiGx_nn_approx(1,[x,],gsm ; oneterm=false)

xplot = range(0.01,1 ; length=300) |> collect

plot(xplot, [moment_an, moment_approx,moment_approx_more] ;
 label=["full" "approx" "approx 2 terms"], legend=:bottom , linewidth=3  )

# repeat for Variance !
gsm = do_gsm(0.833,0,1.333)
variance_an(x) = G.Var_giGx_nn(1,[x,],gsm)
variance_approx(x) = G.Var_giGx_nn_approx(1,[x,],gsm)

xplot = range(0.01,30 ; length=300) |> collect


plot(xplot, [variance_an, variance_approx] ;
 label=["full" "approx" ], legend=:bottom , linewidth=3  )
# in 1D the approx variance does not depend on the
# input ! D'hu

# last one, fano factor

gsm = do_gsm(0.833,0,1.333)
ff_an(x) = G.FFgiGx_nn(1,[x,],gsm)
ff_approx(x) = G.FFgiGx_nn_approx(1,[x,],gsm)

xplot = range(0.01,30 ; length=300) |> collect
plot(xplot, [ff_an, ff_approx] ;
 label=["full" "approx" ], legend=:bottom , linewidth=3  )
##
# ok, let's try in 3D

gsm = let α = 2.123
  cov = [ 1  0.1   0
          0.1  1  -0.1
           0   -0.1  1 ]
  noise = 0 .* cov
  G.GSM(cov,noise,G.RayleighMixer(α))
end

to_vec(x) = [1.1x - 3,sqrt(10x) , log(x)]
##
moment_an(x) = G.EnuGx_nn(to_vec(x),gsm)
moment_approx(x) = G.EnuGx_nn_approx(to_vec(x),gsm ; oneterm=true)
moment_approx_more(x) =  G.EnuGx_nn_approx(to_vec(x),gsm ; oneterm=false)

xplot = range(0.01,20 ; length=300) |> collect

plot(xplot, [moment_an, moment_approx,moment_approx_more] ;
  label=["full" "approx" "approx 2 terms"],
  legend=:bottom , linewidth=3, linestyle=[:solid :dash :dot]  )

##
moment_an(x) = G.EgiGx_nn(1,to_vec(x),gsm)
moment_approx(x) = G.EgiGx_nn_approx(1,to_vec(x),gsm)
moment_approx_more(x) =  G.EgiGx_nn_approx(1,to_vec(x),gsm ; oneterm=false)

xplot = range(0.01,20 ; length=300) |> collect

plot(xplot, [moment_an, moment_approx,moment_approx_more] ;
   label=["full" "approx" "approx 2 terms"],
   legend=:bottom , linewidth=3, linestyle=[:solid :dash :dot]  )

## squared

moment_an(x) = G.Egi_sqGx_nn(2,to_vec(x),gsm)
moment_approx(x) = G.Egi_sqGx_nn_approx(2,to_vec(x),gsm)
moment_approx_more(x) =  G.Egi_sqGx_nn_approx(2,to_vec(x),gsm ; oneterm=false)

xplot = range(0.01,100 ; length=300) |> collect

plot(xplot, [moment_an, moment_approx,moment_approx_more] ;
   label=["full" "approx" "approx 2 terms"],
   legend=:bottom , linewidth=3, linestyle=[:solid :dash :dot]  )



## variance
itest=1
variance_an(x) = G.Var_giGx_nn(itest,to_vec(x),gsm)
variance_approx(x) = G.Var_giGx_nn_approx(itest,to_vec(x),gsm)
variance_approx2(x) = G.Var_giGx_nn_approx_old(itest,to_vec(x),gsm)

xplot = range(0.01,100 ; length=300) |> collect
plot(xplot, [variance_an, variance_approx, variance_approx2] ;
    label=["full" "approx" "also approx" ],
    legend=:bottom , linewidth=3, linestyle=[:solid :dash :dot]  )

## FF

itest=2
ff_an(x) = G.FFgiGx_nn(itest,to_vec(x),gsm)
ff_approx(x) = G.FFgiGx_nn_approx(itest,to_vec(x),gsm)

xplot = range(0.01,100 ; length=300) |> collect
plot(xplot, [ff_an, ff_approx] ;
    label=["full" "approx" ],
    legend=:bottom , linewidth=3, linestyle=[:solid :dash]  )


## what about 5 D ?


gsm = let α = 2.123 , n=5
  cov = diagm(0 => 0.5 .+ rand(n))
  noise = 0 .* cov
  G.GSM(cov,noise,G.RayleighMixer(α))
end

to_vec(x) = [1.1x - 3,sqrt(10x) , log(x), log(x) , log(x)]

itest=1

moment_an(x) = G.Egi_sqGx_nn(itest,to_vec(x),gsm)
moment_approx(x) = G.Egi_sqGx_nn_approx(itest,to_vec(x),gsm)
moment_approx_more(x) =  G.Egi_sqGx_nn_approx(itest,to_vec(x),gsm ; oneterm=false)

xplot = range(0.01,100 ; length=300) |> collect

plot(xplot, [moment_an, moment_approx,moment_approx_more] ;
   label=["full" "approx" "approx 2 terms"],
   legend=:bottom , linewidth=3, linestyle=[:solid :dash :dot]  )

##

variance_an(x) = G.Var_giGx_nn(2,to_vec(x),gsm)
variance_approx(x) = G.Var_giGx_nn_approx(2,to_vec(x),gsm)
variance_approx2(x) = G.Var_giGx_nn_approx_bis(2,to_vec(x),gsm)

xplot = range(0.01,100 ; length=300) |> collect
plot(xplot, [variance_an, variance_approx, variance_approx2] ;
    label=["full" "approx" "also approx" ],
    legend=:bottom , linewidth=3, linestyle=[:solid :dash :dot]  )
