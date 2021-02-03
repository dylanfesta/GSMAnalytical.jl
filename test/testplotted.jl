using Pkg
Pkg.activate(".")
using GSMAnalytical ; const G = GSMAnalytical
using Test

using Random,LinearAlgebra,Statistics,StatsBase,Distributions
using Calculus
using Plots ; theme(:dark)

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


function EMfit_test(xs::AbstractMatrix{<:Real},
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


##  fit GSM data, and cross validate
n=8
Σg = convert(Matrix{Float64},G.random_covariance_matrix(n,1.0))
Σnoise = zero(Σg)
mx = G.NormalMixer(0.0,1.0)
gsum_true = G.GSuM(Σg,Σnoise,mx)

nsampl = 5_000
mix_train,_,x_train =rand(gsum_true,nsampl)
x_test =rand(gsum_true,nsampl)[3]

gsum_fit = let  σ=std(mix_train), mx=G.NormalMixer(mean(mix_train),var(mix_train)),
  mydir=joinpath(@__DIR__,"ciao")
  mkdir(mydir)
  #mxbad = G.NormalMixer(0.0,4.0)
  Σstart = cov(x_train;dims=2)
  ret=G.GSuM(Σstart,zero(Σstart),mx)
  G.EMFit_somesteps(x_train,ret;nsteps=50,debug=true,debug_dir=mydir)
  ret
end

gsum_fit = let (μ,σ)=mean_and_std(mix_train),
  mx=G.NormalMixer(μ,σ),
  Σstart = cov(x_train;dims=2)
  ret=G.GSuM(Σstart,zero(Σstart),mx)
  G.EMFit_somesteps(x_train,ret;nsteps=60,verbose=true)
  ret
end

plotvs(gsum_fit.covariance,gsum_true.covariance)

gsum_fit.covariance
gsum_true.covariance

std(mean(x_train;dims=1))

##

mix_train





G.distr_p_x(gsum_true)


##
using Serialization

mydir = joinpath(@__DIR__,"ciao")
loggy=map(readdir(mydir)) do f
  boh = deserialize(joinpath(mydir,f))
  boh.negloglik
end

plot(loggy)


testnam =let dd=joinpath(@__DIR__,"emfit_debugxE7ISx")
  @assert isdir(dd)
  file=joinpath(dd,"00001.jls")
  @assert isfile(file)
  file
end
costtest = deserialize(testnam)


readdir(mydir)
