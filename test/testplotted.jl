using Pkg
Pkg.activate(".")
using GSMAnalytical ; const G = GSMAnalytical

using Random,LinearAlgebra,Statistics,StatsBase
using Plots
using QuadGK

##
#  first test, generate from random 1D gsm, and take a look at x

mat1d = D.mat1d


gsm = let cov = mat1d(1.0), noise = mat1d(0.0)
  mx = G.RayleighMixer(1.0)
  G.GSM(cov,noise,mx)
end

mx,gs,xs =  rand(gsm,5_000)
histogram(xs[1,:] ; nbins=80, normed=true, leg=false)

# second test p(x) integrates to 1
_ = let f(x) = G.p_x([x,],gsm)
   r = quadgk(f,-Inf,Inf)[1]
   @info "integral is $r"
 end

# third test: samples Vs curve

gsm = let cov = mat1d(10), noise = mat1d(0.0)
  mx = G.RayleighMixer(1.0)
  G.GSM(cov,noise,mx)
end
_ =  let (mx,gs,xs) =  rand(gsm,10_000),
  f(x) = G.p_x([x,],gsm),
  h = normalize(fit(Histogram,xs[1,:]; nbins=101))
  binsc = midpoints(collect(h.edges[1]))
  plot(h ; label="samples")
  plot!(binsc, f ; linewidth=5,opacity=0.8, color="red", label="analytic")
end

## all good ! now p( nu | x), still 1 D

gsm = let cov = mat1d(1), noise = mat1d(0.5)
  mx = G.RayleighMixer(1.0)
  G.GSM(cov,noise,mx)
end
# must integrate to 1 in nu for a given x
_ = let x=0.9 ,
   f(nu) = G.p_nuGx(nu,[x,],gsm)
   r = quadgk(f,0,Inf)[1]
   @info "integral is $r"
 end

#
