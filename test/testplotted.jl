using Pkg
Pkg.activate(".")
using GSMAnalytical ; const G = GSMAnalytical

using Random,LinearAlgebra,Statistics,StatsBase
using Plots
using QuadGK

function do_gsm(cov,noise,mx)
  cov = G.mat1d(cov)
  noise = G.mat1d(noise)
  mx = G.RayleighMixer(mx)
  G.GSM(cov,noise,mx)
end
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
