using GSMAnalytical
using Statistics, LinearAlgebra
using Test

using Random
using QuadGK

const G = GSMAnalytical

mat1d = G.mat1d
Random.seed!(0)
#=
@testset "1D - without noise" begin
    var = 0.1+rand()
    alpha = 0.1+rand()
    noise = 0.1+rand()
    gsm = G.GSM( mat1d(var) , mat1d(0) , G.RayleighMixer(alpha))
    # test p(x) 1D integrates to 1
    @test begin
        f(x) = G.p_x([x,],gsm)
        r = quadgk(f,-Inf,Inf)[1]
        isapprox(r,1.0 ; atol=1E-3)
    end
    # test p(nu|x) integrates to 1
    xfix = randn()
    @test begin
        f(nu) = G.p_nuGx(nu,[xfix,],gsm)
        r = quadgk(f,eps(100.),Inf)[1] # cannot be zero !
        isapprox(r,1.0 ; atol=1E-3)
    end
    # test p(g1|x) integrates to 1
    @test begin
        f(g1) = G.p_giGx(g1,1,[xfix,],gsm)
        if xfix > 0.0
            r = quadgk(f,eps(100.),Inf)[1]
        else
            r = quadgk(f,-Inf,-eps(100.))[1]
        end
        isapprox(r,1.0 ; atol=1E-3)
    end
end

@testset "1D model with noise term" begin
    # create random 1D GSM model
    var = 0.1+rand()
    alpha = 0.1+rand()
    noise = 0.1+rand()
    gsm_noise = G.GSM( mat1d(var) , mat1d(noise) , G.RayleighMixer(alpha))
    # test p(x) 1D integrates to 1
    @test begin
        f(x) = G.p_x([x,],gsm_noise)
        r = quadgk(f,-10,10)[1]
        isapprox(r,1.0 ; atol=1E-3)
    end
    # test p(nu|x) integrates to 1
    xfix = randn()
    @test begin
        f(nu) = G.p_nuGx(nu,[xfix,],gsm_noise)
        r = quadgk(f,0,Inf)[1]
        isapprox(r,1.0 ; atol=1E-3)
    end
    # test p(g1|x) integrates to 1
    @test begin
        f(g1) = G.p_giGx(g1,1,[xfix,],gsm_noise)
        r = quadgk(f,-Inf,Inf)[1]
        @show r
        isapprox(r,1.0 ; atol=1E-3)
    end
end

=#

@testset "Gabor banks" begin
    # index max and close to 1 when filter matches the image
    # otherwise below 0.2
    bank_test = G.GaborBank(G.SameSurround(7,13), 151,30,7,10)
    d=ndims(bank_test)
    imgs = cat([G.show_bank(bank_test;indexes=[i,]) for i in 1:d]...;
        dims=3)
    xs = bank_test(imgs)
    for i in 1:d
      @test isapprox(xs[i,i],1;atol=0.3)
      noti=deleteat!(collect(1:d),i)
      @test isapprox(mean(xs[noti,i]),0.0;atol=0.2)
    end
    #redo, different pars
    bank_test = G.GaborBank(G.SameSurround(4,8), 121,20,5,4)
    d=ndims(bank_test)
    imgs = cat([G.show_bank(bank_test;indexes=[i,]) for i in 1:d]...;
        dims=3)
    xs = bank_test(imgs)
    for i in 1:d
      @test isapprox(xs[i,i],1;atol=0.5)
      noti=deleteat!(collect(1:d),i)
      @test isapprox(mean(xs[noti,i]),0.0;atol=0.2)
    end
end

@testset "Fit GSM" begin
  cov_mat = [ 1.0  -0.22  0.2
             -0.22 1.456  0.11
              0.2  0.11   0.7778 ]
  cov_noise = [ 0.5  0.01  -0.07
                0.01  0.4   -0.08
               -0.07 -0.08  0.234 ]
  mx = G.RayleighMixer(1.345)
  gsm = G.GSM(cov_mat,cov_noise,mx)
  x_train,x_noise = let all = rand(gsm,15_000)
    gs = all[2]
    x_nn = broadcast(*,all[1]',gs)
    noise = all[3]-x_nn
    (x_nn,noise)
  end

  bank_test = G.GaborBank(G.SameSurround(1,2), 121,20,5,4)
  gsm_train = G.GSM_Neuron(x_train,x_noise,mx,bank_test;test_bank=false).gsm
  @test all(
   isapprox.(gsm_train.covariance, gsm.covariance;atol=0.033))
  @test all(
    isapprox.(gsm_train.covariance_noise, gsm.covariance_noise;atol=0.033))

  bank_test = G.GaborBank(G.SameSurround(4,10), 121,20,5,4)
  std_test = 12.34
  noise_lev = 0.333
  xs = std_test .* randn(ndims(bank_test),5_000)
  xs_noise = G.make_noise(10_000,bank_test,noise_lev,xs)
  @test  isapprox(sqrt(mean(diag(cov(xs;dims=2))))*noise_lev ,
      sqrt(mean(diag(cov(xs_noise;dims=2)))) ; atol=0.1)
end
