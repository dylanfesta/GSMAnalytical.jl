using GSMAnalytical
using Statistics, LinearAlgebra
using Test

using Random
using QuadGK,Calculus

const G = GSMAnalytical

mat1d = G.mat1d
Random.seed!(0)

function EMfit_test(xs::AbstractMatrix{<:Real},μmix::R,σmix::R,
    L::LowerTriangular{R}) where R
  μstar,σstar=G.EMfit_Estep(xs,μmix,σmix,L)
  n=size(L,1)
  # analytic
  grad_an=G.EMfit_Mstep_costprime(μstar,σstar,xs,μmix,σmix,L)
  # numeric
  μstar,σstar = G.pars_p_nuGx_nonoise(xs,μmix,σmix,L)
  fun_grad=function(Lij::Real,i::Integer,j::Integer)
    Lpre = L[i,j]
    L[i,j]=Lij
    _ret = G.EMfit_Mstep_cost(μstar,σstar,xs,μmix,σmix,L)
    L[i,j]=Lpre
    return _ret
  end
  grad_num = zeros(size(L)...)
  for j in 1:n, i in j:n
    grad_num[i,j] = Calculus.gradient( Lij->fun_grad(Lij,i,j),L[i,j])
  end
  return grad_num[:],grad_an[:]
end

@testset "1D - without noise" begin
    var = 0.1+rand()
    alpha = 0.1+rand()
    noise = 0.1+rand()
    gsm = G.GSM( mat1d(var) , mat1d(0.0) , G.RayleighMixer(alpha))
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
        isapprox(r,1.0 ; atol=1E-3)
    end
end

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
  xs_noise = G.make_noise(10_000,bank_test)
  @test isapprox(1., G.mean_std(cov(xs_noise;dims=2)); atol=0.01)
end

@testset "Fit with banks and noise" begin
    bank_test = G.GaborBank(G.SameSurround(1,8), 91,12,5,10)
    # train just with noise
    ntrain=10_000
    noise_patches = let n = bank_test.frame_size
      rand(n,n,ntrain)
    end
    noiselev = 0.3456
    gsm_neu = G.GSM_Neuron(noise_patches,noiselev,G.RayleighMixer(0.4876),bank_test)
    @test isapprox(noiselev,
        G.mean_std(gsm_neu.gsm.covariance_noise)/G.mean_std(gsm_neu.gsm.covariance);
        atol=0.01)
    @test isapprox(1.,cor(gsm_neu.gsm.covariance[:],gsm_neu.gsm.covariance_noise[:]);
        atol=0.05)
end


@testset "Fit additive GSM (GSuM) with EM" begin
    n=8
    Σg = convert(Matrix{Float64},G.random_covariance_matrix(n,3.))
    Σnoise = zero(Σg)
    μmix=0.44
    σmix=0.89
    L=cholesky(Σg).L
    mixer=G.NormalMixer(μmix,σmix)
    gsum = G.GSuM(copy(Σg),Σnoise,mixer)
    xstry = let n=300
      G.rand(gsum,n)[3]
    end
    grad_num,grad_an=EMfit_test(xstry,μmix,σmix,L)
    @test all(isapprox.(grad_num,grad_an;atol=0.001))
    μstar,σstar = G.EMfit_Estep(xstry,μmix,σmix,L)
    Lfit,result=G.EMFit_Mstep_optim(μstar,σstar,xstry,μmix,σmix,L)
    Σfit=Lfit*Lfit'
    @test all(isapprox.(Σfit,Σg;atol=0.66))
    ##
    n=4
    Σg = convert(Matrix{Float64},G.random_covariance_matrix(n,3.))
    Σtrue = copy(Σg)
    Σnoise = zero(Σg)
    mixer=G.NormalMixer(0.44,0.890)
    gsum = G.GSuM(Σg,Σnoise,mixer)
    xstry = let n=1_000
      G.rand(gsum,n)[3]
    end
    Σstart=cov(xstry;dims=2)
    copy!(gsum.covariance,Σstart)
    Ltrain,trace=G.EMFit_somesteps(xstry,gsum;nsteps=10,debug=false)
    Σfit=Ltrain*Ltrain'
    msqerr(x,y)=mean( @. (x-y)^2 )
    @test msqerr(Σfit,Σtrue) <  10*msqerr(Σstart,Σtrue)
end


@testset "Cross-validation GSM (true) vs additive GSM" begin
    n=4
    Σg = convert(Matrix{Float64},G.random_covariance_matrix(n,3.))
    Σnoise = zero(Σg)
    mx = G.RayleighMixer(1.345)
    gsm_true = G.GSM(Σg,Σnoise,mx)
    nsampl = 1_000
    mix_train,_,x_train =rand(gsm_true,nsampl)
    x_test =rand(gsm_true,nsampl)[3]

    gsm_fit = G.gsm_momentmatch_given_noise(x_train,
      copy(Σnoise),G.RayleighMixer(mean(mix_train)))
    gsum_fit = let σ=std(mix_train), mx=G.NormalMixer(0.0,σ),
      Σstart = cov(x_train;dims=2)
      ret=G.GSuM(Σstart,zero(Σstart),mx)
      Llearn,_=G.EMFit_somesteps(x_train,ret;nsteps=30,debug=false)
      copy!(ret.covariance,Llearn*Llearn')
      ret
    end
    lp_gsum = G.meanlog_px(x_test,gsum_fit)
    lp_gsm = G.meanlog_px(x_test,gsm_fit)
    @test lp_gsm > lp_gsum
end

@testset "Cross-validation additive GSM (true) vs GSM" begin
    n=4
    Σg = convert(Matrix{Float64},G.random_covariance_matrix(n,3.))
    Σnoise = zero(Σg)
    mx = G.NormalMixer(0.05,1.2)
    gsum_true = G.GSuM(Σg,Σnoise,mx)
    nsampl = 1_000
    mix_train,_,x_train =rand(gsum_true,nsampl)
    x_test =rand(gsum_true,nsampl)[3]
    mx=G.RayleighMixer(mean(mix_train))
    gsm_fit = G.gsm_momentmatch_given_noise(x_train,copy(Σnoise),mx)
    gsum_fit = let σ=std(mix_train), mx=G.NormalMixer(mean(mix_train),var(mix_train)),
      Σstart = cov(x_train;dims=2)
      ret=G.GSuM(Σstart,zero(Σstart),mx)
      Llearn,_=G.EMFit_somesteps(x_train,ret;nsteps=30,debug=false)
      copy!(ret.covariance,Llearn*Llearn')
      ret
    end
    lp_gsum = G.meanlog_px(x_test,gsum_fit)
    lp_gsm = G.meanlog_px(x_test,gsm_fit)

    @test lp_gsum > lp_gsm
end
