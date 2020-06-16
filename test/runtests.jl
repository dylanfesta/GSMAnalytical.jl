using GSMAnalytical
using Test

using Random
using QuadGK

const G = GSMAnalytical

mat1d = G.mat1d
Random.seed!(0)

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
