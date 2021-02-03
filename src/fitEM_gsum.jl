

function dotinvmatprodprime(x::AbstractVector{R},
    Σ::AbstractMatrix{R},y::AbstractVector{R}) where R
  iΣ=inv(Σ)
  L=cholesky(Σ).L
  return dotinvmatprodprime(x,y,iΣ,L)
end
function dotinvmatprodprime(x::AbstractVector{R},y::AbstractVector{R},
    iΣ::AbstractMatrix{R},L::AbstractMatrix{R}) where R
  n=size(iΣ,1)
  ret = zeros(n,n)
  lf, rg = x'*iΣ, iΣ*y
  for j in 1:n, i in j:n
    v = view(L,:,j)
    ret[i,j] = - (dot(lf,v)*rg[i] + dot(rg,v)*lf[i])
  end
  return ret
end



function logdetprime(Σ::AbstractMatrix{<:Real})
  L=cholesky(Σ).L
  iΣ=inv(Σ)
  return logdetprimeiSL(iΣ,L)
end

function logdetprimeiSL(iΣ::AbstractMatrix{R},L::AbstractMatrix{R}) where R
  n=size(L,1)
  ret = zeros(size(iΣ)...)
  ijgrad = zero(ret)
  for j in 1:n, i in j:n
    fill!(ijgrad,0.0)
    ijgrad[i,:] .= L[:,j]
    ijgrad[:,i] .+= L[:,j]
    ret[i,j] =tr(iΣ*ijgrad)
  end
  return ret
end


"""
  EMfit_Estep(x::AbstractArray{R},
      gsum::GSuM{NormalMixer{R},R}) where R

Single E step, finds the mean and variance of the mixer, given the previous parameters
"""
function EMfit_Estep(x::AbstractArray{R},
    gsum::GSuM{NormalMixer{R},R}) where R
  @assert !hasnoise(gsum) "Case with noise not covered yet!"
  return pars_p_nuGx(x,gsum)
end


"""
  EMfit_Mstep_cost(μstar::Vector{R},σstar::Vector{R},xs::Matrix{R},
      gsum::GSuM{NormalMixer{R},R}) where R

Cost for M step, given the vectorized results of the E step
"""
function EMfit_Mstep_cost(μstar::Vector{R},σstar::Vector{R},xs::Matrix{R},
    gsum::GSuM{NormalMixer{R},R}) where R
  @assert !hasnoise(gsum) "Case with noise not covered yet!"
  n=n_dims(gsum)
  icov=inv(gsum.covariance)
  μmix,σmix=gsum.mixer.μ,gsum.mixer.σ
  iv=fill(1.,n_dims(gsum))
  ret = 0.0
  ndat=size(xs,2)
  for (nd,x) in enumerate(eachcol(xs))
    μs,σs=μstar[nd],σstar[nd]
    sumstar = μs^2+σs^2
    ret -= 0.5dot(x,icov,x)
    ret += μs*dot(x,icov,iv)
    ret -= 0.5dot(iv,icov,iv) * sumstar
  end
  detcov = max(eps(),det(gsum.covariance))
  ret -= ndat*0.5*log(detcov)
  return ret
end

"""
  EMfit_Mstep_costprime(μstar::Vector{R},σstar::Vector{R},
      xs::Matrix{R},gsum::GSuM{NormalMixer{R},R},L::Matrix{R}) where R

Gradient of cost for the M step
"""
function EMfit_Mstep_costprime(μstar::Vector{R},σstar::Vector{R},
    xs::Matrix{R},gsum::GSuM{NormalMixer{R},R},L::AbstractMatrix{R}) where R
  @assert !hasnoise(gsum) "Case with noise not covered yet!"
  n=n_dims(gsum)
  Σ=gsum.covariance
  iΣ=inv(Σ)
  μmix,σmix=gsum.mixer.μ,gsum.mixer.σ
  iv=fill(1.,n_dims(gsum))
  dest=zeros(n,n)
  _primethingy=dotinvmatprodprime(iv,iv,iΣ,L)
  for (nd,x) in enumerate(eachcol(xs))
    μs,σs=μstar[nd],σstar[nd]
    sumstar = μs^2+σs^2
    dest .-= 0.5.*dotinvmatprodprime(x,x,iΣ,L)
    dest .+= μs.*dotinvmatprodprime(x,iv,iΣ,L)
    dest .-= 0.5*sumstar.*_primethingy
  end
  ndat=size(xs,2)
  dest .-= ndat*0.5.*logdetprimeiSL(iΣ,L)
  return dest
end


"""
  EMFit_Mstep_optim(μstar::Vector{R},σstar::Vector{R},
      xs::AbstractMatrix{<:Real},gsum::GSuM{NormalMixer{R},R}) where R

"""
function EMFit_Mstep_optim(μstar::Vector{R},σstar::Vector{R},
    xs::AbstractMatrix{<:Real},gsum::GSuM{NormalMixer{R},R}) where R
  Σ=gsum.covariance
  n=size(Σ,1)
  if !isposdef(Σ)
    @warn "Σ not positive definite. Adding a small diagonal"
    for i in 1:n
      Σ[i,i] += 1E-4
    end
  end
  L=cholesky(Σ).L
  Lv0 = L[:]
  costfun = function (Lv::Vector{R})
    L=reshape(Lv,n,n)
    copy!(Σ,L*L')
    return - EMfit_Mstep_cost(μstar,σstar,xs,gsum)
  end
  gradfun! = function (grad::Vector{R},Lv::Vector{R})
    L=reshape(Lv,n,n)
    copy!(Σ,L*L')
    gradMat= EMfit_Mstep_costprime(μstar,σstar,xs,gsum,L)
    copy!(grad,.- gradMat[:])
    return  grad
  end
  #alg=ConjugateGradient() # BFGS()
  alg=LBFGS()
  res=optimize(costfun, gradfun!, Lv0, alg,
    Optim.Options(iterations=100,time_limit=5.0))
  Lout=reshape(Optim.minimizer(res),n,n)
  return Lout*Lout',res
end


"""
    EMFit_somesteps(xs::AbstractMatrix{<:Real},
        gsum::GSuM{NormalMixer{R},R};nsteps::Integer=10,verbose::Bool=true) where R

Fit of additive GSM without noise and with known additive mixer. The fit is only used
for the covariance matrix of the features. The mixer parameters and the initial conditions
should be stored in `gsum`
"""
function EMFit_somesteps(xs::AbstractMatrix{<:Real},
    gsum::GSuM{NormalMixer{R},R};nsteps::Integer=10,
    debug::Bool=true,debug_dir::Union{String,Nothing}=nothing) where R
  if (debug && isnothing(debug_dir))
    test_dir=abspath(@__DIR__,"..","test")
    debug_dir=mktempdir(test_dir;cleanup=true,prefix="emfit_debug")
  end
  for i in 1:nsteps
    μstar,σstar=EMfit_Estep(xs,gsum)
    Σfit,result=EMFit_Mstep_optim(μstar,σstar,xs,gsum)
    if debug
      nam =joinpath(debug_dir,lpad(i,5,"0") *".jls")
      serialize(nam,(Sigmafit=Σfit, negloglik= Optim.minimum(result)))
      println("step $i of $nsteps done")
    end
    dostep=true
    if !isposdef(Σfit)
      ee=minimum(real.(eigvals(Σfit)))
      @error("matrix not positive def ... why?\n"*
        "smaller eigenvalue is $ee")
      #Σfit = Σfit + abs(3*ee)*I
      dostep=false
    end
    if any(abs.(Σfit) .> 50.)
      @error "values too high! Skipping this step!"
      dostep=false
    end
    if dostep
      copy!(gsum.covariance,Σfit)
    end
  end
  return nothing
end
