
using PaddedViews

# Positions , convolution, etc


"""
    function get_cs_coords(radius,nsurr)
Returns coordinates for filters, the first is always `(0,0)`
followed by `nsurr` coordinates in `(x,y)` format that move
counterclockwise , starting from `(radius,0)`
"""
function  get_surround_coordinates(radius::Integer,nsurr::Integer)
  if radius==0
    nsurr==1 && return [(0,0)]
    error("The center can only have 1 filter")
  end
  angs = range(0,2pi;length=nsurr+1)[1:nsurr]
  ivals= @. round(Integer,cos(angs)*radius)
  jvals= @. round(Integer,sin(angs)*radius)
  return collect(zip(ivals,jvals))
end



# bank of filters made by gabors!


"""
    gabor(size_x,size_y,σ,θ,λ,γ,ψ) -> (k_real,k_complex)
Returns a 2 Dimensional Complex Gabor kernel contained in a tuple where
  - `size_x`, `size_y` denote the size of the kernel
  - `σ` denotes the standard deviation of the Gaussian envelope
  - `θ` represents the orientation of the normal to the parallel stripes of a Gabor function
  - `λ` represents the wavelength of the sinusoidal factor
  - `γ` is the spatial aspect ratio, and specifies the ellipticity of the support of the Gabor function
  - `ψ` is the phase offset
#Citation
"Imported" & sligtly adapted from ImageFiltering.jl

N. Petkov and P. Kruizinga, “Computational models of visual neurons specialised in the detection of periodic and aperiodic oriented visual stimuli: bar and grating cells,” Biological Cybernetics, vol. 76, no. 2, pp. 83–96, Feb. 1997. doi.org/10.1007/s004220050323
"""
function gabor(size_x::Integer,size_y::Integer,σ::Real,θ::Real,λ::Real,γ::Real,ψ::Real)
  @assert((size_x>0) && (size_y > 0) && all( [σ,θ,λ,γ] .>= 0),
    "Parameters cannot be negative!")
  σx = σ
  σy = σ/γ
  nstds = 3
  c = cos(θ)
  s = sin(θ)
  xmax = floor(Int64,size_x/2)
  ymax = floor(Int64,size_y/2)
  xmin = -xmax
  ymin = -ymax
  x = [j for i in xmin:xmax,j in ymin:ymax]
  y = [i for i in xmin:xmax,j in ymin:ymax]
  xr = x*c + y*s
  yr = -x*s + y*c
  kernel_real = (exp.(-0.5*(((xr.*xr)/σx^2) + ((yr.*yr)/σy^2))).*cos.(2*(π/λ)*xr .+ ψ))
  kernel_imag = (exp.(-0.5*(((xr.*xr)/σx^2) + ((yr.*yr)/σy^2))).*sin.(2*(π/λ)*xr .+ ψ))
  kernel = (kernel_real,kernel_imag)
  return kernel
end

struct OneGabor{R}
    kernel::Matrix{R}
    parameters::NamedTuple
end
Base.Broadcast.broadcastable(g::OneGabor) = Ref(g)
kernel_size(g::OneGabor) = size(g.kernel,1)

function OneGabor(real::Bool,size::Integer,σ::Real, θ::Real, λ::Real, ψ::Real)
  idx = real ? 1 : 2
  γ = 1.0
  θdeg =round(θ*180/pi;digits=1)
  params = (isreal=real, orientation=θdeg)
  return OneGabor(gabor(size,size,σ,θ,λ,γ,ψ)[idx], params)
end

function position_kernel(kernel::Matrix{<:Real},
        frame_size::Integer,x::Integer,y::Integer)
  wf = div(frame_size,2)
  wk = div(size(kernel,1),2)
  return PaddedView(0.0,kernel,
    (frame_size,frame_size),(wf+x-wk,wf+y-wk))
end
position_kernel(g::OneGabor,siz::Integer,
    x::Integer,y::Integer)=position_kernel(g.kernel,siz,x,y)

struct GaborBank{R}
  frame_size::Integer
  filters::Vector{OneGabor{R}}
  locations
  out_index
end
Base.ndims(gb::GaborBank) = length(vcat(gb.out_index...))

function show_bank(gb::GaborBank ;
        indexes::Union{Nothing,Vector{<:Integer}}=nothing)
  ret=fill(0.0,(gb.frame_size,gb.frame_size))
  indexes=something(indexes, collect(1:2:ndims(gb)))
  for (filt,locs,idxs) in zip(gb.filters,gb.locations,gb.out_index)
    good_idx = findall(i-> i in indexes, idxs)
    for i in good_idx
      pk = position_kernel(filt,gb.frame_size,locs[i]...)
      ret .+= pk
    end
  end
  return ret
end

function meanprod(mat1::AbstractMatrix{<:Real},mat2::AbstractMatrix{<:Real})
    ret = 0.0
    for i in eachindex(mat1)
    @inbounds ret += mat1[i]*mat2[i]
    end
    return 2. * ret/sqrt(length(mat1))
end

function (gb::GaborBank{R})(img::Matrix{R}) where R
  sz = size(img,1)
  @assert all(size(img) .== gb.frame_size ) "Wrong frame size!"
  v = gb(reshape(img,sz,sz,1))
  return v[:]
end

function (gb::GaborBank{R})(imgs::Array{R,3}) where R
  nimg = size(imgs,3)
  @assert all(size(imgs)[1:2] .== gb.frame_size ) "Wrong frame size!"
  ret=Matrix{R}(undef,ndims(gb),nimg)
  for (filt,locs,idxs) in zip(gb.filters,gb.locations,gb.out_index)
    for (loc,idx) in zip(locs,idxs)
      pk = position_kernel(filt,gb.frame_size,loc[1],loc[2])
      ret[idx,:] = mapslices(img->meanprod(pk,img), imgs ; dims=[1,2])
    end
  end
  return ret
end


function test_bounds(gb::GaborBank)
  fs=div(gb.frame_size,2)
  ks= @. div(kernel_size(gb.filters),2)+1
  locs = vcat(gb.locations...)
  locs = vcat(collect.(locs)...)
  if any( (locs .+ maximum(ks)) .> fs)
    @warn "Frame size might be too small for the filters"
    return false
  else
    return true
  end
end

# bank types down here

abstract type BankType end

# surround has 1 orientation only
struct SameSurround <: BankType
    ncenter::Integer
    nsurround::Integer
end

function GaborBank(bt::SameSurround,
        frame_size::Integer,surround_distance::Integer,
        filter_size::Real,spatial_freq::Real,spatial_phase::Real=0.0)
  gab_size = Int32(filter_size*6)
  gab(isreal,θ)=OneGabor(isreal,gab_size,filter_size, θ ,
    spatial_freq, spatial_phase / 180 * pi)
  filters = [gab(true,0.0),gab(false,0.0)]
  center_oris = collect(range(0,pi;length=bt.ncenter+1)[2:bt.ncenter])
  for ori in center_oris
      push!(filters, gab(true,ori), gab(false,ori))
  end
  surround_locations = get_surround_coordinates(surround_distance,bt.nsurround)
  locations = [ vcat([(0,0),],surround_locations) , vcat( [(0,0),],surround_locations),
    [ [(0,0),] for _ in 1:2*(bt.ncenter-1)]... ]
  out_index = map( l-> fill(0,length(l)),locations )
  for (i,o) in  enumerate(out_index)
      o[1]=i
  end
  ns2 = bt.nsurround*2
  nc2 = bt.ncenter*2
  _odd = 1:2:ns2
  _eve = 2:2:ns2
  out_index[1][2:end]= _odd .+ nc2
  out_index[2][2:end]= _eve .+ nc2
  return GaborBank(frame_size, filters,locations,out_index)
end


# surround has 1 orientation only
struct SameSurroundNicePhase <: BankType
    ncenter
    nsurround
end

function GaborBank(bt::SameSurroundNicePhase,
        frame_size::Integer,surround_distance::Integer,
        filter_size::Real,spatial_freq::Real,spatial_phase::Real=0.0)
  gab_size = Int32(filter_size*6)
  gab(isreal,θ,y)=OneGabor(isreal,gab_size,filter_size, θ ,
    spatial_freq, spatial_phase / 180 * pi + y/spatial_freq*(2pi) )
  filters = OneGabor{Float64}[]
  locations = Vector{Vector{Tuple{Int32,Int32}}}(undef,0)
  center_oris = collect(range(0,pi;length=bt.ncenter+1)[1:bt.ncenter])
  for ori in center_oris
      push!(filters, gab(true,ori,0.), gab(false,ori,0.))
      push!(locations,[(0,0),],[(0,0),])
  end
  surround_locations = get_surround_coordinates(surround_distance,bt.nsurround)
  for loc in surround_locations
      y=loc[2]
      push!(filters, gab(true,0.,y), gab(false,0.,y))
      push!(locations,[loc,],[loc,])
  end
  out_index = map( l-> fill(0,length(l)),locations )
  for (i,o) in  enumerate(out_index)
      o[1]=i
  end
  return GaborBank(frame_size, filters,locations,out_index)
end
