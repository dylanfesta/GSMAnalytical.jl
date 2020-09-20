using Pkg
Pkg.activate(".")
using GSMAnalytical ; const G = GSMAnalytical

using Random,LinearAlgebra,Statistics,StatsBase
using Plots ; theme(:dark)
plotbank(g::G.OneGabor) = heatmap(g.kernel ; ratio=1 , color=:greys)
##
_ = let  isreal=true,
    size = 20,
    σ = 5.0,
    θ = 0 / 180. * pi ,
    λ = 10,
    ψ = 0.0 / 180. * pi
    g=G.OneGabor(isreal,size,σ,θ,λ,ψ)
    heatmap(g.kernel ; ratio=1 , color=:greys)
end

_ = let  isreal=false,
    size = 100,
    σ = 5.0,
    θ = 0 / 180. * pi ,
    λ = 10,
    ψ = 0.0 / 180. * pi
    g=G.OneGabor(isreal,size,σ,θ,λ,ψ)
    heatmap(g.kernel ; ratio=1 , color=:greys)
end

##

bank_test = G.GaborBank(G.SameSurround(7,13), 151,30,7,10)
bank_test = G.GaborBank(G.SameSurround(4,8), 121,20,5,4)
heatmap( G.show_bank(bank_test;indexes=[6,17]) ;ratio=1,c=:grays)

img_test = G.show_bank(bank_test;indexes=[6,20])
bar(bank_test(img_test);leg=false)

bank_test(img_test)[20]

img_test1 = G.show_bank(bank_test; indexes=collect(1:2:100))
img_test2 = G.show_bank(bank_test; indexes=collect(2:2:100))
img_test_both = cat(img_test1,img_test2;dims=3)


bank_test(img_test_both)
bar(bank_test(img_test_both)[:,1];leg=false)

heatmap(img_test1;c=:grays)

G.test_bounds(bank_test)

plotbank(bank_test.filters[2])

using PaddedViews

bao = fill(666,(11,11))

uffi = 200*rand(161,161)

heatmap(bao)
baopad = PaddedView(0.0,bao,(161,161),(80,80))
heatmap(baopad,ratio=1)
baopad2 = let x=75,y=75,
   h1=80,h2=5
   PaddedView(0.0,bao,(161,161),(h1+x-h2,h1+y-h2))
end
heatmap(baopad2.+uffi, ratio=1)
