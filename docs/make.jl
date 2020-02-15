using Documenter, GSMAnalytical

makedocs(;
    modules=[GSMAnalytical],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/dylanfesta/GSMAnalytical.jl/blob/{commit}{path}#L{line}",
    sitename="GSMAnalytical.jl",
    authors="Dylan Festa",
    assets=String[],
)

deploydocs(;
    repo="github.com/dylanfesta/GSMAnalytical.jl",
)
