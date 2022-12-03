using BCIInterface
using Documenter

DocMeta.setdocmeta!(BCIInterface, :DocTestSetup, :(using BCIInterface); recursive=true)

makedocs(;
    modules=[BCIInterface],
    authors="Alexander Reimer <alexander.reimer2357@gmail.com>, Matteo Friedrich <matteo.r.friedrich@gmail.com>",
    repo="https://github.com/AR102/BCIInterface.jl/blob/{commit}{path}#{line}",
    sitename="BCIInterface.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://AR102.github.io/BCIInterface.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/AR102/BCIInterface.jl",
    devbranch="main",
)
