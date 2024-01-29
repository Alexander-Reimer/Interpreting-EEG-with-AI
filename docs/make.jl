using BCIInterface
using Documenter

DocMeta.setdocmeta!(BCIInterface, :DocTestSetup, :(using BCIInterface); recursive=true)

makedocs(;
    modules=[BCIInterface],
    authors="Alexander Reimer <alexander.reimer2357@gmail.com>, Matteo Friedrich <matteo.r.friedrich@gmail.com>",
    repo="https://github.com/Alexander-Reimer/Interpreting-EEG-with-AI/blob/{commit}{path}#{line}",
    sitename="BCIInterface.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Alexander-Reimer.github.io/Interpreting-EEG-with-AI",
        edit_link="main",
        assets=String[]
    ),
    pages=[
        "Home" => "index.md",
        "Function & Type Documentation" => "API.md",
        "Supported Boards" => "supported_boards.md",
        "Package Development" => "developers.md"
    ]
)

deploydocs(;
    repo="github.com/Alexander-Reimer/Interpreting-EEG-with-AI",
    devbranch="refactor"
)
