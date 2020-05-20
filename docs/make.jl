using Documenter, yastan

makedocs(;
    modules=[yastan],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/roualdes/yastan.jl/blob/{commit}{path}#L{line}",
    sitename="yastan.jl",
    authors="Edward A. Roualdes <eroualdes@csuchico.edu>, Chico State",
    assets=String[],
)

deploydocs(;
    repo="github.com/roualdes/yastan.jl",
)
