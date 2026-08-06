"""
    dirs_containing([paths]; root=pwd())

Return, as full paths, a list of those top-level subdirectores of `root` that contain one
or more of the full paths specified by `paths`.

```julia-repl
shell> pwd
/Users/anthony/GoogleDrive/Julia/MLJ/StatisticalMeasures

shell> tree -L 2
.
├── docs
│   ├── build
│   ├── make.jl
│   ├── Project.toml
│   └── src
├── LICENSE
├── Project.toml
├── README.md
├── src
│   ├── confusion_matrices.jl
│   ├── continuous.jl
    ├── tools.jl
│   ├── docstrings.jl
│   └── unfussy.jl
└── test
    ├── confusion_matrices.jl
    ├── continuous.jl
    └── tools.jl

julia> paths = [joinpath(pwd(), "src", "confusion_matrices.jl"),];
julia> dirs_containing(paths)
1-element Vector{String}:
 "/Users/anthony/GoogleDrive/Julia/MLJ/StatisticalMeasures/src"
```

"""
function dirs_containing(paths=nothing; root=pwd())
    all_tutorial_dirs = filter(isdir, readdir(root, join=true))
    isnothing(paths) && return all_tutorial_dirs
    filter(all_tutorial_dirs) do dir
        any(path -> startswith(path, dir), paths)
    end |> unique
end

"""
    dirs(; root=pwd())

Return *all* top-level subdirectories of `root` as full paths.

"""
dirs(; kwargs...) = dirs_containing(; kwargs...)
