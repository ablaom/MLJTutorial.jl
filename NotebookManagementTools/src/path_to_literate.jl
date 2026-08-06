# # METHODS

"""
    path_to_literate()

Return the path to a Project.toml file with a [deps] entry for "Literate". Use
[`set_path_to_literate`](@ref) to set or change.

"""
path_to_literate() =  PATH_TO_LITERATE[]

"""
    set_path_to_literate(path)

Point `TutorialsTools` to the location of a directory containing a Project.toml file with
a [deps] entry for "Literate". Typically, this is the absolute path to a repository "docs"
directory.

```julia-repl
julia> pwd()
"/Users/anthony/GoogleDrive/Julia/ClassImbalanceTutorials.jl"

julia> set_path_to_literate("~/GoogleDrive/Julia/ClassImbalanceTutorials.jl/docs/")
```

Use [`path_to_literate()`](@ref) to inspect the path once set.

"""
set_path_to_literate(path) = (PATH_TO_LITERATE[] = expanduser(path))
