"""
    notebook_dirs_containing([paths]; root=pwd())

Return, as full paths, a list of all subdirectores of `root` that contain a file called
"notebook.jl" and one or more of the files with full paths specified by `paths`.

```julia-repl
shell> pwd
/MLJTutorial/NotebookManagementTools/test/dummy_tutorials

shell> tree
.
├── bad_tutorial
│   ├── notebook.jl
│   ├── Project.toml
│   └── runtests.jl
├── good_tutorial
│   ├── notebook.jl
│   ├── Project.toml
│   └── runtests.jl
├── nested
│   └── good_tutorial
│       ├── notebook.jl
│       ├── Project.toml
│       └── runtests.jl
├── tutorial_without_script
│   ├── Project.toml
│   └── runtests.jl
└── tutorial_without_tests
    ├── notebook.jl
    └── Project.toml

julia> path1 = joinpath(pwd(), "nested", "good_tutorial", "Project.toml")
julia> path2 = joinpath(pwd(), "tutorial_without_scripts", "Project.toml")
julia> notebook_dirs_containing([path1, path2])
1-element Vector{String}:
 "MLJTutorial/NotebookManagementTools/test/dummy)tutorials/nested/good_tutorial"
```

"""
function notebook_dirs_containing(paths=nothing; root=pwd())
    all_tutorial_dirs = notebook_dirs(; root)
    isnothing(paths) && return all_tutorial_dirs
    filter(all_tutorial_dirs) do dir
        any(path -> startswith(path, dir), paths)
    end |> unique
end

"""
    notebook_dirs(root=pwd())

Return all subdirectories of `root` (possibly nested) that contain a file called
"notebook.jl". Directores are returned as full paths.

"""
function notebook_dirs(; root=pwd())
    tree = walkdir(root)
    dirs = String[]
    for (dir, _, contents) in tree
        "notebook.jl" in contents && push!(dirs, dir)
    end
    return dirs
end
