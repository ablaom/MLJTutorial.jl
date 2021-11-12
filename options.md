# Options for running the tutorials

### 1. To launch Juptyer or Pluto notebooks from the Julia REPL

Assuming you have [Julia installed](https://julialang.org/downloads/),
launch the Julia REPL and enter these commands:

```julia
using Pkg
Pkg.develop(url="https://github.com/ablaom/MLJTutorial.jl")
dir = joinpath(Pkg.devdir(), "MLJTutorial")
Pkg.activate(dir)
Pkg.instantiate()

using MLJTutorial
```

### For Juptyer

Run 

```julia
juptyer() # or go()
```

This should launch a Jupyter session in your browser, with a directory
of the available notebooks.

### For Pluto

Run

```julia
pluto()
```

This should launch the first tutorial as a Pluto notebook in your
browser.


### 2. To run .jl scripts from an IDE (advanced option)

If you know how to do so, clone this repo and open the file
[/notebooks/01_data_representation/notebook.jl](/notebooks/01_data_representation/notebook.jl),
relative to the root directory of your clone.

Otherwise, enter these commands at the Julia REPL, and follow the instructions generated.

```julia
using Pkg
Pkg.develop(url="https://github.com/ablaom/MLJTutorial.jl")
dir = joinpath(Pkg.devdir(), "MLJTutorial")
Pkg.activate(dir)
Pkg.instantiate()

using MLJTutorial
file = joinpath(dir, "notebooks", "01_data_representation", "notebook.jl")
print("To open the first tutorial, point your Julia-integrated editor to $file")
```
