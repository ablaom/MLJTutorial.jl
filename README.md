# MLJTutorial.jl

UNDER CONSTRUCTION

Notebooks for introducing the machine learning toolbox
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) (Machine
Learning in Julia) 

<div align="center">
	<img src="data/MLJLogo2.svg" alt="MLJ" width="200">
</div>

Based on tutorials originally part of a 3.5 hour [online
workshop](https://github.com/ablaom/MachineLearningInJulia2020).

### Prerequisites

- Familiarity with basic data manipulation in Julia (vectors, tuples,
dictionaries, arrays, generating random numbers, tabular data - such
as DataDrames.jl, basic stats, Distributions.jl).

- Familiarity with [Machine Learning fundamentals](#more-about-the-tutorials). 

### [Options for running the tutorials](#options-for-running-the-tutorials)


### Topics covered

#### Basic

- Part 1 - **Data Representation**

- Part 2 - **Selecting, Training and Evaluating Models**

- Part 3 - **Transformers and Pipelines**

#### Advanced

- Part 4 - **Tuning hyper-parameters**

- Part 5 - **Advanced model composition** 

The tutorials include links to external resources and exercises with
solutions.


## Options for running the tutorials

### 1. Launch Juptyer notebooks from the Julia REPL

Assuming you have [Julia installed](https://julialang.org/downloads/),
launch the Julia REPL and enter these commands:

```julia
using Pkg
Pkg.develop(url="https://github.com/ablaom/MLJTutorial.jl")
dir = joinpath(Pkg.devdir(), "MLJTutorial")
Pkg.activate(dir)
Pkg.instantiate()

using MLJTutorial
go()
```

This should launch a Jupyter session in your browser. Navigate to
`01_data_representation.unexecuted.ipymb` to start Part 1.


### 2. Run .jl scripts from an IDE (advanced option)

If you know how to do so, clone this repo and open the file
[/notebooks/01_data_representation/notebook.jl](/notebooks/01_data_representation/notebook.jl),
relative to the root directory of your clone.

Otherwise, enter these commands at the Julia REPL, and follow the instructions

```julia
using Pkg
Pkg.develop(url="https://github.com/ablaom/MLJTutorial.jl")
dir = joinpath(Pkg.devdir(), "MLJTutorial")
Pkg.activate(dir)
Pkg.instantiate()

using MLJTutorial
file = joinpath(dir, "notebooks", "01_data_representation", "notebook.jl")
print("Point your Julia-integrated editor to $file")
```

## More about the tutorials 

- The tutorials focus on the *machine learning* part of the data
  science workflow, and less on exploratory data analysis and other
  conventional "data analytics" methodology

- Here "machine learning" is meant in a broad sense, and is not
  restricted to so-called *deep learning* (neural networks)

- The tutorials are crafted to rapidly familiarize the user with what
  MLJ can do and how to do it, and are not a substitute for a course
  on machine learning fundamentals. Examples do not necessarily
  represent best practice or the best solution to a problem.


## Additional resources

# - [MLJ Cheatsheet](https://alan-turing-institute.github.io/MLJ.jl/dev/mlj_cheatsheet/)
# - [Common MLJ Workflows](https://alan-turing-institute.github.io/MLJ.jl/dev/common_mlj_workflows/)
# - [MLJ manual](https://alan-turing-institute.github.io/MLJ.jl/dev/)
# - [Data Science Tutorials in Julia](https://juliaai.github.io/DataScienceTutorials.jl/)


