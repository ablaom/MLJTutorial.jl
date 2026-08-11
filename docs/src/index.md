# MLJTutorial.jl

Tutorials for introducing the machine learning toolbox [MLJ](https://juliaml.ai) (Machine
Learning in Julia)

Based on tutorials originally part of a 3.5 hour [online
workshop](https://github.com/ablaom/MachineLearningInJulia2020). Updated August 2026.


## Prerequisites

- Familiarity with basic data manipulation in Julia: vectors, tuples,
  dictionaries, arrays, generating random numbers, tabular data (e.g.,
  DataDrames.jl) basic stats, Distributions.jl.
  
- Familiarity with [Julia package management](https://docs.julialang.org/en/v1/stdlib/Pkg/)

- Familiarity with Machine Learning fundamentals and best practice.

## The tutorials

### Basic

- [Tutorial 1. Data Representation](@ref)
- [Tutorial 2. Selecting, Training and Evaluating Models](@ref)
- [Tutorial 3. Transformers and Pipelines](@ref)

### Advanced

- [Tutorial 4. Tuning hyperparameters](@ref)
- [Tutorial 5. Composition](@ref)
- [Solutions to Exercises](@ref)
- [Lightning Tour of MLJ](@ref)

## Running tutorial code for yourself

You can find the annotated Julia scripts from which tutorials are generated in [these
directories](docs/src/notebooks). Package environment files for this repository are
structured using Julia's package workspaces. To use with Julia 1.12 or later:

1. Clone this GitHub repository to your computer

2. In a new Julia session, activate and instantiate the root project (as specified by [Project.toml](../../Project.toml)). This resolves a valid set of package for all the tutorials, downloads package code to your computer, and carries out some precompilation. This may take a few minutes. You need carry out this step only once. 

3. Activate the project for the particular tutorial of interest (as specified by Project.toml file in the corresponding directory). 
   
4. Execute code blocks copied from the rendered tutorial (as linked above) or form the corresponding .jl script.


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

Visit the [MLJ Learn page](https://juliaai.github.io/MLJ.jl/dev/learning_mlj/) for
additional learning resources.

## Legacy Pluto notebooks

Older Pluto versions of the tutorials, provided by @roland-KA, are also hosted here,
alongside the plain julia scripts, but they have not been updated and are not currently
synchronized with the plain scripts.

---

This repository is maintained with the help of [NoteBookManagementTools](@ref.jl, an
experimental, unregistered package embedded in the repository.
