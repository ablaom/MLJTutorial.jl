module MLJTutorial

import IJulia
using CSV
using DataFrames

const DATASETS = [:horse, :house, :small]
const ROOT = joinpath(@__DIR__, "..")
const NOTEBOOKS = joinpath(ROOT, "notebooks")

export load_house, load_horse, load_small


jupyter() = begin
    IJulia.notebook(dir=joinpath(NOTEBOOKS, "notebooks"))
end

# pluto() = begin
#     Pluto.run(notebook=joinpath(NOTEBOOKS, "sandbox", "notebook.pluto.jl")
# end

include("datasets.jl")

end # module
