module MLJTutorial

import IJulia
import Pluto

const DATASETS = [:horse, :house, :small]
const ROOT = joinpath(@__DIR__, "..")
const NOTEBOOKS = joinpath(ROOT, "notebooks")
const FIRST_PLUTO_NOTEBOOK =
    joinpath(NOTEBOOKS, "01_data_representation", "notebook.pluto.jl")

export jupyter, pluto, go, remove, load_house, load_horse, load_small

jupyter() = begin
    IJulia.notebook(dir=NOTEBOOKS)
end
const go = jupyter

pluto() = begin
    Pluto.run(notebook=FIRST_PLUTO_NOTEBOOK)
end

end # module
