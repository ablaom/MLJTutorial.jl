### A Pluto.jl notebook ###
# v0.17.5

using Markdown
using InteractiveUtils

# ╔═╡ d09256dd-6c0d-4e28-9e54-3f7b3ca87ecb
begin
	using MLJ
	using MLJLinearModels
	using MLJDecisionTreeInterface
	using PlutoUI
end

# ╔═╡ 24abafc3-0e26-4cd2-9bca-7c69b794f8ce
html"""
<div style="
position: absolute;
width: calc(100% - 30px);
border: 50vw solid SteelBlue;
border-top: 500px solid SteelBlue;
border-bottom: none;
box-sizing: content-box;
left: calc(-50vw + 15px);
top: -500px;
height: 300px;
pointer-events: none;
"></div>

<div style="
height: 300px;
width: 100%;
background: SteelBlue;
color: #88BBD6;
padding-top: 68px;
padding-left: 5px;
">

<span style="
font-family: Vollkorn, serif;
font-weight: 700;
font-feature-settings: 'lnum', 'pnum';
"> 

<p style="
font-family: Alegreya sans;
font-size: 1.4rem;
font-weight: 300;
opacity: 1.0;
color: #CDCDCD;
">Tutorial - Part 5</p>
<p style="text-align: left; font-size: 2.5rem;">
Machine Learning in Julia
</p>
"""

# ╔═╡ 0b8dcb85-3804-4c9b-b20a-319ba81ba28c
PlutoUI.TableOfContents(title = "MLJ Tutorial - Part 5")

# ╔═╡ b67c62c2-15ab-4328-b795-033f6f2a0674
md"""
An introduction to the
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/stable/)
toolbox.
"""

# ╔═╡ bc689638-fd19-4c9f-935f-ddf6a6bfbbdd
md"## Set-up"

# ╔═╡ 197fd00e-9068-46ea-af2a-25235e544a31
md"Inspect Julia version:"

# ╔═╡ f6d4f8c4-e441-45c4-8af5-148d95ea2900
VERSION

# ╔═╡ 45740c4d-b789-45dc-a6bf-47194d7e8e12
md"The following instantiates a package environment."

# ╔═╡ 42b0f1e1-16c9-4238-828a-4cc485149963
md"""
The package environment has been created using **Julia 1.6** and may not
instantiate properly for other Julia versions.
"""

# ╔═╡ 499cbc31-83ba-4583-ba1f-6363f43ec697
md"## General resources"

# ╔═╡ db5821e8-956a-4a46-95ea-2955abd45275
md"""
- [MLJ Cheatsheet](https://alan-turing-institute.github.io/MLJ.jl/dev/mlj_cheatsheet/)
- [Common MLJ Workflows](https://alan-turing-institute.github.io/MLJ.jl/dev/common_mlj_workflows/)
- [MLJ manual](https://alan-turing-institute.github.io/MLJ.jl/dev/)
- [Data Science Tutorials in Julia](https://juliaai.github.io/DataScienceTutorials.jl/)
"""

# ╔═╡ 38fdb776-44b5-4df5-b1b3-dd47e367ba54
md"# Part 5 - Advanced Model Composition"

# ╔═╡ 23408dac-0ade-41f4-8d7e-8e449afd1c48
md"""
> **Goals:**
> 1. Learn how to build a prototypes of a composite model, called a *learning network*
> 2. Learn how to use the `@from_network` macro to export a learning network as a new stand-alone model type
"""

# ╔═╡ a6faf8b4-f0a5-4b16-a949-7f3bd292fe31
md"""
Pipelines are great for composing models in an unbranching
sequence. Another built-in type of model composition is a model
*stack*; see
[here](https://alan-turing-institute.github.io/MLJ.jl/dev/model_stacking/#Model-Stacking)
for details. For other more complicated model compositions you'll want to
use MLJ's generic model composition syntax. There are two main
steps:
- **Prototype** the composite model by building a *learning
  network*, which can be tested on some (dummy) data as you build
  it.
- **Export** the learning network as a new stand-alone model type.

Like pipeline models, instances of the exported model type behave
like any other model (and are not bound to any data, until you wrap
them in a machine).
"""

# ╔═╡ bf69ff35-6d45-418e-9873-1b1430e635cc
md"## Building a pipeline using the generic composition syntax"

# ╔═╡ 3d480ea1-bba7-49de-b43e-d0ebe87da176
md"To warm up, we'll do the equivalent of"

# ╔═╡ 72801926-9f1d-4131-be77-3b4fa617d234
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels

# ╔═╡ cebd7290-ff85-4ebd-a89d-0aa1492fb1e0
pipe = Standardizer |> LogisticClassifier(lambda = 0.001)

# ╔═╡ 9e3a5fcf-8959-440c-abd3-cb77d7a79683
md"using the generic syntax."

# ╔═╡ 88ca15eb-49f8-4525-879e-dc128f3db7b3
md"Here's some dummy data we'll be using to test our learning network:"

# ╔═╡ 0b842f20-acd3-40ba-a4f7-edbe2341eb94
X, y = make_blobs(5, 3)

# ╔═╡ c720911a-478b-4aa0-bf32-94657e65284e
md"""
### Step 0 - Combine the models 
Proceed as if you were combining the models "by hand",
using all the data available for training, transforming and
prediction:
"""

# ╔═╡ 48692370-0c4b-4080-9afd-65a5b5facac9
begin
  stand = Standardizer();
  linear = LogisticClassifier(lambda = 0.001);
  
  mach1 = machine(stand, X);
  fit!(mach1);
  Xstand = transform(mach1, X);
  
  mach2 = machine(linear, Xstand, y);
  fit!(mach2);
  yhat = predict(mach2, Xstand)
end

# ╔═╡ b398bc7f-bf22-4bb5-b6c8-54f5ed90a964
md"""
### Step 1 - Wrap in `Source` nodes
Edit your code as follows:
- pre-wrap the data in `Source` nodes
- delete the `fit!` calls
"""

# ╔═╡ e193a088-1262-4b0f-8a29-32f11452654a
begin
  X_node = source(X)  # or X = source() if not testing
  y_node = source(y)  # or y = source()
  
  # stand = Standardizer(); 						# already defined in step 0
  # linear = LogisticClassifier(lambda = 0.001); 	# already defined in step 0
  
  mach1_node = machine(stand, X_node);
  Xstand_node = transform(mach1_node, X_node);
  
  mach2_node = machine(linear, Xstand_node, y_node);
  yhat_node = predict(mach2_node, Xstand_node)
end

# ╔═╡ 2f187af9-21d8-4c18-a5f1-151fcbe229a2
md"""
Now `X_node`, `y_node`, `Xstand_node` and `yhat_node` are *nodes* ("variables" or
"dynamic data") instead of data. All training, predicting and
transforming is now executed lazily, whenever we `fit!` one of these
nodes. We *call* a node to retrieve the data it represents in the
original manual workflow.
"""

# ╔═╡ e13fd789-3016-4f88-bb6d-5087ebe511ce
fit!(Xstand_node)

# ╔═╡ 857a0ae3-346f-4d49-b395-4770aab7d47b
Xstand_node()

# ╔═╡ ee7f00ab-6195-4a65-9ba5-46e06ec18ce5
fit!(yhat_node)

# ╔═╡ 50265a35-512f-4a2f-9d87-1d1fbb0e3997
yhat_node()

# ╔═╡ b53fc776-08cc-4b9d-b950-fddef2a1fb10
md"""
The node `yhat_node` is the "descendant" (in an associated DAG we have
defined) of a unique source node:
"""

# ╔═╡ ac92e1ee-e50c-4408-951b-64cb2a36c8e3
sources(yhat_node)

# ╔═╡ 4b9f5009-0e61-4ef7-b0e6-441461cc8770
md"""
The data at the source node is replaced by `Xnew` to obtain a
new prediction when we call `yhat1_node` like this:
"""

# ╔═╡ 1ed5139a-2618-418d-9adf-109bee706090
Xnew, _ = make_blobs(2, 3)

# ╔═╡ 84d84a02-ae06-4f28-8cb0-de601961bc02
yhat_node(Xnew)

# ╔═╡ c6d367a8-e734-4c1b-a87d-950c50fb2955
md"""
### Step 2 - Export the network
Export the learning network as a new stand-alone model type

Now, somewhat paradoxically, we can wrap the whole network in a
special machine - called a *learning network machine* - before having
defined the new model type. Indeed doing so is a necessary step in
the export process, for this machine will tell the export macro:

- what kind of model the composite will be (`Deterministic`,
  `Probabilistic` or `Unsupervised`)
- which source nodes are input nodes and which are for the target
- which nodes correspond to each operation (`predict`, `transform`,
  etc) that we might want to define
"""

# ╔═╡ 16c9795f-76f5-4d5a-bdce-321ebc6fdbfa
surrogate = Probabilistic()  	# a model with no fields!

# ╔═╡ 7ca171cd-cda4-4160-b372-8c9866e51852
surrogate_mach = machine(surrogate, X_node, y_node; predict = yhat_node)

# ╔═╡ 2d279802-6d07-4ec1-8f3d-b82f1e7b6f7a
md"""
Although we have no real need to use it, this machine behaves like
you'd expect it to:
"""

# ╔═╡ d3074fee-772d-4a64-ba88-50ecf52f7ca8
 Xnew2, _ = make_blobs(2, 3)

# ╔═╡ 76570a4e-4667-4509-b042-114a1161ee86
fit!(surrogate_mach)

# ╔═╡ 8d826787-f719-4d4b-afa8-7f3fd8240442
predict(surrogate_mach, Xnew2)

# ╔═╡ 79f7e1b3-5648-430d-86cf-35420d9e6995
md"""
Now we create a new model type using a Julia `struct` definition
appropriately decorated:
"""

# ╔═╡ fd95ece8-48d5-4bb8-a29a-8d0445351914
@from_network surrogate_mach begin
    mutable struct YourPipe
        standardizer = stand
        classifier = linear::Probabilistic
    end
end

# ╔═╡ 92c5cc35-1dde-4457-be65-28d0fcca50a8
md"Instantiating and evaluating on some new data:"

# ╔═╡ ee4593ba-0d78-487f-9a2e-c468345d87d1
begin
  my_pipe = YourPipe()
  XIris, yIris = @load_iris;   # built-in data set
  my_pipe_mach = machine(my_pipe, XIris, yIris)
  evaluate!(my_pipe_mach, measure = misclassification_rate, operation = predict_mode)
end

# ╔═╡ c52c087e-1ac1-4de2-b5fa-cb79ebf595ef
md"## A composite model to average two regressor predictors"

# ╔═╡ 76116413-6bef-4727-91c5-524da38aa391
md"""
The following is condensed version of
[this](https://github.com/alan-turing-institute/MLJ.jl/blob/master/binder/MLJ_demo.ipynb)
tutorial. We will define a composite model that:
- standardizes the input data
- learns and applies a Box-Cox transformation to the target variable
- blends the predictions of two supervised learning models - a ridge regressor and a random forest regressor; we'll blend using a simple average (for a more sophisticated stacking example, see [here](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/stacking/))
- applies the *inverse* Box-Cox transformation to this blended prediction
"""

# ╔═╡ 63ae3550-6123-44ea-abf6-b9391d496ad7
 RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree

# ╔═╡ 74b64063-8584-418b-9cbc-9a2c39793333
RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels

# ╔═╡ c53ca29b-c401-458b-b887-2b16710e5502
md"### Input layer"

# ╔═╡ 6b25f155-5890-4bbd-91dc-ed41a586787c
XComp_node = source()

# ╔═╡ 8fbcc377-1bef-474b-bac9-011689013339
yComp_node = source()

# ╔═╡ fd9799d5-26f0-43d1-b017-96f1602f2cad
md"### First layer and target transformation"

# ╔═╡ 8e62d2f3-5dbb-4e3a-8be2-2a4017c45345
begin
  std_model = Standardizer()
  std_mach = machine(std_model, XComp_node)
  W = MLJ.transform(std_mach, XComp_node)
  
  box_model = UnivariateBoxCoxTransformer()
  box_mach = machine(box_model, yComp_node)
  z = MLJ.transform(box_mach, yComp_node)
end

# ╔═╡ 04cacf81-e396-463e-a7ad-80d84f5b0070
md"### Second layer"

# ╔═╡ 297612c4-9944-4dc5-8378-5aa686f0b407
begin
  ridge_model = RidgeRegressor(lambda=0.1)
  ridge_mach = machine(ridge_model, W, z)
  
  forest_model = RandomForestRegressor(n_trees=50)
  forest_mach = machine(forest_model, W, z)
  
  ẑ = 0.5 * predict(ridge_mach, W) + 0.5 * predict(forest_mach, W)
end

# ╔═╡ 863e750e-26b8-46e0-9f43-49763e8691a1
md"### Output"

# ╔═╡ 39001eed-a445-4f44-bb0e-3a45761c733a
ŷ = inverse_transform(box_mach, ẑ)

# ╔═╡ d8708b97-67f9-4143-9979-7044b2f2df33
md"With the learning network defined, we're ready to export:"

# ╔═╡ edd7008f-25a8-4613-b2a1-fbbde543f620
@from_network machine(Deterministic(), XComp_node, yComp_node, predict=ŷ) begin
    mutable struct CompositeModel
        rgs1 = ridge_model
        rgs2 = forest_model
    end
end

# ╔═╡ 16574eae-a0b7-4391-910b-f8a9a217eff2
md"Let's instantiate the new model type and try it out on some data:"

# ╔═╡ e4a3d173-fd0a-4529-aa36-d2e8546da46a
composite = CompositeModel()

# ╔═╡ c3896689-5775-4cd8-88a1-a2bf91434413
begin
  XBoston, yBoston = @load_boston;
  mach_comp = machine(composite, XBoston, yBoston);
  evaluate!(mach_comp,
            resampling=CV(nfolds=6, shuffle=true),
            measures=[rms, mae])
end

# ╔═╡ 3524e197-2629-4ee7-a1cb-54e34396a855
md"""
# Resources for Part 5

- From the MLJ manual:
   - [Learning Networks](https://alan-turing-institute.github.io/MLJ.jl/stable/composing_models/#Learning-Networks-1)
- From Data Science Tutorials:
    - [Learning Networks](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/learning-networks/)
    - [Learning Networks 2](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/learning-networks-2/)
  - [Stacking](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/stacking/): an advanced example of model composition
  - [Finer Control](https://alan-turing-institute.github.io/MLJ.jl/dev/composing_models/#Method-II:-Finer-control-(advanced)-1):
      exporting learning networks without a macro for finer control
"""

# ╔═╡ ac4a8d2e-411c-43d5-b7cc-46f4ef988c68
html"<a id='solutions-to-exercises'></a>"

# ╔═╡ 135dac9b-0bd9-4e1d-90fb-7f2721f6fcc7
md"""
---

*This notebook was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
MLJ = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
MLJDecisionTreeInterface = "c6f25543-311c-4c74-83dc-3ea6d1015661"
MLJLinearModels = "6ee0df7b-362f-4a72-a706-9e79364fb692"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
MLJ = "~0.17.0"
MLJDecisionTreeInterface = "~0.1.3"
MLJLinearModels = "~0.6.0"
PlutoUI = "~0.7.29"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ARFFFiles]]
deps = ["CategoricalArrays", "Dates", "Parsers", "Tables"]
git-tree-sha1 = "e8c8e0a2be6eb4f56b1672e46004463033daa409"
uuid = "da404889-ca92-49ff-9e8b-0aa6b4d38dc8"
version = "1.4.1"

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "d0d82f1c0b651173a4f839d84f662d03f3417740"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "4.0.0"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[BSON]]
git-tree-sha1 = "ebcd6e22d69f21249b7b8668351ebf42d6dc87a1"
uuid = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
version = "0.3.4"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "c308f209870fdbd84cb20332b6dfaf14bf3387f8"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.2"

[[CategoricalDistributions]]
deps = ["CategoricalArrays", "Distributions", "Missings", "OrderedCollections", "Random", "ScientificTypesBase", "UnicodePlots"]
git-tree-sha1 = "a5734a58e5dc8c749b5507d03ba5e457d077181b"
uuid = "af321ab8-2d2e-40a6-b165-3d674595d28e"
version = "0.1.4"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "926870acb6cbcf029396f2f2de030282b6bc1941"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.4"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[Crayons]]
git-tree-sha1 = "b618084b49e78985ffa8422f32b9838e397b9fc2"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.0"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DecisionTree]]
deps = ["DelimitedFiles", "Distributed", "LinearAlgebra", "Random", "ScikitLearnBase", "Statistics", "Test"]
git-tree-sha1 = "123adca1e427dc8abc5eec5040644e7842d53c92"
uuid = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb"
version = "0.10.11"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "9bc5dac3c8b6706b58ad5ce24cffd9861f07c94f"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.9.0"

[[Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "6a8dc9f82e5ce28279b6e3e2cea9421154f5bd0d"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.37"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarlyStopping]]
deps = ["Dates", "Statistics"]
git-tree-sha1 = "98fdf08b707aaf69f524a6cd0a67858cefe0cfb6"
uuid = "792122b4-ca99-40de-a6bc-6742525f08b6"
version = "0.3.0"

[[FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "04d13bfa8ef11720c24e4d840c0033d145537df7"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.17"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "b374f22e8565a01d6e5db1e8640c3c5e3fe7d564"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.9.0"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2b72a5624e289ee18256111657663721d59c143e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.24"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterationControl]]
deps = ["EarlyStopping", "InteractiveUtils"]
git-tree-sha1 = "83c84b7b87d3063e48a909a86c3c5bf4c3521962"
uuid = "b3c1a2ee-3fec-4384-bf48-272ea71de57c"
version = "0.5.2"

[[IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1169632f425f79429f245113b775a0e3d121457c"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.2"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JLSO]]
deps = ["BSON", "CodecZlib", "FilePathsBase", "Memento", "Pkg", "Serialization"]
git-tree-sha1 = "e00feb9d56e9e8518e0d60eef4d1040b282771e2"
uuid = "9da8a3cd-07a3-59c0-a743-3fdc52c30d11"
version = "2.6.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[LatinHypercubeSampling]]
deps = ["Random", "StableRNGs", "StatsBase", "Test"]
git-tree-sha1 = "42938ab65e9ed3c3029a8d2c58382ca75bdab243"
uuid = "a5e1c1ea-c99a-51d3-a14d-a9a37257b02d"
version = "1.8.0"

[[LearnBase]]
git-tree-sha1 = "a0d90569edd490b82fdc4dc078ea54a5a800d30a"
uuid = "7f8f8fb0-2700-5f03-b4bd-41f8cfc144b6"
version = "0.4.1"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LinearMaps]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "dbb14c604fc47aa4f2e19d0ebb7b6416f3cfa5f5"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.5.1"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[LossFunctions]]
deps = ["InteractiveUtils", "LearnBase", "Markdown", "RecipesBase", "StatsBase"]
git-tree-sha1 = "0f057f6ea90a84e73a8ef6eebb4dc7b5c330020f"
uuid = "30fc2ffe-d236-52d8-8643-a9d8f7c094a7"
version = "0.7.2"

[[MLJ]]
deps = ["CategoricalArrays", "ComputationalResources", "Distributed", "Distributions", "LinearAlgebra", "MLJBase", "MLJEnsembles", "MLJIteration", "MLJModels", "MLJSerialization", "MLJTuning", "OpenML", "Pkg", "ProgressMeter", "Random", "ScientificTypes", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "3b4ebc5023cc039c65a1089e6d8c248a9b96dfd1"
uuid = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
version = "0.17.0"

[[MLJBase]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Dates", "DelimitedFiles", "Distributed", "Distributions", "InteractiveUtils", "InvertedIndices", "LinearAlgebra", "LossFunctions", "MLJModelInterface", "Missings", "OrderedCollections", "Parameters", "PrettyTables", "ProgressMeter", "Random", "ScientificTypes", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "100d3500b1616088858eb541680c9cc954a53aa7"
uuid = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
version = "0.19.3"

[[MLJDecisionTreeInterface]]
deps = ["DecisionTree", "MLJModelInterface", "Random"]
git-tree-sha1 = "e2a5e2f0fd72cae51d72a83e6c11167de96c7a4c"
uuid = "c6f25543-311c-4c74-83dc-3ea6d1015661"
version = "0.1.3"

[[MLJEnsembles]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Distributed", "Distributions", "MLJBase", "MLJModelInterface", "ProgressMeter", "Random", "ScientificTypesBase", "StatsBase"]
git-tree-sha1 = "4279437ccc8ece8f478ded5139334b888dcce631"
uuid = "50ed68f4-41fd-4504-931a-ed422449fee0"
version = "0.2.0"

[[MLJIteration]]
deps = ["IterationControl", "MLJBase", "Random"]
git-tree-sha1 = "5f32c3d281904d6e5fc64250f55732d4b24014de"
uuid = "614be32b-d00c-4edb-bd02-1eb411ab5e55"
version = "0.4.1"

[[MLJLinearModels]]
deps = ["DocStringExtensions", "IterativeSolvers", "LinearAlgebra", "LinearMaps", "MLJModelInterface", "Optim", "Parameters"]
git-tree-sha1 = "6537fd96eb429c6e11a03c5910880c6da1837488"
uuid = "6ee0df7b-362f-4a72-a706-9e79364fb692"
version = "0.6.0"

[[MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "7ffdd75b2b13d1ec8640bfe80ab81bb158910a1d"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.3.5"

[[MLJModels]]
deps = ["CategoricalArrays", "CategoricalDistributions", "Dates", "Distances", "Distributions", "InteractiveUtils", "LinearAlgebra", "MLJModelInterface", "OrderedCollections", "Parameters", "Pkg", "PrettyPrinting", "REPL", "Random", "Requires", "ScientificTypes", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "cecd98731368f1eb46634d1476f49332560f886f"
uuid = "d491faf4-2d78-11e9-2867-c94bc002c0b7"
version = "0.15.1"

[[MLJSerialization]]
deps = ["IterationControl", "JLSO", "MLJBase", "MLJModelInterface"]
git-tree-sha1 = "cc5877ad02ef02e273d2622f0d259d628fa61cd0"
uuid = "17bed46d-0ab5-4cd4-b792-a5c4b8547c6d"
version = "1.1.3"

[[MLJTuning]]
deps = ["ComputationalResources", "Distributed", "Distributions", "LatinHypercubeSampling", "MLJBase", "ProgressMeter", "Random", "RecipesBase"]
git-tree-sha1 = "a443cc088158b949876d7038a1aa37cfc8c5509b"
uuid = "03970b2e-30c4-11ea-3135-d1576263f10f"
version = "0.6.16"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Memento]]
deps = ["Dates", "Distributed", "Requires", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "9b0b0dbf419fbda7b383dc12d108621d26eeb89f"
uuid = "f28f55f0-a522-5efc-85c2-fe41dfb9b2d9"
version = "1.3.0"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50310f934e55e5ca3912fb941dec199b49ca9b68"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.2"

[[NaNMath]]
git-tree-sha1 = "f755f36b19a5116bb580de457cda0c140153f283"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.6"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenML]]
deps = ["ARFFFiles", "HTTP", "JSON", "Markdown", "Pkg"]
git-tree-sha1 = "06080992e86a93957bfe2e12d3181443cedf2400"
uuid = "8b6db2d4-7670-4922-a472-f9537c81ab66"
version = "0.2.0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "916077e0f0f8966eb0dc98a5c39921fdb8f49eb4"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.6.0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ee26b350276c51697c9c2d88a072b339f9f03d73"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.5"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "d7fa6237da8004be601e19bd6666083056649918"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.3"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "7711172ace7c40dc8449b7aed9d2d6f1cf56a5bd"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.29"

[[PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "2cf929d64681236a2e074ffafb8d568733d2e6af"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.3"

[[PrettyPrinting]]
git-tree-sha1 = "a5db8a42938bc65c2679406c51a8f5fe9597c6e7"
uuid = "54e16d92-306c-5ea0-a30b-337be88ac337"
version = "0.3.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "8f82019e525f4d5c669692772a6f4b0a58b06a6a"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.2.0"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[ScientificTypes]]
deps = ["CategoricalArrays", "ColorTypes", "Dates", "Distributions", "PrettyTables", "Reexport", "ScientificTypesBase", "StatisticalTraits", "Tables"]
git-tree-sha1 = "ba70c9a6e4c81cc3634e3e80bb8163ab5ef57eb8"
uuid = "321657f4-b219-11e9-178b-2701a2544e81"
version = "3.0.0"

[[ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[ScikitLearnBase]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "7877e55c1523a4b336b433da39c8e8c08d2f221f"
uuid = "6e75b9c4-186b-50bd-896f-2d2496a4843e"
version = "0.5.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e08890d19787ec25029113e88c34ec20cac1c91e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.0.0"

[[StableRNGs]]
deps = ["Random", "Test"]
git-tree-sha1 = "3be7d49667040add7ee151fefaf1f8c04c8c8276"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.0"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "b4912cd034cdf968e06ca5f943bb54b17b97793a"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.5.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "88a559da57529581472320892576a486fa2377b9"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.3.1"

[[StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "271a7fea12d319f23d55b785c51f6876aadb9ac0"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.0.0"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "d88665adc9bcf45903013af0982e2fd05ae3d0a6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "51383f2d367eb3b444c961d485c565e4c0cf4ba0"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.14"

[[StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "bedb3e17cc1d94ce0e6e66d3afa47157978ba404"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.14"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "bb1064c9a84c52e277f1096cf41434b675cd368b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.1"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodePlots]]
deps = ["Crayons", "Dates", "SparseArrays", "StatsBase"]
git-tree-sha1 = "3cb994143aba28cfe66615702505b2d294cebd3e"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "2.5.1"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─24abafc3-0e26-4cd2-9bca-7c69b794f8ce
# ╟─0b8dcb85-3804-4c9b-b20a-319ba81ba28c
# ╟─b67c62c2-15ab-4328-b795-033f6f2a0674
# ╟─bc689638-fd19-4c9f-935f-ddf6a6bfbbdd
# ╟─197fd00e-9068-46ea-af2a-25235e544a31
# ╠═f6d4f8c4-e441-45c4-8af5-148d95ea2900
# ╟─45740c4d-b789-45dc-a6bf-47194d7e8e12
# ╟─42b0f1e1-16c9-4238-828a-4cc485149963
# ╠═d09256dd-6c0d-4e28-9e54-3f7b3ca87ecb
# ╟─499cbc31-83ba-4583-ba1f-6363f43ec697
# ╟─db5821e8-956a-4a46-95ea-2955abd45275
# ╟─38fdb776-44b5-4df5-b1b3-dd47e367ba54
# ╟─23408dac-0ade-41f4-8d7e-8e449afd1c48
# ╟─a6faf8b4-f0a5-4b16-a949-7f3bd292fe31
# ╟─bf69ff35-6d45-418e-9873-1b1430e635cc
# ╟─3d480ea1-bba7-49de-b43e-d0ebe87da176
# ╠═72801926-9f1d-4131-be77-3b4fa617d234
# ╠═cebd7290-ff85-4ebd-a89d-0aa1492fb1e0
# ╟─9e3a5fcf-8959-440c-abd3-cb77d7a79683
# ╟─88ca15eb-49f8-4525-879e-dc128f3db7b3
# ╠═0b842f20-acd3-40ba-a4f7-edbe2341eb94
# ╟─c720911a-478b-4aa0-bf32-94657e65284e
# ╠═48692370-0c4b-4080-9afd-65a5b5facac9
# ╟─b398bc7f-bf22-4bb5-b6c8-54f5ed90a964
# ╠═e193a088-1262-4b0f-8a29-32f11452654a
# ╟─2f187af9-21d8-4c18-a5f1-151fcbe229a2
# ╠═e13fd789-3016-4f88-bb6d-5087ebe511ce
# ╠═857a0ae3-346f-4d49-b395-4770aab7d47b
# ╠═ee7f00ab-6195-4a65-9ba5-46e06ec18ce5
# ╠═50265a35-512f-4a2f-9d87-1d1fbb0e3997
# ╟─b53fc776-08cc-4b9d-b950-fddef2a1fb10
# ╠═ac92e1ee-e50c-4408-951b-64cb2a36c8e3
# ╟─4b9f5009-0e61-4ef7-b0e6-441461cc8770
# ╠═1ed5139a-2618-418d-9adf-109bee706090
# ╠═84d84a02-ae06-4f28-8cb0-de601961bc02
# ╟─c6d367a8-e734-4c1b-a87d-950c50fb2955
# ╠═16c9795f-76f5-4d5a-bdce-321ebc6fdbfa
# ╠═7ca171cd-cda4-4160-b372-8c9866e51852
# ╟─2d279802-6d07-4ec1-8f3d-b82f1e7b6f7a
# ╠═d3074fee-772d-4a64-ba88-50ecf52f7ca8
# ╠═76570a4e-4667-4509-b042-114a1161ee86
# ╠═8d826787-f719-4d4b-afa8-7f3fd8240442
# ╟─79f7e1b3-5648-430d-86cf-35420d9e6995
# ╠═fd95ece8-48d5-4bb8-a29a-8d0445351914
# ╟─92c5cc35-1dde-4457-be65-28d0fcca50a8
# ╠═ee4593ba-0d78-487f-9a2e-c468345d87d1
# ╟─c52c087e-1ac1-4de2-b5fa-cb79ebf595ef
# ╟─76116413-6bef-4727-91c5-524da38aa391
# ╠═63ae3550-6123-44ea-abf6-b9391d496ad7
# ╠═74b64063-8584-418b-9cbc-9a2c39793333
# ╟─c53ca29b-c401-458b-b887-2b16710e5502
# ╠═6b25f155-5890-4bbd-91dc-ed41a586787c
# ╠═8fbcc377-1bef-474b-bac9-011689013339
# ╟─fd9799d5-26f0-43d1-b017-96f1602f2cad
# ╠═8e62d2f3-5dbb-4e3a-8be2-2a4017c45345
# ╟─04cacf81-e396-463e-a7ad-80d84f5b0070
# ╠═297612c4-9944-4dc5-8378-5aa686f0b407
# ╟─863e750e-26b8-46e0-9f43-49763e8691a1
# ╠═39001eed-a445-4f44-bb0e-3a45761c733a
# ╟─d8708b97-67f9-4143-9979-7044b2f2df33
# ╠═edd7008f-25a8-4613-b2a1-fbbde543f620
# ╟─16574eae-a0b7-4391-910b-f8a9a217eff2
# ╠═e4a3d173-fd0a-4529-aa36-d2e8546da46a
# ╠═c3896689-5775-4cd8-88a1-a2bf91434413
# ╟─3524e197-2629-4ee7-a1cb-54e34396a855
# ╟─ac4a8d2e-411c-43d5-b7cc-46f4ef988c68
# ╟─135dac9b-0bd9-4e1d-90fb-7f2721f6fcc7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
