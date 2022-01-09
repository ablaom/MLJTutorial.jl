### A Pluto.jl notebook ###
# v0.17.4

using Markdown
using InteractiveUtils

# ╔═╡ bbe90636-9d1f-4ee0-8e35-c285f7b922d0
begin
	import CSV, DataFrames
	using EvoTrees
  	using MLJ  
 	using MLJLinearModels
	using MLJBase
 	using MLJMultivariateStatsInterface
  	using UrlDownload
	using PlutoUI
end


# ╔═╡ 83a7b4d7-6442-44b2-9388-8330b19cd537
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
">Tutorial - Part 3</p>
<p style="text-align: left; font-size: 2.5rem;">
Machine Learning in Julia
</p>
"""

# ╔═╡ 24abafc3-0e26-4cd2-9bca-7c69b794f8ce
PlutoUI.TableOfContents(title = "MLJ Tutorial - Part 3")

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

# ╔═╡ 1e248249-5629-412f-b1b3-dd47e367ba54
md"# Part 3 - Transformers and Pipelines"

# ╔═╡ d055d24f-1b2d-48a6-8d7e-8e449afd1c48
md"## Transformers"

# ╔═╡ 0dc40881-6163-4948-a949-7f3bd292fe31
md"""
Unsupervised models, which receive no target `y` during training,
always have a `transform` operation. They sometimes also support an
`inverse_transform` operation, with obvious meaning, and sometimes
support a `predict` operation (see the clustering example discussed
[here](https://alan-turing-institute.github.io/MLJ.jl/dev/transformers/#Transformers-that-also-predict-1)).
Otherwise, they are handled much like supervised models.
"""

# ╔═╡ 0d066909-c9bf-4ae3-8514-99938a2932db
md"Here's a simple standardization example:"

# ╔═╡ 29ea6531-1337-4527-92bd-6ebccf75659a
 x = rand(100)

# ╔═╡ ad63ad4b-04c0-491b-a0de-1721c1bc2df2
(mean(x), std(x))

# ╔═╡ f86b62eb-6853-482a-bca8-7eec7950fd82
begin
  model = Standardizer() # a built-in model
  mach1 = machine(model, x)
  fit!(mach1)
  xhat = MLJ.transform(mach1, x);
  (mean(xhat), std(xhat))
end

# ╔═╡ 58686fe6-088a-4751-9873-1b1430e635cc
md"This particular model has an `inverse_transform`:"

# ╔═╡ 8b90b209-cb98-4116-b43e-d0ebe87da176
inverse_transform(mach1, xhat) ≈ x

# ╔═╡ f1b47e75-0ce3-4e90-9009-4cfb2012998f
md"## Re-encoding the King County House data as continuous"

# ╔═╡ e904fcb3-7603-40d2-abd3-cb77d7a79683
md"""
For further illustrations of transformers, let's re-encode *all* of the
King County House input features (see [Ex
3](#exercise-3-fixing-scitypes-in-a-table)) into a set of `Continuous`
features. We do this with the `ContinuousEncoder` model, which, by
default, will:
"""

# ╔═╡ e93966db-0dab-42ca-879e-dc128f3db7b3
md"""
- one-hot encode all `Multiclass` features
- coerce all `OrderedFactor` features to `Continuous` ones
- coerce all `Count` features to `Continuous` ones (there aren't any)
- drop any remaining non-Continuous features (none of these either)
"""

# ╔═╡ 2564f31e-4704-4820-a367-9aa3c6cf34d0
md"First, we reload the data and fix the scitypes (Exercise 3):"

# ╔═╡ 7e23f8f6-1634-45c8-bf32-94657e65284e
begin
 
  house_csv = urldownload("https://raw.githubusercontent.com/ablaom/"*
                          "MachineLearningInJulia2020/for-MLJ-version-0.16/"*
                          "data/house.csv");
  house = DataFrames.DataFrame(house_csv)
  coerce!(house, autotype(house_csv));
  coerce!(house, Count => Continuous, :zipcode => Multiclass);
  schema(house)
end

# ╔═╡ 348ff9a2-89ab-434a-9afd-65a5b5facac9
yHouse, XHouse = unpack(house, ==(:price), name -> true, rng=123);

# ╔═╡ c7e83ce7-f067-4049-b6c8-54f5ed90a964
md"Instantiate the unsupervised model (transformer):"

# ╔═╡ a6106065-1baf-4720-9292-fe822525fc77
encoder = ContinuousEncoder() # a built-in model; no need to @load it

# ╔═╡ 703d4ac3-116e-4e05-ae5e-4b83dcbc9675
md"Bind the model to the data and fit!"

# ╔═╡ 5d976f7e-45ee-4cbb-8a29-32f11452654a
mach2 = machine(encoder, XHouse) |> fit!

# ╔═╡ aad55579-14f3-439a-a5f1-151fcbe229a2
md"Transform and inspect the result:"

# ╔═╡ a6c1d3ea-6f43-42c7-8950-2f4327f23c1f
XHouseCont = MLJ.transform(mach2, XHouse)

# ╔═╡ 9f2f583a-e90a-43f6-81bc-2d5603785a09
schema(XHouseCont)

# ╔═╡ 828460a7-7d4e-444e-9d87-1d1fbb0e3997
md"## More transformers"

# ╔═╡ 12631f23-8262-475d-b950-fddef2a1fb10
md"Here's how to list all of MLJ's unsupervised models:"

# ╔═╡ d9069be8-d8c5-43ab-951b-64cb2a36c8e3
models(m->!m.is_supervised)

# ╔═╡ 4dee7ef6-b472-4b9c-b0e6-441461cc8770
md"Some commonly used ones are built-in (do not require `@load`ing):"

# ╔═╡ 7a6df9e6-89f1-4117-8cb0-de601961bc02
md"""
model type                  | does what?
:----------------------------|:----------------------------------------------
ContinuousEncoder | transform input table to a table of `Continuous` features (see above)
FeatureSelector | retain or dump selected features
FillImputer | impute missing values
OneHotEncoder | one-hot encoder `Multiclass` (and optionally `OrderedFactor`) features
Standardizer | standardize (whiten) a vector or all `Continuous` features of a table
UnivariateBoxCoxTransformer | apply a learned Box-Cox transformation to a vector
UnivariateDiscretizer | discretize a `Continuous` vector, and hence render its elscitypw `OrderedFactor`
"""

# ╔═╡ 8ff17b93-2830-41e2-a87d-950c50fb2955
md"""
In addition to "dynamic" transformers (ones that learn something
from the data and must be `fit!`) users can wrap ordinary functions
as transformers, and such *static* transformers can depend on
parameters, like the dynamic ones. See
[here](https://alan-turing-institute.github.io/MLJ.jl/dev/transformers/#Static-transformers-1)
for how to define your own static transformers.
"""

# ╔═╡ d45a81ec-f978-4483-8448-4bcb089096cd
md"## Pipelines"

# ╔═╡ b454845b-3b3b-4c46-a012-a84240254fb6
length(schema(XHouseCont).names)

# ╔═╡ 28cf44d4-3f4b-485f-bbdd-9799f7bb2e60
md"""
Let's suppose that additionally we'd like to reduce the dimension of
our data.  A model that will do this is `PCA` from
`MultivariateStats.jl`:
"""

# ╔═╡ 1e780939-abf0-4203-97a7-88e3af4f10ee
PCA = @load PCA

# ╔═╡ 8e76b5a5-7c80-4684-b372-8c9866e51852
md"""
Now, rather simply repeating the work-flow above, applying the new
transformation to `XHouseCont`, we can combine both the encoding and the
dimension-reducing models into a single model, known as a
*pipeline*. While MLJ offers a powerful interface for composing
models in a variety of ways, we'll stick to these simplest class of
composite models for now. The simplest "hard-wired" composite type is the `Pipeline` type, which is for linear (non-branching) sequences of models. At most one of these can be a supervised model (which often appears last):
"""

# ╔═╡ 5ed77309-afa6-4c94-88c0-761fe6b5a5d4
pipe0 = Pipeline(ContinuousEncoder, PCA)

# ╔═╡ d45ff6ec-2dd6-4a9c-a97e-79337b0be5c8
md"""
Notice that component models now appear as *hyper-parameters* of the pipeline model,  and these have automatically generated field names (which can be overwritten, as in `Pipeline(enc=ContinuousEncoder, reducer=PCA)`). There is also an "arrow" syntax for constructing pipelines. The following defines the same pipeline as above:
"""

# ╔═╡ 380a3bb6-535d-4f4b-8e7c-18148f7cc0b8
pipe1 = ContinuousEncoder |> PCA

# ╔═╡ eee1bcc8-ff0b-4238-950d-551d88aab415
md"""
!!! note 

    In the former macro-based version this was 
    
    `pipe1 = @pipeline encoder reducer`, 

    where `encoder` was instance of `ContinuousEncoder` and `reducer` an instance of `PCA`.
"""

# ╔═╡ e68044d8-42b4-49cd-86cf-35420d9e6995
md"The new model behaves like any other transformer:"

# ╔═╡ 26d649a8-3d96-4479-a29a-8d0445351914
begin
  mach3 = machine(pipe1, XHouse)
  fit!(mach3)
  XHouseSmall = transform(mach3, XHouse)
  schema(XHouseSmall)
end

# ╔═╡ 7f60c74e-b708-46f4-be65-28d0fcca50a8
md"Want to combine this pre-processing with ridge regression?"

# ╔═╡ c8c960f0-2a46-42c2-8754-8bd531c8db8e
RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels

# ╔═╡ f96a84ec-1cc0-400c-9493-9c2a2e3c5b51
pipe2 = pipe1 |> RidgeRegressor

# ╔═╡ e9a2070b-41b5-41bd-9994-5929469b8436
md"""
!!! note

    In the former macro-based version this was

    `pipe2 = @pipeline encoder reducer rgs`

    where `rgs` was an instance of `RidgeRegressor`.
"""

# ╔═╡ fbb24be1-fe01-4489-b5fa-cb79ebf595ef
md"""
Now our pipeline is a supervised model, instead of a transformer,
whose performance we can evaluate:
"""

# ╔═╡ 958f9a9a-23d5-446b-aa54-f7a086cb0264
mach4 = machine(pipe2, XHouse, yHouse)

# ╔═╡ ea64dbb5-963a-4da4-91c5-524da38aa391
evaluate!(mach4, measure=mae, resampling=Holdout()) # CV(nfolds=6) is default

# ╔═╡ 1ce36a90-ea25-48d6-ad90-b4405b216771
md"## Training of composite models is \"smart\""

# ╔═╡ 4270976f-1207-4086-895b-997a92b731e0
md"""
Now notice what happens if we train on all the data, then change a
regressor hyper-parameter and retrain:
"""

# ╔═╡ 1d0bce6b-ffff-4f30-a525-e9b04a4bd246
fit!(mach4)

# ╔═╡ 64a648e0-d5c9-4c3d-80f0-ba7781e173cf
with_terminal() do
  pipe2.ridge_regressor.lambda = 0.1
  fit!(mach4)
end

# ╔═╡ 5dcd0d55-bb5d-459a-9cbc-9a2c39793333
md"Second time only the ridge regressor is retrained!"

# ╔═╡ 234c586e-862f-4216-b887-2b16710e5502
md"""
Mutate a hyper-parameter of the `PCA` model and every model except
the `ContinuousEncoder` (which comes before it will be retrained):
"""

# ╔═╡ b8e84a2c-2c7f-41f5-9452-5298a8a4a401
with_terminal() do
  pipe2.pca.pratio = 0.9999
  fit!(mach4)
end

# ╔═╡ 95b75a71-47b5-49b1-b017-96f1602f2cad
md"## Inspecting composite models"

# ╔═╡ 5de9f9a0-ad8f-47de-8be2-2a4017c45345
md"""
The dot syntax used above to change the values of *nested*
hyper-parameters is also useful when inspecting the learned
parameters and report generated when training a composite model:
"""

# ╔═╡ aa7bdff7-7d1b-43b1-a7ad-80d84f5b0070
fitted_params(mach4).ridge_regressor

# ╔═╡ 92c5bc62-b430-41b8-8378-5aa686f0b407
report(mach4).pca

# ╔═╡ adcb0506-e89d-43b0-9f43-49763e8691a1
md"## Incorporating target transformations"

# ╔═╡ 667917e3-25d2-435c-bb0e-3a45761c733a
md"""
Next, suppose that instead of using the raw `:price` as the
training target, we want to use the log-price (a common practice in
dealing with house price data). However, suppose that we still want
to report final *predictions* on the original linear scale (and use
these for evaluation purposes). Then we supply appropriate functions
to key-word arguments `target` and `inverse`.
"""

# ╔═╡ f71e4957-b6aa-480f-8acf-0c1d237fe9f9
md"""
First we'll overload `log` and `exp` for broadcasting:
"""

# ╔═╡ dc119abd-72a1-4781-811d-5998beb56406
begin
	Base.log(v::AbstractArray) = log.(v)
	Base.exp(v::AbstractArray) = exp.(v)
end

# ╔═╡ 0398adfe-721a-4b97-910b-f8a9a217eff2
md"Now for the new pipeline:"

# ╔═╡ d1af0582-9074-4355-9d03-7ceec3db5d7f
pipe3 = @pipeline ContinuousEncoder PCA RidgeRegressor target=log inverse=exp

# ╔═╡ 06fac83d-ec17-4503-b3b1-670e16d4251a
mach5 = machine(pipe3, XHouse, yHouse)

# ╔═╡ 9335bbb6-7169-47a2-8ccb-85c839a68433
evaluate!(mach5, measure = mae)

# ╔═╡ fbe0163a-ed06-49ea-8faa-017fa7aeca69
md"""
MLJ will also allow you to insert *learned* target
transformations. For example, we might want to apply
`Standardizer()` to the target, to standardize it, or
`UnivariateBoxCoxTransformer()` to make it look Gaussian. Then
instead of specifying a *function* for `target`, we specify a
unsupervised *model* (or model type). One does not specify `inverse`
because only models implementing `inverse_transform` are
allowed.
"""

# ╔═╡ 4fd0807d-61e5-4b4b-a1cb-54e34396a855
md"Let's see which of these two options results in a better outcome:"

# ╔═╡ 1957f387-aee4-4c08-9838-327b915082f8
box = UnivariateBoxCoxTransformer(n=20)

# ╔═╡ 7b59b396-daac-4dea-bc51-54fafe346cd8
stand = Standardizer()

# ╔═╡ dcf51860-a300-453e-87fc-345e741f94d0
pipe4 = @pipeline ContinuousEncoder PCA RidgeRegressor target=box

# ╔═╡ c54f41ee-75e6-4181-8de7-d194004c9700
mach6 = machine(pipe4, XHouse, yHouse)

# ╔═╡ cb026565-6c5c-4f7a-91b6-417cfea74d41
evaluate!(mach6, measure = mae)

# ╔═╡ e073f936-f2a0-4609-83fe-bd8afb87cfc1
pipe4.target = stand

# ╔═╡ 91cca9cb-db5f-4d9f-983b-3f2b20444662
evaluate!(mach6, measure = mae)

# ╔═╡ a0488295-9642-4156-b7cc-46f4ef988c68
md"# Resources for Part 3"

# ╔═╡ d59d9e91-b26d-4342-90fb-7f2721f6fcc7
md"""
- From the MLJ manual:
    - [Transformers and other unsupervised models](https://alan-turing-institute.github.io/MLJ.jl/dev/transformers/)
    - [Linear pipelines](https://alan-turing-institute.github.io/MLJ.jl/dev/linear_pipelines/#Linear-Pipelines)
- From Data Science Tutorials:
    - [Composing models](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/composing-models/)
"""

# ╔═╡ c4f30527-c9b4-44e9-af65-ccd25ecb9818
md"# Exercises for Part 3"

# ╔═╡ 25a6fd3c-60d1-44dd-8890-979691212d9b
md"#### Exercise 7"

# ╔═╡ aed2b274-cc0c-42f2-a6fa-eaf14df5d44b
md"""
Consider again the Horse Colic classification problem considered in
Exercise 6, but with all features, `Finite` and `Infinite`:
"""

# ╔═╡ 0df6be04-24fe-4417-8025-5084804a9f6c
begin
  csv_file = urldownload("https://raw.githubusercontent.com/ablaom/"*
                     "MachineLearningInJulia2020/"*
                     "for-MLJ-version-0.16/data/horse.csv");
  horse = DataFrames.DataFrame(csv_file); # convert to data frame
  coerce!(horse, autotype(horse));
  coerce!(horse, Count => Continuous);
  coerce!(horse,
          :surgery               => Multiclass,
          :age                   => Multiclass,
          :mucous_membranes      => Multiclass,
          :capillary_refill_time => Multiclass,
          :outcome               => Multiclass,
          :cp_data               => Multiclass);
  
  yHorse, XHorse = unpack(horse, ==(:outcome), name -> true);
  schema(XHorse)
end

# ╔═╡ 9abf25a8-7231-43f5-9e90-09023d201062
md"""
(a) Define a pipeline that:
- uses `Standardizer` to ensure that features that are already
  continuous are centered at zero and have unit variance
- re-encodes the full set of features as `Continuous`, using
  `ContinuousEncoder`
- uses the `KMeans` clustering model from `Clustering.jl`
  to reduce the dimension of the feature space to `k=10`.
- trains a `EvoTreeClassifier` (a gradient tree boosting
  algorithm in `EvoTrees.jl`) on the reduced data, using
  `nrounds=50` and default values for the other
   hyper-parameters

(b) Evaluate the pipeline on all data, using 6-fold cross-validation
and `cross_entropy` loss.

 $\star$ (c) Plot a learning curve which examines the effect on this loss
as the tree booster parameter `max_depth` varies from 2 to 10.
"""

# ╔═╡ 6cce6530-0d12-4862-af4f-11c7de9e21dd
html"<a id='part-4-tuning-hyper-parameters'></a>"

# ╔═╡ 135dac9b-0bd9-4e1d-8dba-241d9b744683
md"""
---

*This notebook was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
EvoTrees = "f6006082-12f8-11e9-0c9c-0d5d367ab1e5"
MLJ = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
MLJBase = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
MLJLinearModels = "6ee0df7b-362f-4a72-a706-9e79364fb692"
MLJMultivariateStatsInterface = "1b6a4a23-ba22-4f51-9698-8599985d3728"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
UrlDownload = "856ac37a-3032-4c1c-9122-f86d88358c8b"

[compat]
CSV = "~0.9.11"
DataFrames = "~1.3.1"
EvoTrees = "~0.9.1"
MLJ = "~0.17.0"
MLJBase = "~0.19.2"
MLJLinearModels = "~0.5.7"
MLJMultivariateStatsInterface = "~0.2.2"
PlutoUI = "~0.7.27"
UrlDownload = "~1.0.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ARFFFiles]]
deps = ["CategoricalArrays", "Dates", "Parsers", "Tables"]
git-tree-sha1 = "e8c8e0a2be6eb4f56b1672e46004463033daa409"
uuid = "da404889-ca92-49ff-9e8b-0aa6b4d38dc8"
version = "1.4.1"

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "2ff92b71ba1747c5fdd541f8fc87736d82f40ec9"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.4.0"

[[Arpack_jll]]
deps = ["Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "e214a9b9bd1b4e1b4f15b22c0994862b66af7ff7"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.0+3"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "1ee88c4c76caa995a885dc2f22a5d548dfbbc0ba"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.2.2"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "a598ecb0d717092b5539dbbe890c98bac842b072"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.2.0"

[[BSON]]
git-tree-sha1 = "ebcd6e22d69f21249b7b8668351ebf42d6dc87a1"
uuid = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
version = "0.3.4"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "49f14b6c56a2da47608fe30aed711b5882264d7a"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.9.11"

[[CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "429a1a05348ce948a96adbdd873fbe6d9e5e052f"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "3.6.2"

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

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "cfdfef912b7f93e4b848e80b9befdf9e331bc05a"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.1"

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

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[EarlyStopping]]
deps = ["Dates", "Statistics"]
git-tree-sha1 = "98fdf08b707aaf69f524a6cd0a67858cefe0cfb6"
uuid = "792122b4-ca99-40de-a6bc-6742525f08b6"
version = "0.3.0"

[[EvoTrees]]
deps = ["BSON", "CUDA", "CategoricalArrays", "Distributions", "MLJModelInterface", "NetworkLayout", "Random", "RecipesBase", "SpecialFunctions", "StaticArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "d1d91fd0643d6eb7946c15115b625b0b1e1a5393"
uuid = "f6006082-12f8-11e9-0c9c-0d5d367ab1e5"
version = "0.9.1"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

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
git-tree-sha1 = "8b3c09b56acaf3c0e581c66638b85c8650ee9dca"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.8.1"

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

[[GPUArrays]]
deps = ["Adapt", "LinearAlgebra", "Printf", "Random", "Serialization", "Statistics"]
git-tree-sha1 = "d9681e61fbce7dde48684b40bdb1a319c4083be7"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.1.3"

[[GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "2cac236070c2c4b36de54ae9146b55ee2c34ac7a"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.13.10"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

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

[[InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "8d70835a3759cdd75881426fced1508bb7b7e1b6"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.1"

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

[[IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

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

[[LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "7cc22e69995e2329cc047a879395b2b74647ab5f"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.7.0"

[[LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c5fc4bef251ecd37685bea1c4068a9cfa41e8b9a"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.13+0"

[[LatinHypercubeSampling]]
deps = ["Random", "StableRNGs", "StatsBase", "Test"]
git-tree-sha1 = "42938ab65e9ed3c3029a8d2c58382ca75bdab243"
uuid = "a5e1c1ea-c99a-51d3-a14d-a9a37257b02d"
version = "1.8.0"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

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
git-tree-sha1 = "54cae1f0bde7bbc72fe7ff42353b7880347bd0d5"
uuid = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
version = "0.19.2"

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
git-tree-sha1 = "dfbf4a5a8454034d21b6cfd9fd5a7960c8f7fb88"
uuid = "6ee0df7b-362f-4a72-a706-9e79364fb692"
version = "0.5.7"

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

[[MLJMultivariateStatsInterface]]
deps = ["Distances", "LinearAlgebra", "MLJModelInterface", "MultivariateStats", "StatsBase"]
git-tree-sha1 = "0cfc81ff677ea13ed72894992ee9e5f8ae4dbb9d"
uuid = "1b6a4a23-ba22-4f51-9698-8599985d3728"
version = "0.2.2"

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

[[MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "8d958ff1854b166003238fe191ec34b9d592860a"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.8.0"

[[NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50310f934e55e5ca3912fb941dec199b49ca9b68"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.2"

[[NaNMath]]
git-tree-sha1 = "f755f36b19a5116bb580de457cda0c140153f283"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.6"

[[NetworkLayout]]
deps = ["GeometryBasics", "LinearAlgebra", "Random", "Requires", "SparseArrays"]
git-tree-sha1 = "cac8fc7ba64b699c678094fa630f49b80618f625"
uuid = "46757867-2c16-5918-afeb-47bfcb05e46a"
version = "0.4.4"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

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
git-tree-sha1 = "fed057115644d04fba7f4d768faeeeff6ad11a60"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.27"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "db3a23166af8aebf4db5ef87ac5b00d36eb771e2"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.0"

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

[[Random123]]
deps = ["Libdl", "Random", "RandomNumbers"]
git-tree-sha1 = "0e8b146557ad1c6deb1367655e052276690e71a3"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.4.2"

[[RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

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

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "244586bc07462d22aed0113af9c731f2a518c93e"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.10"

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
git-tree-sha1 = "7f5a513baec6f122401abfc8e9c074fdac54f6c1"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.4.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "de9e88179b584ba9cf3cc5edbb7a41f26ce42cda"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.3.0"

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

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

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

[[TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "7cb456f358e8f9d102a8b25e8dfedf58fa5689bc"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.13"

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

[[UrlDownload]]
deps = ["HTTP", "ProgressMeter"]
git-tree-sha1 = "05f86730c7a53c9da603bd506a4fc9ad0851171c"
uuid = "856ac37a-3032-4c1c-9122-f86d88358c8b"
version = "1.0.0"

[[WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "c69f9da3ff2f4f02e811c3323c22e5dfcb584cfa"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.1"

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
# ╟─83a7b4d7-6442-44b2-9388-8330b19cd537
# ╟─24abafc3-0e26-4cd2-9bca-7c69b794f8ce
# ╟─b67c62c2-15ab-4328-b795-033f6f2a0674
# ╟─bc689638-fd19-4c9f-935f-ddf6a6bfbbdd
# ╟─197fd00e-9068-46ea-af2a-25235e544a31
# ╠═f6d4f8c4-e441-45c4-8af5-148d95ea2900
# ╟─45740c4d-b789-45dc-a6bf-47194d7e8e12
# ╟─42b0f1e1-16c9-4238-828a-4cc485149963
# ╠═bbe90636-9d1f-4ee0-8e35-c285f7b922d0
# ╟─499cbc31-83ba-4583-ba1f-6363f43ec697
# ╟─db5821e8-956a-4a46-95ea-2955abd45275
# ╟─1e248249-5629-412f-b1b3-dd47e367ba54
# ╟─d055d24f-1b2d-48a6-8d7e-8e449afd1c48
# ╟─0dc40881-6163-4948-a949-7f3bd292fe31
# ╟─0d066909-c9bf-4ae3-8514-99938a2932db
# ╠═29ea6531-1337-4527-92bd-6ebccf75659a
# ╠═ad63ad4b-04c0-491b-a0de-1721c1bc2df2
# ╠═f86b62eb-6853-482a-bca8-7eec7950fd82
# ╟─58686fe6-088a-4751-9873-1b1430e635cc
# ╠═8b90b209-cb98-4116-b43e-d0ebe87da176
# ╟─f1b47e75-0ce3-4e90-9009-4cfb2012998f
# ╟─e904fcb3-7603-40d2-abd3-cb77d7a79683
# ╟─e93966db-0dab-42ca-879e-dc128f3db7b3
# ╟─2564f31e-4704-4820-a367-9aa3c6cf34d0
# ╠═7e23f8f6-1634-45c8-bf32-94657e65284e
# ╠═348ff9a2-89ab-434a-9afd-65a5b5facac9
# ╟─c7e83ce7-f067-4049-b6c8-54f5ed90a964
# ╠═a6106065-1baf-4720-9292-fe822525fc77
# ╟─703d4ac3-116e-4e05-ae5e-4b83dcbc9675
# ╠═5d976f7e-45ee-4cbb-8a29-32f11452654a
# ╟─aad55579-14f3-439a-a5f1-151fcbe229a2
# ╠═a6c1d3ea-6f43-42c7-8950-2f4327f23c1f
# ╠═9f2f583a-e90a-43f6-81bc-2d5603785a09
# ╟─828460a7-7d4e-444e-9d87-1d1fbb0e3997
# ╟─12631f23-8262-475d-b950-fddef2a1fb10
# ╠═d9069be8-d8c5-43ab-951b-64cb2a36c8e3
# ╟─4dee7ef6-b472-4b9c-b0e6-441461cc8770
# ╟─7a6df9e6-89f1-4117-8cb0-de601961bc02
# ╟─8ff17b93-2830-41e2-a87d-950c50fb2955
# ╟─d45a81ec-f978-4483-8448-4bcb089096cd
# ╠═b454845b-3b3b-4c46-a012-a84240254fb6
# ╟─28cf44d4-3f4b-485f-bbdd-9799f7bb2e60
# ╠═1e780939-abf0-4203-97a7-88e3af4f10ee
# ╟─8e76b5a5-7c80-4684-b372-8c9866e51852
# ╠═5ed77309-afa6-4c94-88c0-761fe6b5a5d4
# ╟─d45ff6ec-2dd6-4a9c-a97e-79337b0be5c8
# ╠═380a3bb6-535d-4f4b-8e7c-18148f7cc0b8
# ╟─eee1bcc8-ff0b-4238-950d-551d88aab415
# ╟─e68044d8-42b4-49cd-86cf-35420d9e6995
# ╠═26d649a8-3d96-4479-a29a-8d0445351914
# ╟─7f60c74e-b708-46f4-be65-28d0fcca50a8
# ╠═c8c960f0-2a46-42c2-8754-8bd531c8db8e
# ╠═f96a84ec-1cc0-400c-9493-9c2a2e3c5b51
# ╟─e9a2070b-41b5-41bd-9994-5929469b8436
# ╟─fbb24be1-fe01-4489-b5fa-cb79ebf595ef
# ╠═958f9a9a-23d5-446b-aa54-f7a086cb0264
# ╠═ea64dbb5-963a-4da4-91c5-524da38aa391
# ╟─1ce36a90-ea25-48d6-ad90-b4405b216771
# ╟─4270976f-1207-4086-895b-997a92b731e0
# ╠═1d0bce6b-ffff-4f30-a525-e9b04a4bd246
# ╠═64a648e0-d5c9-4c3d-80f0-ba7781e173cf
# ╟─5dcd0d55-bb5d-459a-9cbc-9a2c39793333
# ╟─234c586e-862f-4216-b887-2b16710e5502
# ╠═b8e84a2c-2c7f-41f5-9452-5298a8a4a401
# ╟─95b75a71-47b5-49b1-b017-96f1602f2cad
# ╟─5de9f9a0-ad8f-47de-8be2-2a4017c45345
# ╠═aa7bdff7-7d1b-43b1-a7ad-80d84f5b0070
# ╠═92c5bc62-b430-41b8-8378-5aa686f0b407
# ╟─adcb0506-e89d-43b0-9f43-49763e8691a1
# ╟─667917e3-25d2-435c-bb0e-3a45761c733a
# ╟─f71e4957-b6aa-480f-8acf-0c1d237fe9f9
# ╠═dc119abd-72a1-4781-811d-5998beb56406
# ╟─0398adfe-721a-4b97-910b-f8a9a217eff2
# ╠═d1af0582-9074-4355-9d03-7ceec3db5d7f
# ╠═06fac83d-ec17-4503-b3b1-670e16d4251a
# ╠═9335bbb6-7169-47a2-8ccb-85c839a68433
# ╟─fbe0163a-ed06-49ea-8faa-017fa7aeca69
# ╟─4fd0807d-61e5-4b4b-a1cb-54e34396a855
# ╠═1957f387-aee4-4c08-9838-327b915082f8
# ╠═7b59b396-daac-4dea-bc51-54fafe346cd8
# ╠═dcf51860-a300-453e-87fc-345e741f94d0
# ╠═c54f41ee-75e6-4181-8de7-d194004c9700
# ╠═cb026565-6c5c-4f7a-91b6-417cfea74d41
# ╠═e073f936-f2a0-4609-83fe-bd8afb87cfc1
# ╠═91cca9cb-db5f-4d9f-983b-3f2b20444662
# ╟─a0488295-9642-4156-b7cc-46f4ef988c68
# ╟─d59d9e91-b26d-4342-90fb-7f2721f6fcc7
# ╟─c4f30527-c9b4-44e9-af65-ccd25ecb9818
# ╟─25a6fd3c-60d1-44dd-8890-979691212d9b
# ╟─aed2b274-cc0c-42f2-a6fa-eaf14df5d44b
# ╠═0df6be04-24fe-4417-8025-5084804a9f6c
# ╟─9abf25a8-7231-43f5-9e90-09023d201062
# ╟─6cce6530-0d12-4862-af4f-11c7de9e21dd
# ╟─135dac9b-0bd9-4e1d-8dba-241d9b744683
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
