### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# ╔═╡ 24abafc3-0e26-4cd2-9bca-7c69b794f8ce
md"# Machine Learning in Julia"

# ╔═╡ b67c62c2-15ab-4328-b795-033f6f2a0674
md"""
An introduction to the
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/stable/)
toolbox.
"""

# ╔═╡ bc689638-fd19-4c9f-935f-ddf6a6bfbbdd
md"### Set-up"

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

# ╔═╡ d09256dd-6c0d-4e28-9e54-3f7b3ca87ecb
begin
  using Pkg
  Pkg.activate("env")
  Pkg.instantiate()
end

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
md"## Part 3 - Transformers and Pipelines"

# ╔═╡ d055d24f-1b2d-48a6-8d7e-8e449afd1c48
md"### Transformers"

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

# ╔═╡ ad63ad4b-04c0-491b-a0de-1721c1bc2df2
begin
  using MLJ
  
  x = rand(100);
  @show mean(x) std(x);
end

# ╔═╡ f86b62eb-6853-482a-bca8-7eec7950fd82
begin
  model = Standardizer() # a built-in model
  mach = machine(model, x)
  fit!(mach)
  xhat = transform(mach, x);
  @show mean(xhat) std(xhat);
end

# ╔═╡ 58686fe6-088a-4751-9873-1b1430e635cc
md"This particular model has an `inverse_transform`:"

# ╔═╡ 8b90b209-cb98-4116-b43e-d0ebe87da176
inverse_transform(mach, xhat) ≈ x

# ╔═╡ f1b47e75-0ce3-4e90-9009-4cfb2012998f
md"### Re-encoding the King County House data as continuous"

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
  using UrlDownload, CSV, DataFrames
  house_csv = urldownload("https://raw.githubusercontent.com/ablaom/"*
                          "MachineLearningInJulia2020/for-MLJ-version-0.16/"*
                          "data/house.csv");
  house = DataFrames.DataFrame(house_csv)
  coerce!(house, autotype(house_csv));
  coerce!(house, Count => Continuous, :zipcode => Multiclass);
  schema(house)
end

# ╔═╡ 348ff9a2-89ab-434a-9afd-65a5b5facac9
y, X = unpack(house, ==(:price), name -> true, rng=123);

# ╔═╡ c7e83ce7-f067-4049-b6c8-54f5ed90a964
md"Instantiate the unsupervised model (transformer):"

# ╔═╡ a6106065-1baf-4720-9292-fe822525fc77
encoder = ContinuousEncoder() # a built-in model; no need to @load it

# ╔═╡ 703d4ac3-116e-4e05-ae5e-4b83dcbc9675
md"Bind the model to the data and fit!"

# ╔═╡ 5d976f7e-45ee-4cbb-8a29-32f11452654a
mach = machine(encoder, X) |> fit!;

# ╔═╡ aad55579-14f3-439a-a5f1-151fcbe229a2
md"Transform and inspect the result:"

# ╔═╡ 9f2f583a-e90a-43f6-81bc-2d5603785a09
begin
  Xcont = transform(mach, X);
  schema(Xcont)
end

# ╔═╡ 828460a7-7d4e-444e-9d87-1d1fbb0e3997
md"### More transformers"

# ╔═╡ 12631f23-8262-475d-b950-fddef2a1fb10
md"Here's how to list all of MLJ's unsupervised models:"

# ╔═╡ d9069be8-d8c5-43ab-951b-64cb2a36c8e3
models(m->!m.is_supervised)

# ╔═╡ 4dee7ef6-b472-4b9c-b0e6-441461cc8770
md"Some commonly used ones are built-in (do not require `@load`ing):"

# ╔═╡ 7a6df9e6-89f1-4117-8cb0-de601961bc02
md"""
model type                  | does what?
----------------------------|----------------------------------------------
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
md"### Pipelines"

# ╔═╡ b454845b-3b3b-4c46-a012-a84240254fb6
length(schema(Xcont).names)

# ╔═╡ 28cf44d4-3f4b-485f-bbdd-9799f7bb2e60
md"""
Let's suppose that additionally we'd like to reduce the dimension of
our data.  A model that will do this is `PCA` from
`MultivariateStats.jl`:
"""

# ╔═╡ 1e780939-abf0-4203-97a7-88e3af4f10ee
begin
  PCA = @load PCA
  reducer = PCA()
end

# ╔═╡ 8e76b5a5-7c80-4684-b372-8c9866e51852
md"""
Now, rather simply repeating the work-flow above, applying the new
transformation to `Xcont`, we can combine both the encoding and the
dimension-reducing models into a single model, known as a
*pipeline*. While MLJ offers a powerful interface for composing
models in a variety of ways, we'll stick to these simplest class of
composite models for now. The easiest way to construct them is using
the `@pipeline` macro:
"""

# ╔═╡ f11094fb-b54b-4bb2-8f3d-b82f1e7b6f7a
pipe = @pipeline encoder reducer

# ╔═╡ 4afc9625-b262-48e0-ab04-579e5608ae53
md"""
Notice that `pipe` is an *instance* of an automatically generated
type (called `Pipeline<some digits>`).
"""

# ╔═╡ e68044d8-42b4-49cd-86cf-35420d9e6995
md"The new model behaves like any other transformer:"

# ╔═╡ 26d649a8-3d96-4479-a29a-8d0445351914
begin
  mach = machine(pipe, X)
  fit!(mach)
  Xsmall = transform(mach, X)
  schema(Xsmall)
end

# ╔═╡ 7f60c74e-b708-46f4-be65-28d0fcca50a8
md"Want to combine this pre-processing with ridge regression?"

# ╔═╡ 5b5c1d2a-1f9b-4cc7-9a2e-c468345d87d1
begin
  RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels
  rgs = RidgeRegressor()
  pipe2 = @pipeline encoder reducer rgs
end

# ╔═╡ fbb24be1-fe01-4489-b5fa-cb79ebf595ef
md"""
Now our pipeline is a supervised model, instead of a transformer,
whose performance we can evaluate:
"""

# ╔═╡ ea64dbb5-963a-4da4-91c5-524da38aa391
begin
  mach = machine(pipe2, X, y)
  evaluate!(mach, measure=mae, resampling=Holdout()) # CV(nfolds=6) is default
end

# ╔═╡ 1ce36a90-ea25-48d6-ad90-b4405b216771
md"### Training of composite models is \"smart\""

# ╔═╡ 4270976f-1207-4086-895b-997a92b731e0
md"""
Now notice what happens if we train on all the data, then change a
regressor hyper-parameter and retrain:
"""

# ╔═╡ 1d0bce6b-ffff-4f30-a525-e9b04a4bd246
fit!(mach)

# ╔═╡ 64a648e0-d5c9-4c3d-80f0-ba7781e173cf
begin
  pipe2.ridge_regressor.lambda = 0.1
  fit!(mach)
end

# ╔═╡ 5dcd0d55-bb5d-459a-9cbc-9a2c39793333
md"Second time only the ridge regressor is retrained!"

# ╔═╡ 234c586e-862f-4216-b887-2b16710e5502
md"""
Mutate a hyper-parameter of the `PCA` model and every model except
the `ContinuousEncoder` (which comes before it will be retrained):
"""

# ╔═╡ b8e84a2c-2c7f-41f5-9452-5298a8a4a401
begin
  pipe2.pca.pratio = 0.9999
  fit!(mach)
end

# ╔═╡ 95b75a71-47b5-49b1-b017-96f1602f2cad
md"### Inspecting composite models"

# ╔═╡ 5de9f9a0-ad8f-47de-8be2-2a4017c45345
md"""
The dot syntax used above to change the values of *nested*
hyper-parameters is also useful when inspecting the learned
parameters and report generated when training a composite model:
"""

# ╔═╡ aa7bdff7-7d1b-43b1-a7ad-80d84f5b0070
fitted_params(mach).ridge_regressor

# ╔═╡ 92c5bc62-b430-41b8-8378-5aa686f0b407
report(mach).pca

# ╔═╡ adcb0506-e89d-43b0-9f43-49763e8691a1
md"### Incorporating target transformations"

# ╔═╡ 667917e3-25d2-435c-bb0e-3a45761c733a
md"""
Next, suppose that instead of using the raw `:price` as the
training target, we want to use the log-price (a common practice in
dealing with house price data). However, suppose that we still want
to report final *predictions* on the original linear scale (and use
these for evaluation purposes). Then we supply appropriate functions
to key-word arguments `target` and `inverse`.
"""

# ╔═╡ 72d314eb-74ef-4555-9979-7044b2f2df33
md"First we'll overload `log` and `exp` for broadcasting:"

# ╔═╡ bc2bdda0-8fe3-4ef0-b2a1-fbbde543f620
begin
  Base.log(v::AbstractArray) = log.(v)
  Base.exp(v::AbstractArray) = exp.(v)
end

# ╔═╡ 0398adfe-721a-4b97-910b-f8a9a217eff2
md"Now for the new pipeline:"

# ╔═╡ 943cbab6-cbe2-4054-aa36-d2e8546da46a
begin
  pipe3 = @pipeline encoder reducer rgs target=log inverse=exp
  mach = machine(pipe3, X, y)
  evaluate!(mach, measure=mae)
end

# ╔═╡ e98fef59-a5cb-4c24-88a1-a2bf91434413
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

# ╔═╡ 271f5dae-38ad-4e5b-8037-0de5806e1a54
begin
  box = UnivariateBoxCoxTransformer(n=20)
  stand = Standardizer()
  
  pipe4 = @pipeline encoder reducer rgs target=box
  mach = machine(pipe4, X, y)
  evaluate!(mach, measure=mae)
end

# ╔═╡ 3d8a2959-63c5-4bee-9961-a09632c33fb0
begin
  pipe4.target = stand
  evaluate!(mach, measure=mae)
end

# ╔═╡ a0488295-9642-4156-b7cc-46f4ef988c68
md"### Resources for Part 3"

# ╔═╡ d59d9e91-b26d-4342-90fb-7f2721f6fcc7
md"""
- From the MLJ manual:
    - [Transformers and other unsupervised models](https://alan-turing-institute.github.io/MLJ.jl/dev/transformers/)
    - [Linear pipelines](https://alan-turing-institute.github.io/MLJ.jl/dev/linear_pipelines/#Linear-Pipelines)
- From Data Science Tutorials:
    - [Composing models](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/composing-models/)
"""

# ╔═╡ c4f30527-c9b4-44e9-af65-ccd25ecb9818
md"### Exercises for Part 3"

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
  
  y, X = unpack(horse, ==(:outcome), name -> true);
  schema(X)
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
"""

# ╔═╡ 246a00c8-8c05-4ec6-b7bb-2f33ef765cc0
md"""
(b) Evaluate the pipeline on all data, using 6-fold cross-validation
and `cross_entropy` loss.
"""

# ╔═╡ 9cfb86c0-0283-4e85-9625-01172c4a0081
md"""
&star;(c) Plot a learning curve which examines the effect on this loss
as the tree booster parameter `max_depth` varies from 2 to 10.
"""

# ╔═╡ 6cce6530-0d12-4862-af4f-11c7de9e21dd
md"<a id='part-4-tuning-hyper-parameters'></a>"

# ╔═╡ 135dac9b-0bd9-4e1d-8dba-241d9b744683
md"""
---

*This notebook was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
"""

# ╔═╡ Cell order:
# ╟─24abafc3-0e26-4cd2-9bca-7c69b794f8ce
# ╟─b67c62c2-15ab-4328-b795-033f6f2a0674
# ╟─bc689638-fd19-4c9f-935f-ddf6a6bfbbdd
# ╟─197fd00e-9068-46ea-af2a-25235e544a31
# ╠═f6d4f8c4-e441-45c4-8af5-148d95ea2900
# ╟─45740c4d-b789-45dc-a6bf-47194d7e8e12
# ╟─42b0f1e1-16c9-4238-828a-4cc485149963
# ╠═d09256dd-6c0d-4e28-9e54-3f7b3ca87ecb
# ╟─499cbc31-83ba-4583-ba1f-6363f43ec697
# ╟─db5821e8-956a-4a46-95ea-2955abd45275
# ╟─1e248249-5629-412f-b1b3-dd47e367ba54
# ╟─d055d24f-1b2d-48a6-8d7e-8e449afd1c48
# ╟─0dc40881-6163-4948-a949-7f3bd292fe31
# ╟─0d066909-c9bf-4ae3-8514-99938a2932db
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
# ╠═f11094fb-b54b-4bb2-8f3d-b82f1e7b6f7a
# ╟─4afc9625-b262-48e0-ab04-579e5608ae53
# ╟─e68044d8-42b4-49cd-86cf-35420d9e6995
# ╠═26d649a8-3d96-4479-a29a-8d0445351914
# ╟─7f60c74e-b708-46f4-be65-28d0fcca50a8
# ╠═5b5c1d2a-1f9b-4cc7-9a2e-c468345d87d1
# ╟─fbb24be1-fe01-4489-b5fa-cb79ebf595ef
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
# ╟─72d314eb-74ef-4555-9979-7044b2f2df33
# ╠═bc2bdda0-8fe3-4ef0-b2a1-fbbde543f620
# ╟─0398adfe-721a-4b97-910b-f8a9a217eff2
# ╠═943cbab6-cbe2-4054-aa36-d2e8546da46a
# ╟─e98fef59-a5cb-4c24-88a1-a2bf91434413
# ╟─4fd0807d-61e5-4b4b-a1cb-54e34396a855
# ╠═271f5dae-38ad-4e5b-8037-0de5806e1a54
# ╠═3d8a2959-63c5-4bee-9961-a09632c33fb0
# ╟─a0488295-9642-4156-b7cc-46f4ef988c68
# ╟─d59d9e91-b26d-4342-90fb-7f2721f6fcc7
# ╟─c4f30527-c9b4-44e9-af65-ccd25ecb9818
# ╟─25a6fd3c-60d1-44dd-8890-979691212d9b
# ╟─aed2b274-cc0c-42f2-a6fa-eaf14df5d44b
# ╠═0df6be04-24fe-4417-8025-5084804a9f6c
# ╟─9abf25a8-7231-43f5-9e90-09023d201062
# ╟─246a00c8-8c05-4ec6-b7bb-2f33ef765cc0
# ╟─9cfb86c0-0283-4e85-9625-01172c4a0081
# ╟─6cce6530-0d12-4862-af4f-11c7de9e21dd
# ╟─135dac9b-0bd9-4e1d-8dba-241d9b744683
