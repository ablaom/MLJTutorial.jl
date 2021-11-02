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

# ╔═╡ 9b411d84-e7c7-40cd-b1b3-dd47e367ba54
md"## Part 4 - Tuning Hyper-parameters"

# ╔═╡ 45322ddf-ef46-4aa4-8d7e-8e449afd1c48
md"### Naive tuning of a single parameter"

# ╔═╡ 731ddf87-ae2f-40d9-a949-7f3bd292fe31
md"""
The most naive way to tune a single hyper-parameter is to use
`learning_curve`, which we already saw in Part 2. Let's see this in
the Horse Colic classification problem, a case where the parameter
to be tuned is *nested* (because the model is a pipeline).
"""

# ╔═╡ 5799d7b3-252f-415b-8514-99938a2932db
md"""
Here is the Horse Colic data again, with the type coercions we
already discussed in Part 1:
"""

# ╔═╡ 73e46348-0db9-4602-a0de-1721c1bc2df2
begin
  using MLJ
  using UrlDownload, CSV, DataFrames
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

# ╔═╡ ecf1bf78-63b4-402d-bca8-7eec7950fd82
md"Now for a pipeline model:"

# ╔═╡ 435a88e1-58ff-4745-9873-1b1430e635cc
begin
  LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
  model = @pipeline Standardizer ContinuousEncoder LogisticClassifier
  mach = machine(model, X, y)
end

# ╔═╡ ead28370-421b-47e4-b43e-d0ebe87da176
r = range(model, :(logistic_classifier.lambda), lower = 1e-2, upper=100, scale=:log10)

# ╔═╡ 1efc60d7-6ab3-495f-9009-4cfb2012998f
md"""
If you're curious, you can see what `lambda` values this range will
generate for a given resolution:
"""

# ╔═╡ 9a5cfd4a-e24a-4f94-abd3-cb77d7a79683
iterator(r, 5)

# ╔═╡ 35c29324-e4c7-4f54-879e-dc128f3db7b3
begin
  using Plots
  _, _, lambdas, losses = learning_curve(mach,
                                         range=r,
                                         resampling=CV(nfolds=6),
                                         resolution=30, # default
                                         measure=cross_entropy)
  plt=plot(lambdas, losses, xscale=:log10)
  xlabel!(plt, "lambda")
  ylabel!(plt, "cross entropy using 6-fold CV")
  savefig("learning_curve2.png")
  plt
  
  
  best_lambda = lambdas[argmin(losses)]
end

# ╔═╡ f0bfd7bc-4d12-47f7-a367-9aa3c6cf34d0
md"### Self tuning models"

# ╔═╡ c320ab9f-9da8-42e5-bf32-94657e65284e
md"""
A more sophisticated way to view hyper-parameter tuning (inspired by
MLR) is as a model *wrapper*. The wrapped model is a new model in
its own right and when you fit it, it tunes specified
hyper-parameters of the model being wrapped, before training on all
supplied data. Calling `predict` on the wrapped model is like
calling `predict` on the original model, but with the
hyper-parameters already optimized.
"""

# ╔═╡ ff3b9c3c-c167-48fe-9afd-65a5b5facac9
md"""
In other words, we can think of the wrapped model as a "self-tuning"
version of the original.
"""

# ╔═╡ ec263569-c58a-42e2-b6c8-54f5ed90a964
md"""
We now create a self-tuning version of the pipeline above, adding a
parameter from the `ContinuousEncoder` to the parameters we want
optimized.
"""

# ╔═╡ 52b902bb-cb92-46c4-9292-fe822525fc77
md"""
First, let's choose a tuning strategy (from [these
options](https://github.com/juliaai/MLJTuning.jl#what-is-provided-here)). MLJ
supports ordinary `Grid` search (query `?Grid` for
details). However, as the utility of `Grid` search is limited to a
small number of parameters, and as `Grid` searches are demonstrated
elsewhere (see the [resources below](#resources-for-part-4)) we'll
demonstrate `RandomSearch` here:
"""

# ╔═╡ 93c17a9b-b49c-4780-ae5e-4b83dcbc9675
tuning = RandomSearch(rng=123)

# ╔═╡ 37c4eff3-6c8f-4a6f-8a29-32f11452654a
md"""
In this strategy each parameter is sampled according to a
pre-specified prior distribution that is fit to the one-dimensional
range object constructed using `range` as before. While one has a
lot of control over the specification of the priors (run
`?RandomSearch` for details) we'll let the algorithm generate these
priors automatically.
"""

# ╔═╡ 36226e73-ceda-4615-a5f1-151fcbe229a2
md"#### Unbounded ranges and sampling"

# ╔═╡ c18143b8-7d91-4ab1-81bc-2d5603785a09
md"""
In MLJ a range does not have to be bounded. In a `RandomSearch` a
positive unbounded range is sampled using a `Gamma` distribution, by
default:
"""

# ╔═╡ de3e3484-0bc5-44a2-9d87-1d1fbb0e3997
r = range(model,
          :(logistic_classifier.lambda),
          lower=0,
          origin=6,
          unit=5,
          scale=:log10)

# ╔═╡ 8729b9a8-ac7a-485a-b950-fddef2a1fb10
md"""
The `scale` in a range makes no in a `RandomSearch` (unless it is a
function) but this will effect later plots but it does effect the
later plots.
"""

# ╔═╡ 28aaf005-792c-46a4-951b-64cb2a36c8e3
md"""
Let's see what sampling using a Gamma distribution is going to mean
for this range:
"""

# ╔═╡ 3204bcb1-e102-425a-b0e6-441461cc8770
begin
  import Distributions
  sampler_r = sampler(r, Distributions.Gamma)
  plt = histogram(rand(sampler_r, 10000), nbins=50)
  savefig("gamma_sampler.png")
  plt
end

# ╔═╡ 5e18a767-3617-4758-8cb0-de601961bc02
md"""
The second parameter that we'll add to this is *nominal* (finite) and, by
default, will be sampled uniformly. Since it is nominal, we specify
`values` instead of `upper` and `lower` bounds:
"""

# ╔═╡ de42c92d-9589-4601-a87d-950c50fb2955
s  = range(model, :(continuous_encoder.one_hot_ordered_factors),
           values = [true, false])

# ╔═╡ 6de90c1f-d93a-4d48-8448-4bcb089096cd
md"#### The tuning wrapper"

# ╔═╡ ab5eab32-155b-4de6-a012-a84240254fb6
md"Now for the wrapper, which is an instance of `TunedModel`:"

# ╔═╡ 09aa31d3-891c-4b2f-bbdd-9799f7bb2e60
tuned_model = TunedModel(model=model,
                         ranges=[r, s],
                         resampling=CV(nfolds=6),
                         measures=cross_entropy,
                         tuning=tuning,
                         n=15)

# ╔═╡ c0646c16-50ba-4e87-97a7-88e3af4f10ee
md"""
We can apply the `fit!/predict` work-flow to `tuned_model` just as
for any other model:
"""

# ╔═╡ 6f7ffb78-d44d-4123-b372-8c9866e51852
begin
  tuned_mach = machine(tuned_model, X, y);
  fit!(tuned_mach);
  predict(tuned_mach, rows=1:3)
end

# ╔═╡ 22392335-e988-49b9-8f3d-b82f1e7b6f7a
md"""
The outcomes of the tuning can be inspected from a detailed
report. For example, we have:
"""

# ╔═╡ e938da11-be0e-4c4a-ab04-579e5608ae53
begin
  rep = report(tuned_mach);
  rep.best_model
end

# ╔═╡ f3738cb7-872f-4618-86cf-35420d9e6995
md"By default, sampling of a bounded range is uniform. Lets"

# ╔═╡ 5b72bfac-40e2-4f84-a29a-8d0445351914
md"In the special case of two-parameters, you can also plot the results:"

# ╔═╡ 89dfc4b1-37dc-4f77-be65-28d0fcca50a8
begin
  plt = plot(tuned_mach)
  savefig("tuning.png")
  plt
end

# ╔═╡ efa202a5-5256-473c-9a2e-c468345d87d1
md"""
Finally, let's compare cross-validation estimate of the performance
of the self-tuning model with that of the original model (an example
of [*nested
resampling*]((https://mlr.mlr-org.com/articles/tutorial/nested_resampling.html)
here):
"""

# ╔═╡ 806471e8-a333-4879-b5fa-cb79ebf595ef
err = evaluate!(mach, resampling=CV(nfolds=3), measure=cross_entropy)

# ╔═╡ d37847b8-f78a-47ed-91c5-524da38aa391
tuned_err = evaluate!(tuned_mach, resampling=CV(nfolds=3), measure=cross_entropy)

# ╔═╡ dd8c9d2d-a433-4996-ad90-b4405b216771
md"<a id='resources-for-part-4'></a>"

# ╔═╡ f322e560-911b-498a-895b-997a92b731e0
md"""
### Resources for Part 4

- From the MLJ manual:
   - [Learning Curves](https://alan-turing-institute.github.io/MLJ.jl/dev/learning_curves/)
   - [Tuning Models](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/)
- The [MLJTuning repo](https://github.com/juliaai/MLJTuning.jl#who-is-this-repo-for) - mostly for developers

- From Data Science Tutorials:
    - [Tuning a model](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/model-tuning/)
    - [Crabs with XGBoost](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/crabs-xgb/) `Grid` tuning in stages for a tree-boosting model with many parameters
    - [Boston with LightGBM](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/boston-lgbm/) -  `Grid` tuning for another popular tree-booster
    - [Boston with Flux](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/boston-flux/) - optimizing batch size in a simple neural network regressor
- [UCI Horse Colic Data Set](http://archive.ics.uci.edu/ml/datasets/Horse+Colic)
"""

# ╔═╡ d1d05669-87ef-41d1-a525-e9b04a4bd246
md"### Exercises for Part 4"

# ╔═╡ 5846d352-3641-4754-80f0-ba7781e173cf
md"#### Exercise 8"

# ╔═╡ acb95992-5052-4d9e-9cbc-9a2c39793333
md"""
This exercise continues our analysis of the King County House price
prediction problem (Part 1, Exercise 3 and Part 2):
"""

# ╔═╡ c78372ed-6214-49cd-b887-2b16710e5502
begin
  house_csv = urldownload("https://raw.githubusercontent.com/ablaom/"*
                          "MachineLearningInJulia2020/for-MLJ-version-0.16/"*
                          "data/house.csv");
  house = DataFrames.DataFrame(house_csv)
  coerce!(house, autotype(house_csv));
  coerce!(house, Count => Continuous, :zipcode => Multiclass);
  y, X = unpack(house, ==(:price), name -> true, rng=123);
  schema(X)
end

# ╔═╡ c216fce7-5cec-4c52-9452-5298a8a4a401
md"""
Your task will be to tune the following pipeline regression model,
which includes a gradient tree boosting component:
"""

# ╔═╡ 3479d87e-248c-4908-b017-96f1602f2cad
begin
  EvoTreeRegressor = @load EvoTreeRegressor
  tree_booster = EvoTreeRegressor(nrounds = 70)
  model = @pipeline ContinuousEncoder tree_booster
end

# ╔═╡ e8ebca0d-06ca-4686-8be2-2a4017c45345
md"""
(a) Construct a bounded range `r1` for the `evo_tree_booster`
parameter `max_depth`, varying between 1 and 12.
"""

# ╔═╡ 51f26da2-ed46-4cbd-a7ad-80d84f5b0070
md"\\star&(b) For the `nbins` parameter of the `EvoTreeRegressor`, define the range"

# ╔═╡ 8ed3b5dd-d8af-4b51-8378-5aa686f0b407
r2 = range(model,
           :(evo_tree_regressor.nbins),
           lower = 2.5,
           upper= 7.5, scale=x->2^round(Int, x))

# ╔═╡ 8572a486-f629-4ec6-9f43-49763e8691a1
md"""
Notice that in this case we've specified a *function* instead of a
canned scale, like `:log10`. In this case the `scale` function is
applied after sampling (uniformly) between the limits of `lower` and
`upper`. Perhaps you can guess the outputs of the following lines of
code?
"""

# ╔═╡ 04ada702-d9b2-48b8-bb0e-3a45761c733a
begin
  r2_sampler = sampler(r2, Distributions.Uniform)
  samples = rand(r2_sampler, 1000);
  plt = histogram(samples, nbins=50)
  savefig("uniform_sampler.png")
  
  plt
end

# ╔═╡ ea77596a-b8f1-4cee-9979-7044b2f2df33
md"![](uniform_sampler.png)"

# ╔═╡ 0ed3d037-1f3b-4fcc-b2a1-fbbde543f620
sort(unique(samples))

# ╔═╡ b994a419-1a4d-469e-910b-f8a9a217eff2
md"""
(c) Optimize `model` over these the parameter ranges `r1` and `r2`
using a random search with uniform priors (the default). Use
`Holdout()` resampling, and implement your search by first
constructing a "self-tuning" wrap of `model`, as described
above. Make `mae` (mean absolute error) the loss function that you
optimize, and search over a total of 40 combinations of
hyper-parameters.  If you have time, plot the results of your
search. Feel free to use all available data.
"""

# ╔═╡ d6c7f929-bd6b-428d-aa36-d2e8546da46a
md"""
(d) Evaluate the best model found in the search using 3-fold
cross-validation and compare with that of the self-tuning model
(which is different!). Setting data hygiene concerns aside, feel
free to use all available data.
"""

# ╔═╡ 4243d219-c486-4daf-88a1-a2bf91434413
md"<a id='part-5-advanced-model-composition'>"

# ╔═╡ 135dac9b-0bd9-4e1d-a1cb-54e34396a855
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
# ╟─9b411d84-e7c7-40cd-b1b3-dd47e367ba54
# ╟─45322ddf-ef46-4aa4-8d7e-8e449afd1c48
# ╟─731ddf87-ae2f-40d9-a949-7f3bd292fe31
# ╟─5799d7b3-252f-415b-8514-99938a2932db
# ╠═73e46348-0db9-4602-a0de-1721c1bc2df2
# ╟─ecf1bf78-63b4-402d-bca8-7eec7950fd82
# ╠═435a88e1-58ff-4745-9873-1b1430e635cc
# ╠═ead28370-421b-47e4-b43e-d0ebe87da176
# ╟─1efc60d7-6ab3-495f-9009-4cfb2012998f
# ╠═9a5cfd4a-e24a-4f94-abd3-cb77d7a79683
# ╠═35c29324-e4c7-4f54-879e-dc128f3db7b3
# ╟─f0bfd7bc-4d12-47f7-a367-9aa3c6cf34d0
# ╟─c320ab9f-9da8-42e5-bf32-94657e65284e
# ╟─ff3b9c3c-c167-48fe-9afd-65a5b5facac9
# ╟─ec263569-c58a-42e2-b6c8-54f5ed90a964
# ╟─52b902bb-cb92-46c4-9292-fe822525fc77
# ╠═93c17a9b-b49c-4780-ae5e-4b83dcbc9675
# ╟─37c4eff3-6c8f-4a6f-8a29-32f11452654a
# ╟─36226e73-ceda-4615-a5f1-151fcbe229a2
# ╟─c18143b8-7d91-4ab1-81bc-2d5603785a09
# ╠═de3e3484-0bc5-44a2-9d87-1d1fbb0e3997
# ╟─8729b9a8-ac7a-485a-b950-fddef2a1fb10
# ╟─28aaf005-792c-46a4-951b-64cb2a36c8e3
# ╠═3204bcb1-e102-425a-b0e6-441461cc8770
# ╟─5e18a767-3617-4758-8cb0-de601961bc02
# ╠═de42c92d-9589-4601-a87d-950c50fb2955
# ╟─6de90c1f-d93a-4d48-8448-4bcb089096cd
# ╟─ab5eab32-155b-4de6-a012-a84240254fb6
# ╠═09aa31d3-891c-4b2f-bbdd-9799f7bb2e60
# ╟─c0646c16-50ba-4e87-97a7-88e3af4f10ee
# ╠═6f7ffb78-d44d-4123-b372-8c9866e51852
# ╟─22392335-e988-49b9-8f3d-b82f1e7b6f7a
# ╠═e938da11-be0e-4c4a-ab04-579e5608ae53
# ╟─f3738cb7-872f-4618-86cf-35420d9e6995
# ╟─5b72bfac-40e2-4f84-a29a-8d0445351914
# ╠═89dfc4b1-37dc-4f77-be65-28d0fcca50a8
# ╟─efa202a5-5256-473c-9a2e-c468345d87d1
# ╠═806471e8-a333-4879-b5fa-cb79ebf595ef
# ╠═d37847b8-f78a-47ed-91c5-524da38aa391
# ╟─dd8c9d2d-a433-4996-ad90-b4405b216771
# ╟─f322e560-911b-498a-895b-997a92b731e0
# ╟─d1d05669-87ef-41d1-a525-e9b04a4bd246
# ╟─5846d352-3641-4754-80f0-ba7781e173cf
# ╟─acb95992-5052-4d9e-9cbc-9a2c39793333
# ╠═c78372ed-6214-49cd-b887-2b16710e5502
# ╟─c216fce7-5cec-4c52-9452-5298a8a4a401
# ╠═3479d87e-248c-4908-b017-96f1602f2cad
# ╟─e8ebca0d-06ca-4686-8be2-2a4017c45345
# ╟─51f26da2-ed46-4cbd-a7ad-80d84f5b0070
# ╠═8ed3b5dd-d8af-4b51-8378-5aa686f0b407
# ╟─8572a486-f629-4ec6-9f43-49763e8691a1
# ╠═04ada702-d9b2-48b8-bb0e-3a45761c733a
# ╟─ea77596a-b8f1-4cee-9979-7044b2f2df33
# ╠═0ed3d037-1f3b-4fcc-b2a1-fbbde543f620
# ╟─b994a419-1a4d-469e-910b-f8a9a217eff2
# ╟─d6c7f929-bd6b-428d-aa36-d2e8546da46a
# ╟─4243d219-c486-4daf-88a1-a2bf91434413
# ╟─135dac9b-0bd9-4e1d-a1cb-54e34396a855
