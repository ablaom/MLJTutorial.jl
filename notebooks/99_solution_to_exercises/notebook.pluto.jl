### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# ╔═╡ 0f97c6bb-9ced-4cbe-9bca-7c69b794f8ce
md"# Machine Learning in Julia (conclusion)"

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

# ╔═╡ 6b1758e2-96e8-4272-b1b3-dd47e367ba54
md"## Solutions to exercises"

# ╔═╡ c7449612-1de3-436a-8d7e-8e449afd1c48
using MLJ, UrlDownload, CSV, DataFrames, Plots

# ╔═╡ 945f2179-e861-4ae7-a949-7f3bd292fe31
md"#### Exercise 2 solution"

# ╔═╡ e62654a0-cc6e-4072-8514-99938a2932db
md"From the question statememt:"

# ╔═╡ 7f9003b5-95c7-418b-a0de-1721c1bc2df2
quality = ["good", "poor", "poor", "excellent", missing, "good", "excellent"]

# ╔═╡ c037caf3-700e-4cc8-bca8-7eec7950fd82
begin
  quality = coerce(quality, OrderedFactor);
  levels!(quality, ["poor", "good", "excellent"]);
  elscitype(quality)
end

# ╔═╡ e9be221c-06a1-41bc-9873-1b1430e635cc
md"#### Exercise 3 solution"

# ╔═╡ 9f13b374-8918-4bae-b43e-d0ebe87da176
md"From the question statement:"

# ╔═╡ 9a417692-fea9-4606-9009-4cfb2012998f
begin
  house_csv = urldownload("https://raw.githubusercontent.com/ablaom/"*
                          "MachineLearningInJulia2020/for-MLJ-version-0.16/"*
                          "data/house.csv");
  house = DataFrames.DataFrame(house_csv)
end

# ╔═╡ e926dc2e-2b3b-4683-abd3-cb77d7a79683
md"First pass:"

# ╔═╡ cb767b7d-bc7f-4759-879e-dc128f3db7b3
begin
  coerce!(house, autotype(house));
  schema(house)
end

# ╔═╡ 0ec099e3-6e30-41f6-a367-9aa3c6cf34d0
md"""
All the "sqft" fields refer to "square feet" so are
really `Continuous`. We'll regard `:yr_built` (the other `Count`
variable above) as `Continuous` as well. So:
"""

# ╔═╡ 159c8967-ab87-4b2a-bf32-94657e65284e
coerce!(house, Count => Continuous);

# ╔═╡ f93fec06-260f-4693-9afd-65a5b5facac9
md"And `:zipcode` should not be ordered:"

# ╔═╡ 17809199-b8c5-4b29-b6c8-54f5ed90a964
begin
  coerce!(house, :zipcode => Multiclass);
  schema(house)
end

# ╔═╡ 8eaa2b50-c1f7-4f8b-9292-fe822525fc77
md"""
`:bathrooms` looks like it has a lot of levels, but on further
inspection we see why, and `OrderedFactor` remains appropriate:
"""

# ╔═╡ 0f5162b2-daf6-43a1-ae5e-4b83dcbc9675
begin
  import StatsBase.countmap
  countmap(house.bathrooms)
end

# ╔═╡ 44e87adc-146a-4da9-8a29-32f11452654a
md"#### Exercise 4 solution"

# ╔═╡ 9f13b374-8918-4bae-a5f1-151fcbe229a2
md"From the question statement:"

# ╔═╡ 1f822943-6f7a-4615-81bc-2d5603785a09
begin
  import Distributions
  poisson = Distributions.Poisson
  
  age = 18 .+ 60*rand(10);
  salary = coerce(rand(["small", "big", "huge"], 10), OrderedFactor);
  levels!(salary, ["small", "big", "huge"]);
  small = salary[1]
  
  X4 = DataFrames.DataFrame(age=age, salary=salary)
  
  n_devices(salary) = salary > small ? rand(poisson(1.3)) : rand(poisson(2.9))
  y4 = [n_devices(row.salary) for row in eachrow(X4)]
end

# ╔═╡ e3cc20c2-6492-45e0-9d87-1d1fbb0e3997
md"4(a)"

# ╔═╡ 63f952e4-5d3a-48ac-b950-fddef2a1fb10
md"There are *no* models that apply immediately:"

# ╔═╡ ae1cb765-ba67-496e-951b-64cb2a36c8e3
models(matching(X4, y4))

# ╔═╡ d30bc3a2-c019-406f-b0e6-441461cc8770
md"4(b)"

# ╔═╡ af31e97b-cfba-4caa-8cb0-de601961bc02
begin
  y4 = coerce(y4, Continuous);
  models(matching(X4, y4))
end

# ╔═╡ 08f6a160-3148-40fc-a87d-950c50fb2955
md"#### Exercise 6 solution"

# ╔═╡ 9f13b374-8918-4bae-8448-4bcb089096cd
md"From the question statement:"

# ╔═╡ 6df9f266-f9d9-4506-a012-a84240254fb6
begin
  using UrlDownload, CSV
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
  schema(horse)
end

# ╔═╡ 9a4ff2d2-c5b6-4488-bbdd-9799f7bb2e60
md"6(a)"

# ╔═╡ affb9923-2afa-4655-97a7-88e3af4f10ee
y, X = unpack(horse,
              ==(:outcome),
              name -> elscitype(Tables.getcolumn(horse, name)) == Continuous);

# ╔═╡ 9995fba2-e61b-4877-b372-8c9866e51852
md"6(b)(i)"

# ╔═╡ 29825df3-1e51-4b8d-8f3d-b82f1e7b6f7a
begin
  train, test = partition(eachindex(y), 0.7)
  model = (@load LogisticClassifier pkg=MLJLinearModels)();
  model.lambda = 100
  mach = machine(model, X, y)
  fit!(mach, rows=train)
  fitted_params(mach)
end

# ╔═╡ c01a3d86-3f7e-494b-ab04-579e5608ae53
begin
  coefs_given_feature = Dict(fitted_params(mach).coefs)
  coefs_given_feature[:pulse]
  
  #6(b)(ii)
  
  yhat = predict(mach, rows=test); # or predict(mach, X[test,:])
  err = cross_entropy(yhat, y[test]) |> mean
end

# ╔═╡ 1af34706-479e-42a8-86cf-35420d9e6995
md"6(b)(iii)"

# ╔═╡ be4e773b-284c-406f-a29a-8d0445351914
md"""
The predicted probabilities of the actual observations in the test
are given by
"""

# ╔═╡ f8bc8787-ea6a-4935-be65-28d0fcca50a8
p = broadcast(pdf, yhat, y[test]);

# ╔═╡ 7e046dfe-49ce-47e4-9a2e-c468345d87d1
md"The number of times this probability exceeds 50% is:"

# ╔═╡ 2f3f1724-0d11-4159-b5fa-cb79ebf595ef
n50 = filter(x -> x > 0.5, p) |> length

# ╔═╡ 4c06c69e-8a21-43d7-91c5-524da38aa391
md"Or, as a proportion:"

# ╔═╡ 89b9ee21-1940-4e5e-ad90-b4405b216771
n50/length(test)

# ╔═╡ 83fba101-4cf1-48aa-895b-997a92b731e0
md"6(b)(iv)"

# ╔═╡ 0fb6fa7a-f0c0-4926-a525-e9b04a4bd246
misclassification_rate(mode.(yhat), y[test])

# ╔═╡ e56be556-d75a-40c9-80f0-ba7781e173cf
md"6(c)(i)"

# ╔═╡ 28a64fbd-ceeb-44b3-9cbc-9a2c39793333
begin
  model = (@load RandomForestClassifier pkg=DecisionTree)()
  mach = machine(model, X, y)
  evaluate!(mach, resampling=CV(nfolds=6), measure=cross_entropy)
  
  r = range(model, :n_trees, lower=10, upper=70, scale=:log10)
end

# ╔═╡ 5d00e3fe-feb5-4394-b887-2b16710e5502
md"""
Since random forests are inherently randomized, we generate multiple
curves:
"""

# ╔═╡ c25a8ee7-61b3-4b07-9452-5298a8a4a401
begin
  plt = plot()
  for i in 1:4
      one_curve = learning_curve(mach,
                             range=r,
                             resampling=Holdout(),
                             measure=cross_entropy)
      plot!(one_curve.parameter_values, one_curve.measurements)
  end
  xlabel!(plt, "n_trees")
  ylabel!(plt, "cross entropy")
  savefig("exercise_6ci.png")
  plt
end

# ╔═╡ 0e8a02cd-4571-45d8-b017-96f1602f2cad
md"6(c)(ii)"

# ╔═╡ 5a351c52-5915-4e2b-8be2-2a4017c45345
begin
  evaluate!(mach, resampling=CV(nfolds=9),
                  measure=cross_entropy,
                  rows=train).measurement[1]
  
  model.n_trees = 90
end

# ╔═╡ faef6958-7f5d-42e4-a7ad-80d84f5b0070
md"6(c)(iii)"

# ╔═╡ 240987eb-4252-42e3-8378-5aa686f0b407
err_forest = evaluate!(mach, resampling=Holdout(),
                       measure=cross_entropy).measurement[1]

# ╔═╡ 25a6fd3c-60d1-44dd-9f43-49763e8691a1
md"#### Exercise 7"

# ╔═╡ c67079e3-91e2-4507-bb0e-3a45761c733a
md"(a)"

# ╔═╡ dc4580d7-bdd0-4321-9979-7044b2f2df33
begin
  KMeans = @load KMeans pkg=Clustering
  EvoTreeClassifier = @load EvoTreeClassifier
  pipe = @pipeline(Standardizer,
                   ContinuousEncoder,
                   KMeans(k=10),
                   EvoTreeClassifier(nrounds=50))
end

# ╔═╡ fa50e0ac-0d0c-43ff-b2a1-fbbde543f620
md"(b)"

# ╔═╡ b9814590-55d4-4fa5-910b-f8a9a217eff2
begin
  mach = machine(pipe, X, y)
  evaluate!(mach, resampling=CV(nfolds=6), measure=cross_entropy)
end

# ╔═╡ e06b315d-1c67-4920-aa36-d2e8546da46a
md"(c)"

# ╔═╡ a26c8113-18ff-480c-88a1-a2bf91434413
begin
  r = range(pipe, :(evo_tree_classifier.max_depth), lower=1, upper=10)
  
  curve = learning_curve(mach,
                         range=r,
                         resampling=CV(nfolds=6),
                         measure=cross_entropy)
  
  plt = plot(curve.parameter_values, curve.measurements)
  xlabel!(plt, "max_depth")
  ylabel!(plt, "CV estimate of cross entropy")
  savefig("exercise_7c.png")
  plt
end

# ╔═╡ ac1277e1-dd28-42f6-a1cb-54e34396a855
md"Here's a second curve using a different random seed for the booster:"

# ╔═╡ e77629c1-10ee-4527-8037-0de5806e1a54
begin
  using Random
  pipe.evo_tree_classifier.rng = MersenneTwister(123)
  curve = learning_curve(mach,
                         range=r,
                         resampling=CV(nfolds=6),
                         measure=cross_entropy)
  plot!(curve.parameter_values, curve.measurements)
  savefig("exercise_7c_2.png")
  plt
end

# ╔═╡ c5a91339-900f-4f82-9961-a09632c33fb0
md"""
One can automate the production of multiple curves with different
seeds in the following way:
"""

# ╔═╡ 2c4ee7b3-e43e-4269-b7cc-46f4ef988c68
begin
  curves = learning_curve(mach,
                          range=r,
                          resampling=CV(nfolds=6),
                          measure=cross_entropy,
                          rng_name=:(evo_tree_classifier.rng),
                          rngs=6) # list of RNGs, or num to auto generate
  plt = plot(curves.parameter_values, curves.measurements)
  savefig("exercise_7c_3.png")
  plt
end

# ╔═╡ e62f9a56-5ddb-4ad3-90fb-7f2721f6fcc7
md"""
If you have multiple threads available in your julia session, you
can add the option `acceleration=CPUThreads()` to speed up this
computation.
"""

# ╔═╡ 5846d352-3641-4754-af65-ccd25ecb9818
md"#### Exercise 8"

# ╔═╡ 9f13b374-8918-4bae-8890-979691212d9b
md"From the question statement:"

# ╔═╡ 391536d7-1981-4518-a6fa-eaf14df5d44b
begin
  y, X = unpack(house, ==(:price), name -> true, rng=123); # from Exercise 3
  
  EvoTreeRegressor = @load EvoTreeRegressor
  tree_booster = EvoTreeRegressor(nrounds = 70)
  model = @pipeline ContinuousEncoder tree_booster
  
  r2 = range(model,
             :(evo_tree_regressor.nbins),
             lower = 2.5,
             upper= 7.5, scale=x->2^round(Int, x))
end

# ╔═╡ c67079e3-91e2-4507-8025-5084804a9f6c
md"(a)"

# ╔═╡ 2bc25e17-89b0-477a-9e90-09023d201062
r1 = range(model, :(evo_tree_regressor.max_depth), lower=1, upper=12)

# ╔═╡ e06b315d-1c67-4920-b7bb-2f33ef765cc0
md"(c)"

# ╔═╡ 546c47b0-a0f2-4d48-9625-01172c4a0081
begin
  tuned_model = TunedModel(model=model,
                           ranges=[r1, r2],
                           resampling=Holdout(),
                           measures=mae,
                           tuning=RandomSearch(rng=123),
                           n=40)
  
  tuned_mach = machine(tuned_model, X, y) |> fit!
  plt = plot(tuned_mach)
  savefig("exercise_8c.png")
  plt
end

# ╔═╡ d7da0d7d-6776-4614-af4f-11c7de9e21dd
md"(d)"

# ╔═╡ db1190f8-def5-403a-8dba-241d9b744683
begin
  best_model = report(tuned_mach).best_model;
  best_mach = machine(best_model, X, y);
  best_err = evaluate!(best_mach, resampling=CV(nfolds=3), measure=mae)
end

# ╔═╡ f47fee00-3ca4-462c-a6e5-2f0b4dca5c59
tuned_err = evaluate!(tuned_mach, resampling=CV(nfolds=3), measure=mae)

# ╔═╡ 135dac9b-0bd9-4e1d-8550-20498aa03ed0
md"""
---

*This notebook was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
"""

# ╔═╡ Cell order:
# ╟─0f97c6bb-9ced-4cbe-9bca-7c69b794f8ce
# ╟─b67c62c2-15ab-4328-b795-033f6f2a0674
# ╟─bc689638-fd19-4c9f-935f-ddf6a6bfbbdd
# ╟─197fd00e-9068-46ea-af2a-25235e544a31
# ╠═f6d4f8c4-e441-45c4-8af5-148d95ea2900
# ╟─45740c4d-b789-45dc-a6bf-47194d7e8e12
# ╟─42b0f1e1-16c9-4238-828a-4cc485149963
# ╠═d09256dd-6c0d-4e28-9e54-3f7b3ca87ecb
# ╟─499cbc31-83ba-4583-ba1f-6363f43ec697
# ╟─db5821e8-956a-4a46-95ea-2955abd45275
# ╟─6b1758e2-96e8-4272-b1b3-dd47e367ba54
# ╠═c7449612-1de3-436a-8d7e-8e449afd1c48
# ╟─945f2179-e861-4ae7-a949-7f3bd292fe31
# ╟─e62654a0-cc6e-4072-8514-99938a2932db
# ╠═7f9003b5-95c7-418b-a0de-1721c1bc2df2
# ╠═c037caf3-700e-4cc8-bca8-7eec7950fd82
# ╟─e9be221c-06a1-41bc-9873-1b1430e635cc
# ╟─9f13b374-8918-4bae-b43e-d0ebe87da176
# ╠═9a417692-fea9-4606-9009-4cfb2012998f
# ╟─e926dc2e-2b3b-4683-abd3-cb77d7a79683
# ╠═cb767b7d-bc7f-4759-879e-dc128f3db7b3
# ╟─0ec099e3-6e30-41f6-a367-9aa3c6cf34d0
# ╠═159c8967-ab87-4b2a-bf32-94657e65284e
# ╟─f93fec06-260f-4693-9afd-65a5b5facac9
# ╠═17809199-b8c5-4b29-b6c8-54f5ed90a964
# ╟─8eaa2b50-c1f7-4f8b-9292-fe822525fc77
# ╠═0f5162b2-daf6-43a1-ae5e-4b83dcbc9675
# ╟─44e87adc-146a-4da9-8a29-32f11452654a
# ╟─9f13b374-8918-4bae-a5f1-151fcbe229a2
# ╠═1f822943-6f7a-4615-81bc-2d5603785a09
# ╟─e3cc20c2-6492-45e0-9d87-1d1fbb0e3997
# ╟─63f952e4-5d3a-48ac-b950-fddef2a1fb10
# ╠═ae1cb765-ba67-496e-951b-64cb2a36c8e3
# ╟─d30bc3a2-c019-406f-b0e6-441461cc8770
# ╠═af31e97b-cfba-4caa-8cb0-de601961bc02
# ╟─08f6a160-3148-40fc-a87d-950c50fb2955
# ╟─9f13b374-8918-4bae-8448-4bcb089096cd
# ╠═6df9f266-f9d9-4506-a012-a84240254fb6
# ╟─9a4ff2d2-c5b6-4488-bbdd-9799f7bb2e60
# ╠═affb9923-2afa-4655-97a7-88e3af4f10ee
# ╟─9995fba2-e61b-4877-b372-8c9866e51852
# ╠═29825df3-1e51-4b8d-8f3d-b82f1e7b6f7a
# ╠═c01a3d86-3f7e-494b-ab04-579e5608ae53
# ╟─1af34706-479e-42a8-86cf-35420d9e6995
# ╟─be4e773b-284c-406f-a29a-8d0445351914
# ╠═f8bc8787-ea6a-4935-be65-28d0fcca50a8
# ╟─7e046dfe-49ce-47e4-9a2e-c468345d87d1
# ╠═2f3f1724-0d11-4159-b5fa-cb79ebf595ef
# ╟─4c06c69e-8a21-43d7-91c5-524da38aa391
# ╠═89b9ee21-1940-4e5e-ad90-b4405b216771
# ╟─83fba101-4cf1-48aa-895b-997a92b731e0
# ╠═0fb6fa7a-f0c0-4926-a525-e9b04a4bd246
# ╟─e56be556-d75a-40c9-80f0-ba7781e173cf
# ╠═28a64fbd-ceeb-44b3-9cbc-9a2c39793333
# ╟─5d00e3fe-feb5-4394-b887-2b16710e5502
# ╠═c25a8ee7-61b3-4b07-9452-5298a8a4a401
# ╟─0e8a02cd-4571-45d8-b017-96f1602f2cad
# ╠═5a351c52-5915-4e2b-8be2-2a4017c45345
# ╟─faef6958-7f5d-42e4-a7ad-80d84f5b0070
# ╠═240987eb-4252-42e3-8378-5aa686f0b407
# ╟─25a6fd3c-60d1-44dd-9f43-49763e8691a1
# ╟─c67079e3-91e2-4507-bb0e-3a45761c733a
# ╠═dc4580d7-bdd0-4321-9979-7044b2f2df33
# ╟─fa50e0ac-0d0c-43ff-b2a1-fbbde543f620
# ╠═b9814590-55d4-4fa5-910b-f8a9a217eff2
# ╟─e06b315d-1c67-4920-aa36-d2e8546da46a
# ╠═a26c8113-18ff-480c-88a1-a2bf91434413
# ╟─ac1277e1-dd28-42f6-a1cb-54e34396a855
# ╠═e77629c1-10ee-4527-8037-0de5806e1a54
# ╟─c5a91339-900f-4f82-9961-a09632c33fb0
# ╠═2c4ee7b3-e43e-4269-b7cc-46f4ef988c68
# ╟─e62f9a56-5ddb-4ad3-90fb-7f2721f6fcc7
# ╟─5846d352-3641-4754-af65-ccd25ecb9818
# ╟─9f13b374-8918-4bae-8890-979691212d9b
# ╠═391536d7-1981-4518-a6fa-eaf14df5d44b
# ╟─c67079e3-91e2-4507-8025-5084804a9f6c
# ╠═2bc25e17-89b0-477a-9e90-09023d201062
# ╟─e06b315d-1c67-4920-b7bb-2f33ef765cc0
# ╠═546c47b0-a0f2-4d48-9625-01172c4a0081
# ╟─d7da0d7d-6776-4614-af4f-11c7de9e21dd
# ╠═db1190f8-def5-403a-8dba-241d9b744683
# ╠═f47fee00-3ca4-462c-a6e5-2f0b4dca5c59
# ╟─135dac9b-0bd9-4e1d-8550-20498aa03ed0
