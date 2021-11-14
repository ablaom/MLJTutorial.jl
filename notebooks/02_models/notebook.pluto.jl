### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# ╔═╡ d09256dd-6c0d-4e28-9e54-3f7b3ca87ecb
begin
  using Pkg
  Pkg.activate("env")
  Pkg.instantiate()
end

# ╔═╡ ae4f097b-4c18-4025-8514-99938a2932db
begin
  using MLJ
  OpenML.describe_dataset(61)
end

# ╔═╡ 88c11aa3-2c4c-4200-a0de-1721c1bc2df2
begin
  iris = OpenML.load(61); # a column dictionary table
  
  using DataFrames
  iris = DataFrames.DataFrame(iris);
  first(iris, 4)
end

# ╔═╡ 6df9f266-f9d9-4506-b8ad-0340f15a03ba
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

# ╔═╡ a05a4f6d-c831-4fb1-9bca-7c69b794f8ce
md"# Machine Learning in Julia, JuliaCon2020"

# ╔═╡ bea73bf9-96c0-42fb-b795-033f6f2a0674
md"""
A workshop introducing the machine learning toolbox
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/stable/).
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

# ╔═╡ 499cbc31-83ba-4583-ba1f-6363f43ec697
md"## General resources"

# ╔═╡ 2bac0883-5e62-4e1a-95ea-2955abd45275
md"""
- [List of methods introduced in this tutorial](methods.md)
- [MLJ Cheatsheet](https://alan-turing-institute.github.io/MLJ.jl/dev/mlj_cheatsheet/)
- [Common MLJ Workflows](https://alan-turing-institute.github.io/MLJ.jl/dev/common_mlj_workflows/)
- [MLJ manual](https://alan-turing-institute.github.io/MLJ.jl/dev/)
- [Data Science Tutorials in Julia](https://juliaai.github.io/DataScienceTutorials.jl/)
"""

# ╔═╡ 4afc3a94-c3e3-493e-b1b3-dd47e367ba54
md"## Part 2 - Selecting, Training and Evaluating Models"

# ╔═╡ ab471845-15a5-40f8-8d7e-8e449afd1c48
md"""
> **Goals:**
> 1. Search MLJ's database of model metadata to identify model candidates for a supervised learning task.
> 2. Evaluate the performance of a model on a holdout set using basic `fit!`/`predict` work-flow.
> 3. Inspect the outcomes of training and save these to a file.
> 3. Evaluate performance using other resampling strategies, such as cross-validation, in one line, using `evaluate!`
> 4. Plot a "learning curve", to inspect performance as a function of some model hyper-parameter, such as an iteration parameter
"""

# ╔═╡ 985baf9f-1507-442d-a949-7f3bd292fe31
md"""
The "Hello World!" of machine learning is to classify Fisher's
famous iris data set. This time, we'll grab the data from
[OpenML](https://www.openml.org):
"""

# ╔═╡ 27b51025-5fc0-4453-bca8-7eec7950fd82
md"""
**Main goal.** To build and evaluate models for predicting the
`:class` variable, given the four remaining measurement variables.
"""

# ╔═╡ 3b869e8a-24c4-4b00-9873-1b1430e635cc
md"### Step 1. Inspect and fix scientific types"

# ╔═╡ 59e3c087-617d-457b-b43e-d0ebe87da176
schema(iris)

# ╔═╡ 09c0f67c-60b8-42dd-9009-4cfb2012998f
begin
  coerce!(iris,
          Union{Missing,Continuous}=>Continuous,
          Union{Missing,Multiclass}=>Multiclass)
  schema(iris)
end

# ╔═╡ 992ca8fb-1a20-4664-abd3-cb77d7a79683
md"### Step 2. Split data into input and target parts"

# ╔═╡ bf7f99f5-7096-441f-879e-dc128f3db7b3
md"""
Here's how we split the data into target and input features, which
is needed for MLJ supervised models. We randomize the data at the
same time:
"""

# ╔═╡ b081e89d-3ded-4061-bf32-94657e65284e
md"""
Here's one way to access the documentation (at the REPL, `?unpack`
also works):
"""

# ╔═╡ 325b5554-5521-48da-9afd-65a5b5facac9
@doc unpack

# ╔═╡ 2f1d872a-8471-4e85-b6c8-54f5ed90a964
md"### On searching for a model"

# ╔═╡ ce0ec15e-a419-4517-9292-fe822525fc77
md"Here's how to see *all* models (not immediately useful):"

# ╔═╡ 95d1dc5f-75d9-4297-ae5e-4b83dcbc9675
all_models = models()

# ╔═╡ d31909f8-f5f1-4773-8a29-32f11452654a
md"Each entry contains metadata for a model whose defining code is not yet loaded:"

# ╔═╡ c1f7376d-6f42-497f-a5f1-151fcbe229a2
meta = all_models[3]

# ╔═╡ 2640cb64-bc0e-4bca-81bc-2d5603785a09
targetscitype = meta.target_scitype

# ╔═╡ 5e706d1f-93f3-4d50-b950-fddef2a1fb10
md"So this model won't do. Let's  find all pure julia classifiers:"

# ╔═╡ 7de5633e-8d59-4ad8-951b-64cb2a36c8e3
begin
  filter_julia_classifiers(meta) =
      AbstractVector{Finite} <: meta.target_scitype &&
      meta.is_pure_julia
  
  models(filter_julia_classifiers)
end

# ╔═╡ bd5206ab-a4f8-4bb2-b0e6-441461cc8770
md"Find all models with \"Classifier\" in `name` (or `docstring`):"

# ╔═╡ 66bd66a0-05d9-48eb-8cb0-de601961bc02
models("Classifier")

# ╔═╡ a42a2993-098f-4943-a87d-950c50fb2955
md"Find all (supervised) models that match my data!"

# ╔═╡ 34feba2c-e700-423a-a012-a84240254fb6
md"### Step 3. Select and instantiate a model"

# ╔═╡ ad6637a0-b5f3-430c-bbdd-9799f7bb2e60
md"To load the code defining a new model type we use the `@load` macro:"

# ╔═╡ 622d0b24-ca8f-4bd4-97a7-88e3af4f10ee
NeuralNetworkClassifier = @load NeuralNetworkClassifier

# ╔═╡ b98100f5-976f-4d6a-b372-8c9866e51852
md"""
Other ways to load model code are described
[here](https://alan-turing-institute.github.io/MLJ.jl/dev/loading_model_code/#Loading-Model-Code).
"""

# ╔═╡ 984397e2-ab08-4ef5-8f3d-b82f1e7b6f7a
md"""
We'll instantiate this type with default values for the
hyperparameters:
"""

# ╔═╡ 335d13dc-807d-491f-ab04-579e5608ae53
model = NeuralNetworkClassifier()

# ╔═╡ 96cba2ba-f64e-44a7-86cf-35420d9e6995
info(model)

# ╔═╡ e1e8e176-74da-437a-a29a-8d0445351914
md"""
In MLJ a *model* is just a struct containing hyper-parameters, and
that's all. A model does not store *learned* parameters. Models are
mutable:
"""

# ╔═╡ 83b4291e-3842-4819-be65-28d0fcca50a8
model.epochs = 12

# ╔═╡ 9a1e2ea6-4e31-49d5-9a2e-c468345d87d1
md"""
And all models have a key-word constructor that works once `@load`
has been performed:
"""

# ╔═╡ af0c0a55-d994-452d-b5fa-cb79ebf595ef
NeuralNetworkClassifier(epochs=12) == model

# ╔═╡ 39a680a3-2e87-4b48-91c5-524da38aa391
md"### On fitting, predicting, and inspecting models"

# ╔═╡ 595983b4-3c4e-44af-ad90-b4405b216771
md"""
In MLJ a model and training/validation data are typically bound
together in a machine:
"""

# ╔═╡ c8700a21-e95b-4f0f-a525-e9b04a4bd246
md"""
A machine stores *learned* parameters, among other things. We'll
train this machine on 70% of the data and evaluate on a 30% holdout
set. Let's start by dividing all row indices into `train` and `test`
subsets:
"""

# ╔═╡ a232a792-8692-45cc-9cbc-9a2c39793333
md"Now we can `fit!`..."

# ╔═╡ 3856cc2d-ea15-4077-9452-5298a8a4a401
md"... and `predict`:"

# ╔═╡ 7d8e68c8-430c-46dd-8be2-2a4017c45345
md"We'll have more to say on the form of this prediction shortly."

# ╔═╡ fb9a3a2d-6b1a-4d64-a7ad-80d84f5b0070
md"After training, one can inspect the learned parameters:"

# ╔═╡ d5308047-a4ee-41c8-9f43-49763e8691a1
md"""
Everything else the user might be interested in is accessed from the
training *report*:
"""

# ╔═╡ 2ecd9ccb-45c4-4de4-9979-7044b2f2df33
md"You save a machine like this:"

# ╔═╡ 43c25729-4bda-4c21-910b-f8a9a217eff2
md"And retrieve it like this:"

# ╔═╡ 2a1d3e22-d17c-452d-88a1-a2bf91434413
md"If you want to fit a retrieved model, you will need to bind some data to it:"

# ╔═╡ 9dbeeed8-d882-4e04-8037-0de5806e1a54
md"""
Machines remember the last set of hyper-parameters used during fit,
which, in the case of iterative models, allows for a warm restart of
computations in the case that only the iteration parameter is
increased:
"""

# ╔═╡ 8ce96a51-938d-43d1-b7cc-46f4ef988c68
md"""
For this particular model we can also increase `:learning_rate`
without triggering a cold restart:
"""

# ╔═╡ 35af5a52-69f9-4b74-af65-ccd25ecb9818
md"""
However, change any other parameter and training will restart from
scratch:
"""

# ╔═╡ 87065854-efb2-44b7-a6fa-eaf14df5d44b
md"""
Iterative models that implement warm-restart for training can be
controlled externally (eg, using an out-of-sample stopping
criterion). See
[here](https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/)
for details.
"""

# ╔═╡ 925a7014-51dc-4a55-8025-5084804a9f6c
md"""
Let's train silently for a total of 50 epochs, and look at a
prediction:
"""

# ╔═╡ 2faf541c-1a2c-4274-b7bb-2f33ef765cc0
md"What's going on here?"

# ╔═╡ ca444411-88c0-4c2c-9625-01172c4a0081
info(model).prediction_type

# ╔═╡ 49bcdff9-0e08-45e3-af4f-11c7de9e21dd
md"""
**Important**:
- In MLJ, a model that can predict probabilities (and not just point values) will do so by default.
- For most probabilistic predictors, the predicted object is a `Distributions.Distribution` object, supporting the `Distributions.jl` [API](https://juliastats.org/Distributions.jl/latest/extends/#Create-a-Distribution-1) for such objects. In particular, the methods `rand`,  `pdf`, `logpdf`, `mode`, `median` and `mean` will apply, where appropriate.
"""

# ╔═╡ 9653dbb8-a168-4a07-8dba-241d9b744683
md"""
So, to obtain the probability of "Iris-virginica" in the first test
prediction, we do
"""

# ╔═╡ b760d242-43de-4248-8550-20498aa03ed0
md"To get the most likely observation, we do"

# ╔═╡ 523bcb12-f95d-4b90-bce6-0a1379cc1259
md"These can be broadcast over multiple predictions in the usual way:"

# ╔═╡ 275f2017-5520-4072-8d9e-644a1b3cc6b6
md"""
Or, alternatively, you can use the `predict_mode` operation instead
of `predict`:
"""

# ╔═╡ 1d455c06-389b-402a-8535-1a088a6a3228
md"For a more conventional matrix of probabilities you can do this:"

# ╔═╡ bf534937-b3c5-4e01-bcca-51a27994a151
md"""
However, in a typical MLJ work-flow, this is not as useful as you
might imagine. In particular, all probabilistic performance measures
in MLJ expect distribution objects in their first slot:
"""

# ╔═╡ 8b98a48c-d2c6-4ecf-b45d-88d068bb0fa2
md"To apply a deterministic measure, we first need to obtain point-estimates:"

# ╔═╡ 09a60a33-8b8d-4aed-abf5-96f457eb2bdf
md"""
We note in passing that there is also a search tool for measures
analogous to `models`:
"""

# ╔═╡ e08d911c-f1a4-40b5-8a60-2e0214c059f5
measures()

# ╔═╡ 8323c2ea-16bc-4913-a38a-beb5c7157b57
md"### Step 4. Evaluate the model performance"

# ╔═╡ b2227a29-b214-4683-81f5-507803ea9ed6
md"""
Naturally, MLJ provides boilerplate code for carrying out a model
evaluation with a lot less fuss. Let's repeat the performance
evaluation above and add an extra measure, `brier_score`:
"""

# ╔═╡ f829c00b-f082-4ec4-b98c-3df1f31879bf
md"Or applying cross-validation instead:"

# ╔═╡ 31359fe0-8d95-4740-b122-214de244406c
md"""
Or, Monte Carlo cross-validation (cross-validation repeated
randomized folds)
"""

# ╔═╡ 3c3b7c0f-ef92-4eeb-a8b6-f7ff516dedc4
md"""
One can access the following properties of the output `e` of an
evaluation: `measure`, `measurement`, `per_fold` (measurement for
each fold) and `per_observation` (measurement per observation, if
reported).
"""

# ╔═╡ a6725314-8f98-4869-81e1-74ef03c2e79e
md"""
We finally note that you can restrict the rows of observations from
which train and test folds are drawn, by specifying `rows=...`. For
example, imagining the last 30% of target observations are `missing`
you might have a work-flow like this:
"""

# ╔═╡ 51f56327-7309-4aa2-b979-3458f2f26667
md"### On learning curves"

# ╔═╡ f2becd33-0b66-4b13-97e3-6677afc6ca9f
md"""
Since our model is an iterative one, we might want to inspect the
out-of-sample performance as a function of the iteration
parameter. For this we can use the `learning_curve` function (which,
incidentally can be applied to any model hyper-parameter). This
starts by defining a one-dimensional range object for the parameter
(more on this when we discuss tuning in Part 4):
"""

# ╔═╡ 1d65816d-d818-434d-b10e-562ce21caa04
r = range(model, :epochs, lower=1, upper=50, scale=:log)

# ╔═╡ e80b6400-2c9e-4198-a8a4-a53f5149481e
md"We will return to learning curves when we look at tuning in Part 4."

# ╔═╡ fc268355-011e-44d5-8704-dbab8e09b4f1
md"### Resources for Part 2"

# ╔═╡ 25164534-5e28-44f4-a02f-27b6c05e4d02
md"""
- From the MLJ manual:
    - [Getting Started](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/)
    - [Model Search](https://alan-turing-institute.github.io/MLJ.jl/dev/model_search/)
    - [Evaluating Performance](https://alan-turing-institute.github.io/MLJ.jl/dev/evaluating_model_performance/) (using `evaluate!`)
    - [Learning Curves](https://alan-turing-institute.github.io/MLJ.jl/dev/learning_curves/)
    - [Performance Measures](https://alan-turing-institute.github.io/MLJ.jl/dev/performance_measures/) (loss functions, scores, etc)
- From Data Science Tutorials:
    - [Choosing and evaluating a model](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/choosing-a-model/)
    - [Fit, predict, transform](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/fit-and-predict/)
"""

# ╔═╡ 1ffaccf9-a0ec-451c-bea4-3d467d48781c
md"### Exercises for Part 2"

# ╔═╡ 14a71476-a7ca-4324-97c5-2979af8a507d
md"#### Exercise 4"

# ╔═╡ 47c32a4a-5596-4970-b639-63c76c72c513
md"""
(a) Identify all supervised MLJ models that can be applied (without
type coercion or one-hot encoding) to a supervised learning problem
with input features `X4` and target `y4` defined below:
"""

# ╔═╡ 9a75c053-59ba-4c53-8f5a-e1ee9eb5c15c
begin
  import Distributions
  poisson = Distributions.Poisson
  
  age = 18 .+ 60*rand(10);
  salary = coerce(rand(["small", "big", "huge"], 10), OrderedFactor);
  levels!(salary, ["small", "big", "huge"]);
  small = salary[1]
end

# ╔═╡ f3de7b01-d9e5-4b5c-adc5-e671db8bca5d
begin
  X4 = DataFrames.DataFrame(age=age, salary=salary)
  
  n_devices(salary) = salary > small ? rand(poisson(1.3)) : rand(poisson(2.9))
  y4 = [n_devices(row.salary) for row in eachrow(X4)]
end

# ╔═╡ 158df0a8-625f-4389-86f1-0ea38de21abb
md"""
(b) What models can be applied if you coerce the salary to a
`Continuous` scitype?
"""

# ╔═╡ 0cec3967-8a9e-4da5-a55c-0cb44ab816d7
md"#### Exercise 5 (unpack)"

# ╔═╡ 1dcadbe8-6897-48bb-be86-92ecfd0d2343
md"After evaluating the following ..."

# ╔═╡ f303e667-6485-4949-9cf1-facf39e3f302
begin
  data = (a = [1, 2, 3, 4],
          b = rand(4),
          c = rand(4),
          d = coerce(["male", "female", "female", "male"], OrderedFactor));
  pretty(data)
end

# ╔═╡ ba53ad4c-e092-40fd-9d87-1d1fbb0e3997
scitype(y) <: targetscitype

# ╔═╡ db2d9ba2-1fc4-4a7f-8448-4bcb089096cd
models(matching(X, y))

# ╔═╡ ef2282bc-db28-4d54-a1cb-54e34396a855
begin
  mach3 = machine("neural_net.jlso", X, y)
  fit!(mach3)
end

# ╔═╡ 62be4fb6-3b7e-4b80-b887-2b16710e5502
fit!(mach, rows=train, verbosity=2)

# ╔═╡ deef14fe-9822-4fa8-8378-5aa686f0b407
fitted_params(mach)

# ╔═╡ d8f73aeb-2b82-4afd-bb0e-3a45761c733a
report(mach)

# ╔═╡ 43887c78-378f-4bb8-b2a1-fbbde543f620
MLJ.save("neural_net.jlso", mach)

# ╔═╡ 7547798a-24c9-4c5f-9961-a09632c33fb0
begin
  model.epochs = model.epochs + 4
  fit!(mach, rows=train, verbosity=2)
end

# ╔═╡ 04255053-ed5f-48d6-90fb-7f2721f6fcc7
begin
  model.epochs = model.epochs + 4
  model.optimiser.eta = 10*model.optimiser.eta
  fit!(mach, rows=train, verbosity=2)
end

# ╔═╡ 55531d1d-31e9-4e61-8890-979691212d9b
begin
  model.lambda = 0.001
  fit!(mach, rows=train, verbosity=2)
end

# ╔═╡ 1613d417-80ac-40bd-a6e5-2f0b4dca5c59
pdf(yhat[1], "Iris-virginica")

# ╔═╡ f40a1541-ee37-487e-9e7b-5b943cf6b560
mode(yhat[1])

# ╔═╡ 691b9fb3-72c7-43bd-9608-b5032c116833
broadcast(pdf, yhat[1:4], "Iris-versicolor")

# ╔═╡ 880f9d8d-3095-45b5-b473-4aa968e6937a
mode.(yhat[1:4])

# ╔═╡ 76bc1d1e-3de7-461c-a39f-7893c73eef39
begin
  L = levels(y)
  pdf(yhat, L)[1:4, :]
end

# ╔═╡ 93533a8e-e507-4950-ac0a-23ded81445da
predict_mode(mach, X[test,:])[1:4] # or predict_mode(mach, rows=test)[1:4]

# ╔═╡ 8d59229c-1a70-40b2-9b33-09b4b6661170
cross_entropy(yhat, y[test]) |> mean

# ╔═╡ 0fb6fa7a-f0c0-4926-92c8-6db3a590d963
misclassification_rate(mode.(yhat), y[test])

# ╔═╡ c63ed6db-6d2a-4ee7-9b21-6d4eb642d87e
evaluate!(mach, resampling=Holdout(fraction_train=0.7),
          measures=[cross_entropy, brier_score])

# ╔═╡ 727ecfae-2e87-4e05-92b7-4ceba56e97ad
evaluate!(mach, resampling=CV(nfolds=6),
          measures=[cross_entropy, brier_score])

# ╔═╡ 1383b285-59f2-4652-8a4b-c83694978e38
e = evaluate!(mach, resampling=CV(nfolds=6, rng=123),
              repeats=3,
              measures=[cross_entropy, brier_score])

# ╔═╡ 97400000-e067-4c66-8f78-ab549ef1544e
begin
  curve = learning_curve(mach,
                         range=r,
                         resampling=Holdout(fraction_train=0.7), # (default)
                         measure=cross_entropy)
  
  using Plots
  gr(size=(490,300))
  plt=plot(curve.parameter_values, curve.measurements)
  xlabel!(plt, "epochs")
  ylabel!(plt, "cross entropy on holdout set")
  savefig("learning_curve.png")
  plt
end

# ╔═╡ b5b23e79-35ce-4857-9727-ed822e4fd85d
md"...attempt to guess the evaluations of the following:"

# ╔═╡ 8346cbaa-16bb-4f7a-b2f2-e08965e5be66
y

# ╔═╡ 0044efc0-7431-465d-897c-f79712f9ec7c
pretty(X)

# ╔═╡ 76dc6ac8-03fe-4c6e-a543-f9324a87efad
w

# ╔═╡ 0860f65e-8e80-4394-8657-9fba0caf3cb7
md"#### Exercise 6 (first steps in modeling Horse Colic)"

# ╔═╡ 17b6bf6e-31d0-4981-a217-e58f442fc85c
md"""
Here is the Horse Colic data introduced in Part 1, together with the
type coercions we performed there:
"""

# ╔═╡ 0afa73e6-f371-4b04-946d-b124a8db5f7c
md"""
(a) Suppose we want to use predict the `:outcome` variable, based on
the remaining variables that are `Continuous` (one-hot encoding
categorical variables is discussed later in Part 3) *while ignoring
the others*.  Extract from the `horse` data set (defined in Part 1)
appropriate input features `X` and target variable `y`. (Do not,
however, randomize the observations.)
"""

# ╔═╡ fed7482b-707f-46f0-b582-dfcbeb05bcc5
md"""
(b) Create a 70:30 `train`/`test` split of the data and train a
`LogisticClassifier` model, from the `MLJLinearModels` package, on
the `train` rows. Use `lambda=100` and default values for the
other hyper-parameters. (Although one would normally standardize
(whiten) the continuous features for this model, do not do so here.)
After training:
"""

# ╔═╡ 6f40b49a-ec18-4a3a-9143-546b2286a5fe
md"""
- (i) Recalling that a logistic classifier (aka logistic regressor) is
  a linear-based model learning a *vector* of coefficients for each
  feature (one coefficient for each target class), use the
  `fitted_params` method to find this vector of coefficients in the
  case of the `:pulse` feature. (You can convert a vector of pairs `v =
  [x1 => y1, x2 => y2, ...]` into a dictionary with `Dict(v)`.)
"""

# ╔═╡ ad211315-d320-434c-a7d8-c3b94fb18495
md"""
- (ii) Evaluate the `cross_entropy` performance on the `test`
  observations.
"""

# ╔═╡ cd4374d7-2c80-4007-8396-93fa872d2512
md"""
- &star;(iii) In how many `test` observations does the predicted
  probability of the observed class exceed 50%?
"""

# ╔═╡ 50f4e323-7318-4f68-a4ac-2ddb495858ce
md"""
- (iv) Find the `misclassification_rate` in the `test`
  set. (*Hint.* As this measure is deterministic, you will either
  need to broadcast `mode` or use `predict_mode` instead of
  `predict`.)
"""

# ╔═╡ 84ccf4c1-e3c5-4d25-806e-1bc580dc349d
md"""
(c) Instead use a `RandomForestClassifier` model from the
    `DecisionTree` package and:

- (i) Generate an appropriate learning curve to convince yourself
  that out-of-sample estimates of the `cross_entropy` loss do not
  substantially improve for `n_trees > 50`. Use default values for
  all other hyper-parameters, and feel free to use all available
  data to generate the curve.
"""

# ╔═╡ 6fda3d06-7f51-4966-9703-288eae064e2a
md"""
- (ii) Fix `n_trees=90` and use `evaluate!` to obtain a 9-fold
  cross-validation estimate of the `cross_entropy`, restricting
  sub-sampling to the `train` observations.
"""

# ╔═╡ 7c1dceac-4e37-4434-b2c3-413ae5867f7d
md"""
- (iii) Now use *all* available data but set
  `resampling=Holdout(fraction_train=0.7)` to obtain a score you can
  compare with the `KNNClassifier` in part (b)(iii). Which model is
  better?
"""

# ╔═╡ c2cda2eb-6110-4b1f-93d8-53c527b0a48c
md"<a id='part-3-transformers-and-pipelines'></a>"

# ╔═╡ 135dac9b-0bd9-4e1d-af98-8dffdf3118fc
md"""
---

*This notebook was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
"""

# ╔═╡ de239414-32a8-4691-aa36-d2e8546da46a
begin
  mach2 = machine("neural_net.jlso")
  yhat = predict(mach2, X);
  yhat[1:3]
end

# ╔═╡ df4bd436-9b0e-484b-9e90-09023d201062
begin
  model.epochs = 50
  fit!(mach, rows=train)
  yhat = predict(mach, X[test,:]); # or predict(mach, rows=test)
  yhat[1]
end

# ╔═╡ 174b9a92-2cf5-4630-a04e-0b76409c14a7
begin
  train, test = partition(eachindex(y), 0.7)
  mach = machine(model, X, y)
  evaluate!(mach, resampling=CV(nfolds=6),
            measures=[cross_entropy, brier_score],
            rows=train)     # cv estimate, resampling from `train`
  fit!(mach, rows=train)    # re-train using all of `train` observations
  predict(mach, rows=test); # and predict missing targets
end

# ╔═╡ 933cc42f-6be5-4b9b-a367-9aa3c6cf34d0
begin
  y, X = unpack(iris, ==(:class), name->true; rng=123);
  scitype(y)
end

# ╔═╡ a1c60d89-5d61-4a26-b61c-748aec38e674
begin
  using Tables
  y, X, w = unpack(data,
                   ==(:a),
                   name -> elscitype(Tables.getcolumn(data, name)) == Continuous,
                   name -> true);
end

# ╔═╡ 5b8d30eb-bd5c-4b09-80f0-ba7781e173cf
train, test = partition(eachindex(y), 0.7)

# ╔═╡ 72510848-3148-408a-895b-997a92b731e0
mach = machine(model, X, y)

# ╔═╡ 5a86fd44-8e9b-4694-b017-96f1602f2cad
begin
  yhat = predict(mach, rows=test);  # or `predict(mach, Xnew)`
  yhat[1:3]
end

# ╔═╡ Cell order:
# ╟─a05a4f6d-c831-4fb1-9bca-7c69b794f8ce
# ╟─bea73bf9-96c0-42fb-b795-033f6f2a0674
# ╟─bc689638-fd19-4c9f-935f-ddf6a6bfbbdd
# ╟─197fd00e-9068-46ea-af2a-25235e544a31
# ╠═f6d4f8c4-e441-45c4-8af5-148d95ea2900
# ╟─45740c4d-b789-45dc-a6bf-47194d7e8e12
# ╟─42b0f1e1-16c9-4238-828a-4cc485149963
# ╠═d09256dd-6c0d-4e28-9e54-3f7b3ca87ecb
# ╟─499cbc31-83ba-4583-ba1f-6363f43ec697
# ╟─2bac0883-5e62-4e1a-95ea-2955abd45275
# ╟─4afc3a94-c3e3-493e-b1b3-dd47e367ba54
# ╟─ab471845-15a5-40f8-8d7e-8e449afd1c48
# ╟─985baf9f-1507-442d-a949-7f3bd292fe31
# ╠═ae4f097b-4c18-4025-8514-99938a2932db
# ╠═88c11aa3-2c4c-4200-a0de-1721c1bc2df2
# ╟─27b51025-5fc0-4453-bca8-7eec7950fd82
# ╟─3b869e8a-24c4-4b00-9873-1b1430e635cc
# ╠═59e3c087-617d-457b-b43e-d0ebe87da176
# ╠═09c0f67c-60b8-42dd-9009-4cfb2012998f
# ╟─992ca8fb-1a20-4664-abd3-cb77d7a79683
# ╟─bf7f99f5-7096-441f-879e-dc128f3db7b3
# ╠═933cc42f-6be5-4b9b-a367-9aa3c6cf34d0
# ╟─b081e89d-3ded-4061-bf32-94657e65284e
# ╠═325b5554-5521-48da-9afd-65a5b5facac9
# ╟─2f1d872a-8471-4e85-b6c8-54f5ed90a964
# ╟─ce0ec15e-a419-4517-9292-fe822525fc77
# ╠═95d1dc5f-75d9-4297-ae5e-4b83dcbc9675
# ╟─d31909f8-f5f1-4773-8a29-32f11452654a
# ╠═c1f7376d-6f42-497f-a5f1-151fcbe229a2
# ╠═2640cb64-bc0e-4bca-81bc-2d5603785a09
# ╠═ba53ad4c-e092-40fd-9d87-1d1fbb0e3997
# ╟─5e706d1f-93f3-4d50-b950-fddef2a1fb10
# ╠═7de5633e-8d59-4ad8-951b-64cb2a36c8e3
# ╟─bd5206ab-a4f8-4bb2-b0e6-441461cc8770
# ╠═66bd66a0-05d9-48eb-8cb0-de601961bc02
# ╟─a42a2993-098f-4943-a87d-950c50fb2955
# ╠═db2d9ba2-1fc4-4a7f-8448-4bcb089096cd
# ╟─34feba2c-e700-423a-a012-a84240254fb6
# ╟─ad6637a0-b5f3-430c-bbdd-9799f7bb2e60
# ╠═622d0b24-ca8f-4bd4-97a7-88e3af4f10ee
# ╟─b98100f5-976f-4d6a-b372-8c9866e51852
# ╟─984397e2-ab08-4ef5-8f3d-b82f1e7b6f7a
# ╠═335d13dc-807d-491f-ab04-579e5608ae53
# ╠═96cba2ba-f64e-44a7-86cf-35420d9e6995
# ╟─e1e8e176-74da-437a-a29a-8d0445351914
# ╠═83b4291e-3842-4819-be65-28d0fcca50a8
# ╟─9a1e2ea6-4e31-49d5-9a2e-c468345d87d1
# ╠═af0c0a55-d994-452d-b5fa-cb79ebf595ef
# ╟─39a680a3-2e87-4b48-91c5-524da38aa391
# ╟─595983b4-3c4e-44af-ad90-b4405b216771
# ╠═72510848-3148-408a-895b-997a92b731e0
# ╟─c8700a21-e95b-4f0f-a525-e9b04a4bd246
# ╠═5b8d30eb-bd5c-4b09-80f0-ba7781e173cf
# ╟─a232a792-8692-45cc-9cbc-9a2c39793333
# ╠═62be4fb6-3b7e-4b80-b887-2b16710e5502
# ╟─3856cc2d-ea15-4077-9452-5298a8a4a401
# ╠═5a86fd44-8e9b-4694-b017-96f1602f2cad
# ╟─7d8e68c8-430c-46dd-8be2-2a4017c45345
# ╟─fb9a3a2d-6b1a-4d64-a7ad-80d84f5b0070
# ╠═deef14fe-9822-4fa8-8378-5aa686f0b407
# ╟─d5308047-a4ee-41c8-9f43-49763e8691a1
# ╠═d8f73aeb-2b82-4afd-bb0e-3a45761c733a
# ╟─2ecd9ccb-45c4-4de4-9979-7044b2f2df33
# ╠═43887c78-378f-4bb8-b2a1-fbbde543f620
# ╟─43c25729-4bda-4c21-910b-f8a9a217eff2
# ╠═de239414-32a8-4691-aa36-d2e8546da46a
# ╟─2a1d3e22-d17c-452d-88a1-a2bf91434413
# ╠═ef2282bc-db28-4d54-a1cb-54e34396a855
# ╟─9dbeeed8-d882-4e04-8037-0de5806e1a54
# ╠═7547798a-24c9-4c5f-9961-a09632c33fb0
# ╟─8ce96a51-938d-43d1-b7cc-46f4ef988c68
# ╠═04255053-ed5f-48d6-90fb-7f2721f6fcc7
# ╟─35af5a52-69f9-4b74-af65-ccd25ecb9818
# ╠═55531d1d-31e9-4e61-8890-979691212d9b
# ╟─87065854-efb2-44b7-a6fa-eaf14df5d44b
# ╟─925a7014-51dc-4a55-8025-5084804a9f6c
# ╠═df4bd436-9b0e-484b-9e90-09023d201062
# ╟─2faf541c-1a2c-4274-b7bb-2f33ef765cc0
# ╠═ca444411-88c0-4c2c-9625-01172c4a0081
# ╟─49bcdff9-0e08-45e3-af4f-11c7de9e21dd
# ╟─9653dbb8-a168-4a07-8dba-241d9b744683
# ╠═1613d417-80ac-40bd-a6e5-2f0b4dca5c59
# ╟─b760d242-43de-4248-8550-20498aa03ed0
# ╠═f40a1541-ee37-487e-9e7b-5b943cf6b560
# ╟─523bcb12-f95d-4b90-bce6-0a1379cc1259
# ╠═691b9fb3-72c7-43bd-9608-b5032c116833
# ╠═880f9d8d-3095-45b5-b473-4aa968e6937a
# ╟─275f2017-5520-4072-8d9e-644a1b3cc6b6
# ╠═93533a8e-e507-4950-ac0a-23ded81445da
# ╟─1d455c06-389b-402a-8535-1a088a6a3228
# ╠═76bc1d1e-3de7-461c-a39f-7893c73eef39
# ╟─bf534937-b3c5-4e01-bcca-51a27994a151
# ╠═8d59229c-1a70-40b2-9b33-09b4b6661170
# ╟─8b98a48c-d2c6-4ecf-b45d-88d068bb0fa2
# ╠═0fb6fa7a-f0c0-4926-92c8-6db3a590d963
# ╟─09a60a33-8b8d-4aed-abf5-96f457eb2bdf
# ╠═e08d911c-f1a4-40b5-8a60-2e0214c059f5
# ╟─8323c2ea-16bc-4913-a38a-beb5c7157b57
# ╟─b2227a29-b214-4683-81f5-507803ea9ed6
# ╠═c63ed6db-6d2a-4ee7-9b21-6d4eb642d87e
# ╟─f829c00b-f082-4ec4-b98c-3df1f31879bf
# ╠═727ecfae-2e87-4e05-92b7-4ceba56e97ad
# ╟─31359fe0-8d95-4740-b122-214de244406c
# ╠═1383b285-59f2-4652-8a4b-c83694978e38
# ╟─3c3b7c0f-ef92-4eeb-a8b6-f7ff516dedc4
# ╟─a6725314-8f98-4869-81e1-74ef03c2e79e
# ╠═174b9a92-2cf5-4630-a04e-0b76409c14a7
# ╟─51f56327-7309-4aa2-b979-3458f2f26667
# ╟─f2becd33-0b66-4b13-97e3-6677afc6ca9f
# ╠═1d65816d-d818-434d-b10e-562ce21caa04
# ╠═97400000-e067-4c66-8f78-ab549ef1544e
# ╟─e80b6400-2c9e-4198-a8a4-a53f5149481e
# ╟─fc268355-011e-44d5-8704-dbab8e09b4f1
# ╟─25164534-5e28-44f4-a02f-27b6c05e4d02
# ╟─1ffaccf9-a0ec-451c-bea4-3d467d48781c
# ╟─14a71476-a7ca-4324-97c5-2979af8a507d
# ╟─47c32a4a-5596-4970-b639-63c76c72c513
# ╠═9a75c053-59ba-4c53-8f5a-e1ee9eb5c15c
# ╠═f3de7b01-d9e5-4b5c-adc5-e671db8bca5d
# ╟─158df0a8-625f-4389-86f1-0ea38de21abb
# ╟─0cec3967-8a9e-4da5-a55c-0cb44ab816d7
# ╟─1dcadbe8-6897-48bb-be86-92ecfd0d2343
# ╠═f303e667-6485-4949-9cf1-facf39e3f302
# ╠═a1c60d89-5d61-4a26-b61c-748aec38e674
# ╟─b5b23e79-35ce-4857-9727-ed822e4fd85d
# ╠═8346cbaa-16bb-4f7a-b2f2-e08965e5be66
# ╠═0044efc0-7431-465d-897c-f79712f9ec7c
# ╠═76dc6ac8-03fe-4c6e-a543-f9324a87efad
# ╟─0860f65e-8e80-4394-8657-9fba0caf3cb7
# ╟─17b6bf6e-31d0-4981-a217-e58f442fc85c
# ╠═6df9f266-f9d9-4506-b8ad-0340f15a03ba
# ╟─0afa73e6-f371-4b04-946d-b124a8db5f7c
# ╟─fed7482b-707f-46f0-b582-dfcbeb05bcc5
# ╟─6f40b49a-ec18-4a3a-9143-546b2286a5fe
# ╟─ad211315-d320-434c-a7d8-c3b94fb18495
# ╟─cd4374d7-2c80-4007-8396-93fa872d2512
# ╟─50f4e323-7318-4f68-a4ac-2ddb495858ce
# ╟─84ccf4c1-e3c5-4d25-806e-1bc580dc349d
# ╟─6fda3d06-7f51-4966-9703-288eae064e2a
# ╟─7c1dceac-4e37-4434-b2c3-413ae5867f7d
# ╟─c2cda2eb-6110-4b1f-93d8-53c527b0a48c
# ╟─135dac9b-0bd9-4e1d-af98-8dffdf3118fc
