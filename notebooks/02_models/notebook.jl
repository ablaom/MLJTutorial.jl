# # Machine Learning in Julia (continued)

# An introduction to the
# [MLJ](https://alan-turing-institute.github.io/MLJ.jl/stable/)
# toolbox.


# ### Set-up

# Inspect Julia version:

VERSION

# The following instantiates a package environment.

# The package environment has been created using **Julia 1.6** and may not
# instantiate properly for other Julia versions.

using Pkg
Pkg.activate("env")
Pkg.instantiate()


# ## General resources

# - [MLJ Cheatsheet](https://alan-turing-institute.github.io/MLJ.jl/dev/mlj_cheatsheet/)
# - [Common MLJ Workflows](https://alan-turing-institute.github.io/MLJ.jl/dev/common_mlj_workflows/)
# - [MLJ manual](https://alan-turing-institute.github.io/MLJ.jl/dev/)
# - [Data Science Tutorials in Julia](https://juliaai.github.io/DataScienceTutorials.jl/)


# ## Part 2 - Selecting, Training and Evaluating Models

# > **Goals:**
# > 1. Search MLJ's database of model metadata to identify model candidates for a supervised learning task.
# > 2. Evaluate the performance of a model on a holdout set using basic `fit!`/`predict` work-flow.
# > 3. Inspect the outcomes of training and save these to a file.
# > 3. Evaluate performance using other resampling strategies, such as cross-validation, in one line, using `evaluate!`
# > 4. Plot a "learning curve", to inspect performance as a function of some model hyper-parameter, such as an iteration parameter

# The "Hello World!" of machine learning is to classify Fisher's
# famous iris data set. This time, we'll grab the data from
# [OpenML](https://www.openml.org):

using MLJ
OpenML.describe_dataset(61)

#-

iris = OpenML.load(61); # a column dictionary table

import DataFrames
iris = DataFrames.DataFrame(iris);
first(iris, 4)

# **Main goal.** To build and evaluate models for predicting the
# `:class` variable, given the four remaining measurement variables.


# ### Step 1. Inspect and fix scientific types

schema(iris)

# These look fine.

# ### Step 2. Split data into input and target parts

# Here's how we split the data into target and input features, which
# is needed for MLJ supervised models. We can randomize the data at
# the same time:

y, X = unpack(iris, ==(:class), rng=123);
scitype(y)

# This puts the `:class` column into a vector `y`, and all remaining
# columns into a table `X`.

# Here's one way to access the documentation (at the REPL, `?unpack`
# also works):

@doc unpack #!md

# <display omitted, as not markdown renderable> #md


# ### On searching for a model

# Here's how to see *all* models (not immediately useful):

all_models = models()

# If you already have an idea about the name of the model, you could
# search by string or regex:

some_models = models("LinearRegressor")

# Each entry contains metadata for a model whose defining code is not
# yet loaded:

meta = some_models[1]

#-

targetscitype = meta.target_scitype

#-

scitype(y) <: targetscitype

# So this model won't do. Let's find all pure julia classifiers:

filter_julia_classifiers(meta) =
    AbstractVector{Finite} <: meta.target_scitype &&
    meta.is_pure_julia

models(filter_julia_classifiers)

# Find all (supervised) models that match my data!

models(matching(X, y))



# ### Step 3. Select and instantiate a model

# To load the code defining a new model type we use the `@load` macro:

NeuralNetworkClassifier = @load NeuralNetworkClassifier

# Other ways to load model code are described
# [here](https://alan-turing-institute.github.io/MLJ.jl/dev/loading_model_code/#Loading-Model-Code).

# We'll instantiate this type with default values for the
# hyperparameters:

model = NeuralNetworkClassifier()

#-

info(model)

# In MLJ a *model* is just a struct containing hyper-parameters, and
# that's all. A model does not store *learned* parameters. Models are
# mutable:

model.epochs = 12

# And all models have a key-word constructor that works once `@load`
# has been performed:

NeuralNetworkClassifier(epochs=12) == model


# ### On fitting, predicting, and inspecting models

# In MLJ a model and training/validation data are typically bound
# together in a machine:

mach = machine(model, X, y)

# A machine stores *learned* parameters, among other things. We'll
# train this machine on 70% of the data and evaluate on a 30% holdout
# set. Let's start by dividing all row indices into `train` and `test`
# subsets:

train, test = partition(1:length(y), 0.7)

# Now we can `fit!`...

fit!(mach, rows=train, verbosity=2)

# ... and `predict`:

yhat = predict(mach, rows=test);  # or `predict(mach, Xnew)`
yhat[1:3]

# We'll have more to say on the form of this prediction shortly.

# After training, one can inspect the learned parameters:

fitted_params(mach)

#-

# Everything else the user might be interested in is accessed from the
# training *report*:

report(mach)

# You save a machine like this:

MLJ.save("neural_net.jlso", mach)

# And retrieve it like this:

mach2 = machine("neural_net.jlso")
yhat = predict(mach2, X);
yhat[1:3]

# If you want to fit a retrieved model, you will need to bind some data to it:

mach3 = machine("neural_net.jlso", X, y)
fit!(mach3)

# Machines remember the last set of hyper-parameters used during fit,
# which, in the case of iterative models, allows for a warm restart of
# computations in the case that only the iteration parameter is
# increased:

model.epochs = model.epochs + 4
fit!(mach, rows=train, verbosity=2)

# For this particular model we can also increase `:learning_rate`
# without triggering a cold restart:

model.epochs = model.epochs + 4
model.optimiser.eta = 10*model.optimiser.eta
fit!(mach, rows=train, verbosity=2)

# However, change any other parameter and training will restart from
# scratch:

model.lambda = 0.001
fit!(mach, rows=train, verbosity=2)

# Iterative models that implement warm-restart for training can be
# controlled externally (eg, using an out-of-sample stopping
# criterion). See
# [here](https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/)
# for details.


# Let's train silently for a total of 50 epochs, and look at a
# prediction:

model.epochs = 50
fit!(mach, rows=train)
yhat = predict(mach, X[test,:]); # or predict(mach, rows=test)
yhat[1]

# What's going on here?

info(model).prediction_type

# **Important**: - In MLJ, a model that can predict probabilities (and
# not just point values) will do so by default.  - For most
# probabilistic predictors, the predicted object is a
# `Distributions.Distribution` object or a
# `CategoricalDistributions.UnivariateFinite` object (the case here)
# which all support the following methods: `rand`, `pdf`, `logpdf`;
# and, where appropriate: `mode`, `median` and `mean`.

# So, to obtain the probability of "Iris-virginica" in the first test
# prediction, we do

pdf(yhat[1], "Iris-virginica")

# To get the most likely observation, we do

mode(yhat[1])

# These can be broadcast over multiple predictions in the usual way:

broadcast(pdf, yhat[1:4], "Iris-versicolor")

#-

mode.(yhat[1:4])

# Or, alternatively, you can use the `predict_mode` operation instead
# of `predict`:

predict_mode(mach, X[test,:])[1:4] # or predict_mode(mach, rows=test)[1:4]

# For a more conventional matrix of probabilities you can do this:

L = levels(y)
pdf(yhat, L)[1:4, :]

# However, in a typical MLJ work-flow, this is not as useful as you
# might imagine. In particular, all probabilistic performance measures
# in MLJ expect distribution objects in their first slot:

cross_entropy(yhat, y[test]) |> mean

# To apply a deterministic measure, we first need to obtain point-estimates:

misclassification_rate(mode.(yhat), y[test])

# We note in passing that there is also a search tool for measures
# analogous to `models`:

measures()


# ### Step 4. Evaluate the model performance

# Naturally, MLJ provides boilerplate code for carrying out a model
# evaluation with a lot less fuss. Let's repeat the performance
# evaluation above and add an extra measure, `brier_score`:

evaluate!(mach, resampling=Holdout(fraction_train=0.7),
          measures=[cross_entropy, misclassification_rate, brier_score])

# Or applying cross-validation instead:

evaluate!(mach, resampling=CV(nfolds=6),
          measures=[cross_entropy, misclassification_rate, brier_score])

# Or, Monte Carlo cross-validation (cross-validation repeated
# randomized folds)

e = evaluate!(mach, resampling=CV(nfolds=6, rng=123),
              repeats=3,
              measures=[cross_entropy, misclassification_rate, brier_score])

# One can access the following properties of the output `e` of an
# evaluation: `measure`, `measurement`, `per_fold` (measurement for
# each fold) and `per_observation` (measurement per observation, if
# reported).

# We finally note that you can restrict the rows of observations from
# which train and test folds are drawn, by specifying `rows=...`. For
# example, imagining the last 30% of target observations are `missing`
# you might have a work-flow like this:

train, test = partition(eachindex(y), 0.7)
mach = machine(model, X, y)
evaluate!(mach, resampling=CV(nfolds=6),
          measures=[cross_entropy, brier_score],
          rows=train)     # cv estimate, resampling from `train`
fit!(mach, rows=train)    # re-train using all of `train` observations
predict(mach, rows=test); # and predict missing targets


# ### On learning curves

# Since our model is an iterative one, we might want to inspect the
# out-of-sample performance as a function of the iteration
# parameter. For this we can use the `learning_curve` function (which,
# incidentally can be applied to any model hyper-parameter). This
# starts by defining a one-dimensional range object for the parameter
# (more on this when we discuss tuning in Part 4):

r = range(model, :epochs, lower=1, upper=50, scale=:log10)

#-

curve = learning_curve(mach,
                       range=r,
                       resampling=Holdout(fraction_train=0.7), # (default)
                       measure=cross_entropy)

#-

using Plots
gr(size=(490,300))
plt=plot(curve.parameter_values, curve.measurements)
xlabel!(plt, "epochs")
ylabel!(plt, "cross entropy on holdout set")
savefig("learning_curve.png")
plt #!md
# ![](learning_curve.png) #md

# We will return to learning curves when we look at tuning in Part 4.


# ### Resources for Part 2

# - From the MLJ manual:
#     - [Getting Started](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/)
#     - [Model Search](https://alan-turing-institute.github.io/MLJ.jl/dev/model_search/)
#     - [Evaluating Performance](https://alan-turing-institute.github.io/MLJ.jl/dev/evaluating_model_performance/) (using `evaluate!`)
#     - [Learning Curves](https://alan-turing-institute.github.io/MLJ.jl/dev/learning_curves/)
#     - [Performance Measures](https://alan-turing-institute.github.io/MLJ.jl/dev/performance_measures/) (loss functions, scores, etc)
# - From Data Science Tutorials:
#     - [Choosing and evaluating a model](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/choosing-a-model/)
#     - [Fit, predict, transform](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/fit-and-predict/)


# ### Exercises for Part 2


# #### Exercise 4

# (a) Identify all supervised MLJ models that can be applied (without
# type coercion or one-hot encoding) to a supervised learning problem
# with input features `X4` and target `y4` defined below:

import Distributions
poisson = Distributions.Poisson

age = 18 .+ 60*rand(10);
salary = coerce(rand(["small", "big", "huge"], 10), OrderedFactor);
levels!(salary, ["small", "big", "huge"]);
small = salary[1]

#-

X4 = DataFrames.DataFrame(age=age, salary=salary)

n_devices(salary) = salary > small ? rand(poisson(1.3)) : rand(poisson(2.9))
y4 = [n_devices(row.salary) for row in eachrow(X4)]

# (b) What models can be applied if you coerce the salary to a
# `Continuous` scitype?


# #### Exercise 5 (unpack)

# After evaluating the following ...

data = (a = [1, 2, 3, 4],
        b = rand(4),
        c = rand(4),
        d = coerce(["male", "female", "female", "male"], OrderedFactor));
pretty(data)

#-

using Tables

y, X, w = unpack(data,
                 ==(:a),
                 name -> elscitype(Tables.getcolumn(data, name)) == Continuous);

# ...attempt to guess the evaluations of the following:

y

#-

pretty(X)

#-

w

# #### Exercise 6 (first steps in modeling Horse Colic)

# Here is the Horse Colic data introduced in Part 1, together with the
# type coercions we performed there:

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

# (a) Suppose we want to use predict the `:outcome` variable, based on
# the remaining variables that are `Continuous` (one-hot encoding
# categorical variables is discussed later in Part 3) *while ignoring
# the others*.  Extract from the `horse` data set (defined in Part 1)
# appropriate input features `X` and target variable `y`. (Do not,
# however, randomize the observations.)

# (b) Create a 70:30 `train`/`test` split of the data and train a
# `LogisticClassifier` model, from the `MLJLinearModels` package, on
# the `train` rows. Use `lambda=100` and default values for the
# other hyper-parameters. (Although one would normally standardize
# (whiten) the continuous features for this model, do not do so here.)
# After training:

# - (i) Recalling that a logistic classifier (aka logistic regressor) is
#   a linear-based model learning a *vector* of coefficients for each
#   feature (one coefficient for each target class), use the
#   `fitted_params` method to find this vector of coefficients in the
#   case of the `:pulse` feature. (You can convert a vector of pairs `v =
#   [x1 => y1, x2 => y2, ...]` into a dictionary with `Dict(v)`.)

# - (ii) Evaluate the `cross_entropy` performance on the `test`
#   observations.

# - &star;(iii) In how many `test` observations does the predicted
#   probability of the observed class exceed 50%?

# - (iv) Find the `misclassification_rate` in the `test`
#   set. (*Hint.* As this measure is deterministic, you will either
#   need to broadcast `mode` or use `predict_mode` instead of
#   `predict`.)

# (c) Instead use a `RandomForestClassifier` model from the
#     `DecisionTree` package and:
#
# - (i) Generate an appropriate learning curve to convince yourself
#   that out-of-sample estimates of the `cross_entropy` loss do not
#   substantially improve for `n_trees > 50`. Use default values for
#   all other hyper-parameters, and feel free to use all available
#   data to generate the curve.

# - (ii) Fix `n_trees=90` and use `evaluate!` to obtain a 9-fold
#   cross-validation estimate of the `cross_entropy`, restricting
#   sub-sampling to the `train` observations.

# - (iii) Now use *all* available data but set
#   `resampling=Holdout(fraction_train=0.7)` to obtain a score you can
#   compare with the `KNNClassifier` in part (b)(iii). Which model is
#   better?

# <a id='part-3-transformers-and-pipelines'></a>

