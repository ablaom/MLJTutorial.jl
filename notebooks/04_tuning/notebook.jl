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


# ## Part 4 - Tuning Hyper-parameters

# ### Naive tuning of a single parameter

# The most naive way to tune a single hyper-parameter is to use
# `learning_curve`, which we already saw in Part 2. Let's see this in
# the Horse Colic classification problem, a case where the parameter
# to be tuned is *nested* (because the model is a pipeline).

# Here is the Horse Colic data again, with the type coercions we
# already discussed in Part 1:

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

# Now for a pipeline model:

LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
model = @pipeline Standardizer ContinuousEncoder LogisticClassifier
mach = machine(model, X, y)

#-

r = range(model, :(logistic_classifier.lambda), lower = 1e-2, upper=100, scale=:log10)

# If you're curious, you can see what `lambda` values this range will
# generate for a given resolution:

iterator(r, 5)

#-

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
plt #!md

# ![](learning_curve2.png) #md

best_lambda = lambdas[argmin(losses)]


# ### Self tuning models

# A more sophisticated way to view hyper-parameter tuning (inspired by
# MLR) is as a model *wrapper*. The wrapped model is a new model in
# its own right and when you fit it, it tunes specified
# hyper-parameters of the model being wrapped, before training on all
# supplied data. Calling `predict` on the wrapped model is like
# calling `predict` on the original model, but with the
# hyper-parameters already optimized.

# In other words, we can think of the wrapped model as a "self-tuning"
# version of the original.

# We now create a self-tuning version of the pipeline above, adding a
# parameter from the `ContinuousEncoder` to the parameters we want
# optimized.

# First, let's choose a tuning strategy (from [these
# options](https://github.com/juliaai/MLJTuning.jl#what-is-provided-here)). MLJ
# supports ordinary `Grid` search (query `?Grid` for
# details). However, as the utility of `Grid` search is limited to a
# small number of parameters, and as `Grid` searches are demonstrated
# elsewhere (see the [resources below](#resources-for-part-4)) we'll
# demonstrate `RandomSearch` here:

tuning = RandomSearch(rng=123)

# In this strategy each parameter is sampled according to a
# pre-specified prior distribution that is fit to the one-dimensional
# range object constructed using `range` as before. While one has a
# lot of control over the specification of the priors (run
# `?RandomSearch` for details) we'll let the algorithm generate these
# priors automatically.


# #### Unbounded ranges and sampling

# In MLJ a range does not have to be bounded. In a `RandomSearch` a
# positive unbounded range is sampled using a `Gamma` distribution, by
# default:

r = range(model,
          :(logistic_classifier.lambda),
          lower=0,
          origin=6,
          unit=5,
          scale=:log10)

# The `scale` in a range makes no in a `RandomSearch` (unless it is a
# function) but this will effect later plots but it does effect the
# later plots.

# Let's see what sampling using a Gamma distribution is going to mean
# for this range:

import Distributions
sampler_r = sampler(r, Distributions.Gamma)
plt = histogram(rand(sampler_r, 10000), nbins=50)
savefig("gamma_sampler.png")
plt #!md

# ![](gamma_sampler.png) #md

# The second parameter that we'll add to this is *nominal* (finite) and, by
# default, will be sampled uniformly. Since it is nominal, we specify
# `values` instead of `upper` and `lower` bounds:

s  = range(model, :(continuous_encoder.one_hot_ordered_factors),
           values = [true, false])


# #### The tuning wrapper

# Now for the wrapper, which is an instance of `TunedModel`:

tuned_model = TunedModel(model=model,
                         ranges=[r, s],
                         resampling=CV(nfolds=6),
                         measures=cross_entropy,
                         tuning=tuning,
                         n=15)

# We can apply the `fit!/predict` work-flow to `tuned_model` just as
# for any other model:

tuned_mach = machine(tuned_model, X, y);
fit!(tuned_mach);
predict(tuned_mach, rows=1:3)

# The outcomes of the tuning can be inspected from a detailed
# report. For example, we have:

rep = report(tuned_mach);
rep.best_model

# By default, sampling of a bounded range is uniform. Lets

# In the special case of two-parameters, you can also plot the results:

plt = plot(tuned_mach)
savefig("tuning.png")
plt #!md

# ![](tuning.png) #md

# Finally, let's compare cross-validation estimate of the performance
# of the self-tuning model with that of the original model (an example
# of [*nested
# resampling*]((https://mlr.mlr-org.com/articles/tutorial/nested_resampling.html)
# here):

err = evaluate!(mach, resampling=CV(nfolds=3), measure=cross_entropy)

#-

tuned_err = evaluate!(tuned_mach, resampling=CV(nfolds=3), measure=cross_entropy)

# <a id='resources-for-part-4'></a>


# ### Resources for Part 4
#
# - From the MLJ manual:
#    - [Learning Curves](https://alan-turing-institute.github.io/MLJ.jl/dev/learning_curves/)
#    - [Tuning Models](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/)
# - The [MLJTuning repo](https://github.com/juliaai/MLJTuning.jl#who-is-this-repo-for) - mostly for developers
#
# - From Data Science Tutorials:
#     - [Tuning a model](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/model-tuning/)
#     - [Crabs with XGBoost](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/crabs-xgb/) `Grid` tuning in stages for a tree-boosting model with many parameters
#     - [Boston with LightGBM](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/boston-lgbm/) -  `Grid` tuning for another popular tree-booster
#     - [Boston with Flux](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/boston-flux/) - optimizing batch size in a simple neural network regressor
# - [UCI Horse Colic Data Set](http://archive.ics.uci.edu/ml/datasets/Horse+Colic)


# ### Exercises for Part 4

# #### Exercise 8

# This exercise continues our analysis of the King County House price
# prediction problem (Part 1, Exercise 3 and Part 2):

house_csv = urldownload("https://raw.githubusercontent.com/ablaom/"*
                        "MachineLearningInJulia2020/for-MLJ-version-0.16/"*
                        "data/house.csv");
house = DataFrames.DataFrame(house_csv)
coerce!(house, autotype(house_csv));
coerce!(house, Count => Continuous, :zipcode => Multiclass);
y, X = unpack(house, ==(:price), name -> true, rng=123);
schema(X)

# Your task will be to tune the following pipeline regression model,
# which includes a gradient tree boosting component:

EvoTreeRegressor = @load EvoTreeRegressor
tree_booster = EvoTreeRegressor(nrounds = 70)
model = @pipeline ContinuousEncoder tree_booster

# (a) Construct a bounded range `r1` for the `evo_tree_booster`
# parameter `max_depth`, varying between 1 and 12.

# \star&(b) For the `nbins` parameter of the `EvoTreeRegressor`, define the range

r2 = range(model,
           :(evo_tree_regressor.nbins),
           lower = 2.5,
           upper= 7.5, scale=x->2^round(Int, x))

# Notice that in this case we've specified a *function* instead of a
# canned scale, like `:log10`. In this case the `scale` function is
# applied after sampling (uniformly) between the limits of `lower` and
# `upper`. Perhaps you can guess the outputs of the following lines of
# code?

r2_sampler = sampler(r2, Distributions.Uniform)
samples = rand(r2_sampler, 1000);
plt = histogram(samples, nbins=50)
savefig("uniform_sampler.png")

plt #!md

# ![](uniform_sampler.png)

sort(unique(samples))

# (c) Optimize `model` over these the parameter ranges `r1` and `r2`
# using a random search with uniform priors (the default). Use
# `Holdout()` resampling, and implement your search by first
# constructing a "self-tuning" wrap of `model`, as described
# above. Make `mae` (mean absolute error) the loss function that you
# optimize, and search over a total of 40 combinations of
# hyper-parameters.  If you have time, plot the results of your
# search. Feel free to use all available data.

# (d) Evaluate the best model found in the search using 3-fold
# cross-validation and compare with that of the self-tuning model
# (which is different!). Setting data hygiene concerns aside, feel
# free to use all available data.

# <a id='part-5-advanced-model-composition'>
