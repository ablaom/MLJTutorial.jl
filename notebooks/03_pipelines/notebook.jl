# # Machine Learning in Julia

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


# ## Part 3 - Transformers and Pipelines

# ### Transformers

# Unsupervised models, which receive no target `y` during training,
# always have a `transform` operation. They sometimes also support an
# `inverse_transform` operation, with obvious meaning, and sometimes
# support a `predict` operation (see the clustering example discussed
# [here](https://alan-turing-institute.github.io/MLJ.jl/dev/transformers/#Transformers-that-also-predict-1)).
# Otherwise, they are handled much like supervised models.

# Here's a simple standardization example:

using MLJ

x = rand(100);
@show mean(x) std(x);

#-

model = Standardizer() # a built-in model
mach = machine(model, x)
fit!(mach)
xhat = transform(mach, x);
@show mean(xhat) std(xhat);

# This particular model has an `inverse_transform`:

inverse_transform(mach, xhat) ≈ x


# ### Re-encoding the King County House data as continuous

# For further illustrations of transformers, let's re-encode *all* of the
# King County House input features (see [Ex
# 3](#exercise-3-fixing-scitypes-in-a-table)) into a set of `Continuous`
# features. We do this with the `ContinuousEncoder` model, which, by
# default, will:

# - one-hot encode all `Multiclass` features
# - coerce all `OrderedFactor` features to `Continuous` ones
# - coerce all `Count` features to `Continuous` ones (there aren't any)
# - drop any remaining non-Continuous features (none of these either)

# First, we reload the data and fix the scitypes (Exercise 3):

using UrlDownload, CSV, DataFrames
house_csv = urldownload("https://raw.githubusercontent.com/ablaom/"*
                        "MachineLearningInJulia2020/for-MLJ-version-0.16/"*
                        "data/house.csv");
house = DataFrames.DataFrame(house_csv)
coerce!(house, autotype(house_csv));
coerce!(house, Count => Continuous, :zipcode => Multiclass);
schema(house)

#-

y, X = unpack(house, ==(:price), name -> true, rng=123);

# Instantiate the unsupervised model (transformer):

encoder = ContinuousEncoder() # a built-in model; no need to @load it

# Bind the model to the data and fit!

mach = machine(encoder, X) |> fit!;

# Transform and inspect the result:

Xcont = transform(mach, X);
schema(Xcont)


# ### More transformers

# Here's how to list all of MLJ's unsupervised models:

models(m->!m.is_supervised)

# Some commonly used ones are built-in (do not require `@load`ing):

# model type                  | does what?
# ----------------------------|----------------------------------------------
# ContinuousEncoder | transform input table to a table of `Continuous` features (see above)
# FeatureSelector | retain or dump selected features
# FillImputer | impute missing values
# OneHotEncoder | one-hot encoder `Multiclass` (and optionally `OrderedFactor`) features
# Standardizer | standardize (whiten) a vector or all `Continuous` features of a table
# UnivariateBoxCoxTransformer | apply a learned Box-Cox transformation to a vector
# UnivariateDiscretizer | discretize a `Continuous` vector, and hence render its elscitypw `OrderedFactor`


# In addition to "dynamic" transformers (ones that learn something
# from the data and must be `fit!`) users can wrap ordinary functions
# as transformers, and such *static* transformers can depend on
# parameters, like the dynamic ones. See
# [here](https://alan-turing-institute.github.io/MLJ.jl/dev/transformers/#Static-transformers-1)
# for how to define your own static transformers.


# ### Pipelines

length(schema(Xcont).names)

# Let's suppose that additionally we'd like to reduce the dimension of
# our data.  A model that will do this is `PCA` from
# `MultivariateStats.jl`:

PCA = @load PCA
reducer = PCA()

# Now, rather simply repeating the work-flow above, applying the new
# transformation to `Xcont`, we can combine both the encoding and the
# dimension-reducing models into a single model, known as a
# *pipeline*. While MLJ offers a powerful interface for composing
# models in a variety of ways, we'll stick to these simplest class of
# composite models for now. The easiest way to construct them is using
# the `@pipeline` macro:

pipe = @pipeline encoder reducer

# Notice that `pipe` is an *instance* of an automatically generated
# type (called `Pipeline<some digits>`).

# The new model behaves like any other transformer:

mach = machine(pipe, X)
fit!(mach)
Xsmall = transform(mach, X)
schema(Xsmall)

# Want to combine this pre-processing with ridge regression?

RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels
rgs = RidgeRegressor()
pipe2 = @pipeline encoder reducer rgs

# Now our pipeline is a supervised model, instead of a transformer,
# whose performance we can evaluate:

mach = machine(pipe2, X, y)
evaluate!(mach, measure=mae, resampling=Holdout()) # CV(nfolds=6) is default


# ### Training of composite models is "smart"

# Now notice what happens if we train on all the data, then change a
# regressor hyper-parameter and retrain:

fit!(mach)

#-

pipe2.ridge_regressor.lambda = 0.1
fit!(mach)

# Second time only the ridge regressor is retrained!

# Mutate a hyper-parameter of the `PCA` model and every model except
# the `ContinuousEncoder` (which comes before it will be retrained):

pipe2.pca.pratio = 0.9999
fit!(mach)


# ### Inspecting composite models

# The dot syntax used above to change the values of *nested*
# hyper-parameters is also useful when inspecting the learned
# parameters and report generated when training a composite model:

fitted_params(mach).ridge_regressor

#-

report(mach).pca


# ### Incorporating target transformations

# Next, suppose that instead of using the raw `:price` as the
# training target, we want to use the log-price (a common practice in
# dealing with house price data). However, suppose that we still want
# to report final *predictions* on the original linear scale (and use
# these for evaluation purposes). Then we supply appropriate functions
# to key-word arguments `target` and `inverse`.

# First we'll overload `log` and `exp` for broadcasting:
Base.log(v::AbstractArray) = log.(v)
Base.exp(v::AbstractArray) = exp.(v)

# Now for the new pipeline:

pipe3 = @pipeline encoder reducer rgs target=log inverse=exp
mach = machine(pipe3, X, y)
evaluate!(mach, measure=mae)

# MLJ will also allow you to insert *learned* target
# transformations. For example, we might want to apply
# `Standardizer()` to the target, to standardize it, or
# `UnivariateBoxCoxTransformer()` to make it look Gaussian. Then
# instead of specifying a *function* for `target`, we specify a
# unsupervised *model* (or model type). One does not specify `inverse`
# because only models implementing `inverse_transform` are
# allowed.

# Let's see which of these two options results in a better outcome:

box = UnivariateBoxCoxTransformer(n=20)
stand = Standardizer()

pipe4 = @pipeline encoder reducer rgs target=box
mach = machine(pipe4, X, y)
evaluate!(mach, measure=mae)

#-

pipe4.target = stand
evaluate!(mach, measure=mae)


# ### Resources for Part 3

# - From the MLJ manual:
#     - [Transformers and other unsupervised models](https://alan-turing-institute.github.io/MLJ.jl/dev/transformers/)
#     - [Linear pipelines](https://alan-turing-institute.github.io/MLJ.jl/dev/linear_pipelines/#Linear-Pipelines)
# - From Data Science Tutorials:
#     - [Composing models](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/composing-models/)


# ### Exercises for Part 3

# #### Exercise 7

# Consider again the Horse Colic classification problem considered in
# Exercise 6, but with all features, `Finite` and `Infinite`:

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

# (a) Define a pipeline that:
# - uses `Standardizer` to ensure that features that are already
#   continuous are centered at zero and have unit variance
# - re-encodes the full set of features as `Continuous`, using
#   `ContinuousEncoder`
# - uses the `KMeans` clustering model from `Clustering.jl`
#   to reduce the dimension of the feature space to `k=10`.
# - trains a `EvoTreeClassifier` (a gradient tree boosting
#   algorithm in `EvoTrees.jl`) on the reduced data, using
#   `nrounds=50` and default values for the other
#    hyper-parameters

# (b) Evaluate the pipeline on all data, using 6-fold cross-validation
# and `cross_entropy` loss.

# &star;(c) Plot a learning curve which examines the effect on this loss
# as the tree booster parameter `max_depth` varies from 2 to 10.

# <a id='part-4-tuning-hyper-parameters'></a>

