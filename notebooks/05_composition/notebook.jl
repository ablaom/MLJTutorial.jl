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


# ## Part 5 - Advanced Model Composition

# > **Goals:**
# > 1. Learn how to build a prototypes of a composite model, called a *learning network*
# > 2. Learn how to use the `@from_network` macro to export a learning network as a new stand-alone model type

# Pipelines are great for composing models in an unbranching
# sequence. Another built-in type of model composition is a model
# *stack*; see
# [here](https://alan-turing-institute.github.io/MLJ.jl/dev/model_stacking/#Model-Stacking)
# for details. For other more complicated model compositions you'll want to
# use MLJ's generic model composition syntax. There are two main
# steps:

# - **Prototype** the composite model by building a *learning
#   network*, which can be tested on some (dummy) data as you build
#   it.

# - **Export** the learning network as a new stand-alone model type.

# Like pipeline models, instances of the exported model type behave
# like any other model (and are not bound to any data, until you wrap
# them in a machine).


# ### Building a pipeline using the generic composition syntax

# To warm up, we'll do the equivalent of

using MLJ
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
pipe = Standardizer |> LogisticClassifier;

# using the generic syntax.

# Here's some dummy data we'll be using to test our learning network:

X, y = make_blobs(5, 3)
pretty(X)

# **Step 0** - Proceed as if you were combining the models "by hand",
# using all the data available for training, transforming and
# prediction:

stand = Standardizer();
linear = LogisticClassifier();

mach1 = machine(stand, X);
fit!(mach1);
Xstand = transform(mach1, X);

mach2 = machine(linear, Xstand, y);
fit!(mach2);
yhat = predict(mach2, Xstand)

# **Step 1** - Edit your code as follows:

# - pre-wrap the data in `Source` nodes

# - delete the `fit!` calls

X = source(X)  # or X = source() if not testing
y = source(y)  # or y = source()

stand = Standardizer();
linear = LogisticClassifier();

mach1 = machine(stand, X);
Xstand = transform(mach1, X);

mach2 = machine(linear, Xstand, y);
yhat = predict(mach2, Xstand)

# Now `X`, `y`, `Xstand` and `yhat` are *nodes* ("variables" or
# "dynammic data") instead of data. All training, predicting and
# transforming is now executed lazily, whenever we `fit!` one of these
# nodes. We *call* a node to retrieve the data it represents in the
# original manual workflow.

fit!(Xstand)
Xstand() |> pretty

#-

fit!(yhat);
yhat()

# The node `yhat` is the "descendant" (in an associated DAG we have
# defined) of a unique source node:

sources(yhat)

#-

# The data at the source node is replaced by `Xnew` to obtain a
# new prediction when we call `yhat` like this:

Xnew, _ = make_blobs(2, 3);
yhat(Xnew)


# **Step 2** - Export the learning network as a new stand-alone model type

# Now, somewhat paradoxically, we can wrap the whole network in a
# special machine - called a *learning network machine* - before have
# defined the new model type. Indeed doing so is a necessary step in
# the export process, for this machine will tell the export macro:

# - what kind of model the composite will be (`Deterministic`,
#   `Probabilistic` or `Unsupervised`)a

# - which source nodes are input nodes and which are for the target

# - which nodes correspond to each operation (`predict`, `transform`,
#   etc) that we might want to define

surrogate = Probabilistic()     # a model with no fields!
mach = machine(surrogate, X, y; predict=yhat)

# Although we have no real need to use it, this machine behaves like
# you'd expect it to:

Xnew, _ = make_blobs(2, 3)
fit!(mach)
predict(mach, Xnew)

#-

# Now we create a new model type using a Julia `struct` definition
# appropriately decorated:

@from_network mach begin
    mutable struct YourPipe
        standardizer = stand
        classifier = linear::Probabilistic
    end
end

# Instantiating and evaluating on some new data:

pipe = YourPipe()
X, y = @load_iris;   # built-in data set
mach = machine(pipe, X, y)
evaluate!(mach, measure=misclassification_rate, operation=predict_mode)


# ### A composite model to average two regressor predictors

# The following is condensed version of
# [this](https://github.com/alan-turing-institute/MLJ.jl/blob/master/binder/MLJ_demo.ipynb)
# tutorial. We will define a composite model that:

# - standardizes the input data

# - learns and applies a Box-Cox transformation to the target variable

# - blends the predictions of two supervised learning models - a ridge
#  regressor and a random forest regressor; we'll blend using a simple
#  average (for a more sophisticated stacking example, see
#  [here](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/stacking/))

# - applies the *inverse* Box-Cox transformation to this blended prediction

RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree
RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels

# **Input layer**

X = source()
y = source()

# **First layer and target transformation**

std_model = Standardizer()
stand = machine(std_model, X)
W = MLJ.transform(stand, X)

box_model = UnivariateBoxCoxTransformer()
box = machine(box_model, y)
z = MLJ.transform(box, y)

# **Second layer**

ridge_model = RidgeRegressor(lambda=0.1)
ridge = machine(ridge_model, W, z)

forest_model = RandomForestRegressor(n_trees=50)
forest = machine(forest_model, W, z)

ẑ = 0.5*predict(ridge, W) + 0.5*predict(forest, W)

# **Output**

ŷ = inverse_transform(box, ẑ)

# With the learning network defined, we're ready to export:

@from_network machine(Deterministic(), X, y, predict=ŷ) begin
    mutable struct CompositeModel
        rgs1 = ridge_model
        rgs2 = forest_model
    end
end

# Let's instantiate the new model type and try it out on some data:

composite = CompositeModel()

#-

X, y = @load_boston;
mach = machine(composite, X, y);
evaluate!(mach,
          resampling=CV(nfolds=6, shuffle=true),
          measures=[rms, mae])


# ### Resources for Part 5
#
# - From the MLJ manual:
#    - [Learning Networks](https://alan-turing-institute.github.io/MLJ.jl/stable/composing_models/#Learning-Networks-1)
# - From Data Science Tutorials:
#     - [Learning Networks](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/learning-networks/)
#     - [Learning Networks 2](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/learning-networks-2/)
#     - [Stacking](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/stacking/): an advanced example of model composition
#     - [Finer Control](https://alan-turing-institute.github.io/MLJ.jl/dev/composing_models/#Method-II:-Finer-control-(advanced)-1):
#       exporting learning networks without a macro for finer control

# <a id='solutions-to-exercises'></a>
