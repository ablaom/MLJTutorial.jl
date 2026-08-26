```@meta
EditURL = "notebook.jl"
```

# Lightning Tour of MLJ

To run the code in this tutorial in a live Julia session, first follow the instructions
given [here](@ref instructions).

In MLJ a *model* is just a container for hyper-parameters, and that's all. Here we will
apply several kinds of model composition before binding the resulting "meta-model" to
data in a *machine* for evaluation, using cross-validation.

Loading and instantiating a gradient tree-boosting model:

````@julia
using MLJ

Booster = @load EvoTreeRegressor # loads code defining a model type
booster = Booster(max_depth=2)   # specify hyper-parameter at construction
````

````
EvoTreeRegressor(
  loss = :mse, 
  metric = :mse, 
  nrounds = 100, 
  bagging_size = 1, 
  early_stopping_rounds = 9223372036854775807, 
  L2 = 1.0, 
  lambda = 0.0, 
  gamma = 0.0, 
  eta = 0.1, 
  max_depth = 2, 
  min_weight = 1.0, 
  rowsample = 1.0, 
  colsample = 1.0, 
  nbins = 64, 
  alpha = 0.5, 
  alphas = [0.1, 0.5, 0.9], 
  monotone_constraints = Dict{Int64, Int64}(), 
  tree_type = :binary, 
  seed = 123, 
  device = :cpu)
````

````@julia
booster.nrounds=50               # or mutate post facto
booster
````

````
EvoTreeRegressor(
  loss = :mse, 
  metric = :mse, 
  nrounds = 50, 
  bagging_size = 1, 
  early_stopping_rounds = 9223372036854775807, 
  L2 = 1.0, 
  lambda = 0.0, 
  gamma = 0.0, 
  eta = 0.1, 
  max_depth = 2, 
  min_weight = 1.0, 
  rowsample = 1.0, 
  colsample = 1.0, 
  nbins = 64, 
  alpha = 0.5, 
  alphas = [0.1, 0.5, 0.9], 
  monotone_constraints = Dict{Int64, Int64}(), 
  tree_type = :binary, 
  seed = 123, 
  device = :cpu)
````

This model is an example of an iterative model. As is stands, the
number of iterations `nrounds` is fixed.

### Composition 1: Wrapping the model to make it "self-iterating"

Let's create a new model that automatically learns the number of iterations,
using the `NumberSinceBest(3)` criterion, as applied to an
out-of-sample `l1` loss:

````@julia
iterated_booster = IteratedModel(
    model=booster,
    resampling=Holdout(fraction_train=0.8),
    controls=[Step(2), NumberSinceBest(3), NumberLimit(300)],
    measure=l1,
    retrain=true,
)
````

````
DeterministicIteratedModel(
  model = EvoTreeRegressor(
        loss = :mse, 
        metric = :mse, 
        nrounds = 50, 
        bagging_size = 1, 
        early_stopping_rounds = 9223372036854775807, 
        L2 = 1.0, 
        lambda = 0.0, 
        gamma = 0.0, 
        eta = 0.1, 
        max_depth = 2, 
        min_weight = 1.0, 
        rowsample = 1.0, 
        colsample = 1.0, 
        nbins = 64, 
        alpha = 0.5, 
        alphas = [0.1, 0.5, 0.9], 
        monotone_constraints = Dict{Int64, Int64}(), 
        tree_type = :binary, 
        seed = 123, 
        device = :cpu), 
  controls = Any[IterationControl.Step(2), EarlyStopping.NumberSinceBest(3), EarlyStopping.NumberLimit(300)], 
  resampling = Holdout(
        fraction_train = 0.8, 
        shuffle = false, 
        rng = Random.TaskLocalRNG()), 
  measure = LPLoss(p = 1), 
  weights = nothing, 
  class_weights = nothing, 
  operation = nothing, 
  retrain = true, 
  check_measure = true, 
  iteration_parameter = nothing, 
  cache = true, 
  logger = nothing)
````

### Composition 2: Preprocess the input features

Combining the model with categorical feature encoding:

````@julia
pipe = ContinuousEncoder |> iterated_booster
````

````
DeterministicPipeline(
  continuous_encoder = ContinuousEncoder(
        drop_last = false, 
        one_hot_ordered_factors = false), 
  deterministic_iterated_model = DeterministicIteratedModel(
        model = EvoTreeRegressor(loss = mse, …), 
        controls = Any[IterationControl.Step(2), EarlyStopping.NumberSinceBest(3), EarlyStopping.NumberLimit(300)], 
        resampling = Holdout(fraction_train = 0.8, …), 
        measure = LPLoss(p = 1), 
        weights = nothing, 
        class_weights = nothing, 
        operation = nothing, 
        retrain = true, 
        check_measure = true, 
        iteration_parameter = nothing, 
        cache = true, 
        logger = nothing), 
  cache = true)
````

### Composition 3: Wrapping the model to make it "self-tuning"

First, we define a hyper-parameter range for optimization of a
(nested) hyper-parameter:

````@julia
max_depth_range = range(
    pipe,
    :(deterministic_iterated_model.model.max_depth),
    lower = 1,
    upper = 10,
)
````

````
NumericRange(1 ≤ deterministic_iterated_model.model.max_depth ≤ 10; origin=5.5, unit=4.5)
````

Now we can wrap the pipeline model in an optimization strategy to make it "self-tuning":

````@julia
self_tuning_pipe = TunedModel(
    model=pipe,
    tuning=RandomSearch(),
    ranges = max_depth_range,
    resampling=CV(nfolds=3, rng=456),
    measure=l1,
    acceleration=CPUThreads(),
    n=50,
)
````

````
DeterministicTunedModel(
  model = DeterministicPipeline(
        continuous_encoder = ContinuousEncoder(drop_last = false, …), 
        deterministic_iterated_model = DeterministicIteratedModel(model = EvoTreeRegressor(loss = mse, …), …), 
        cache = true), 
  tuning = RandomSearch(
        bounded = Distributions.Uniform, 
        positive_unbounded = Distributions.Gamma, 
        other = Distributions.Normal, 
        rng = Random.TaskLocalRNG()), 
  resampling = CV(
        nfolds = 3, 
        shuffle = true, 
        rng = Random.MersenneTwister(456)), 
  measure = LPLoss(p = 1), 
  weights = nothing, 
  class_weights = nothing, 
  operation = nothing, 
  range = NumericRange(1 ≤ deterministic_iterated_model.model.max_depth ≤ 10; origin=5.5, unit=4.5), 
  selection_heuristic = MLJTuning.NaiveSelection(nothing), 
  train_best = true, 
  repeats = 1, 
  n = 50, 
  acceleration = ComputationalResources.CPUThreads{Int64}(1), 
  acceleration_resampling = ComputationalResources.CPU1{Nothing}(nothing), 
  check_measure = true, 
  cache = true, 
  compact_history = true, 
  logger = nothing)
````

### Binding to data and evaluating performance

Generating some synthetic data:

````@julia
X, y = make_regression();
````

Binding the "self-tuning" pipeline model to data in a *machine* (which will additionally
store *learned* parameters):

````@julia
mach = machine(self_tuning_pipe, X, y)
````

````
untrained Machine; does not cache data
  model: DeterministicTunedModel(model = DeterministicPipeline(continuous_encoder = ContinuousEncoder(drop_last = false, …), …), …)
  args: 
    1:	Source @630 ⏎ ScientificTypesBase.Table{AbstractVector{ScientificTypesBase.Continuous}}
    2:	Source @583 ⏎ AbstractVector{ScientificTypesBase.Continuous}

````

Fit and predict:

````@julia
fit!(mach, rows=1:60)
yhat = predict(mach, rows=61:100)
first(yhat, 3)
````

````
3-element Vector{Float32}:
 2.114296
 2.5297465
 2.18611
````

Evaluating the "self-tuning" pipeline model's performance using all data and 5-fold
cross-validation (implies multiple layers of nested resampling):

````@julia
evaluate!(
    mach,
    measures=[l1, rsquared],
    resampling=CV(nfolds=5, rng=123),
    acceleration=CPUThreads(),
)
````

````
PerformanceEvaluation object with these fields:
  model, tag, measure, operation,
  measurement, uncertainty_radius_95, per_fold, per_observation,
  fitted_params_per_fold, report_per_fold,
  train_test_rows, resampling, repeats
Tag: DeterministicTunedModel-323
Extract:
┌───┬────────────┬───────────┬─────────────┐
│   │ measure    │ operation │ measurement │
├───┼────────────┼───────────┼─────────────┤
│ A │ LPLoss(    │ predict   │ 0.0923      │
│   │   p = 1)   │           │             │
│ B │ RSquared() │ predict   │ 0.857       │
└───┴────────────┴───────────┴─────────────┘
┌───┬─────────────────────────────────────┬─────────┐
│   │ per_fold                            │ 1.96*SE │
├───┼─────────────────────────────────────┼─────────┤
│ A │ [0.097, 0.05, 0.0769, 0.108, 0.129] │ 0.0297  │
│ B │ [0.818, 0.965, 0.906, 0.776, 0.819] │ 0.0752  │
└───┴─────────────────────────────────────┴─────────┘

````

Compare to a dummy model:

````@julia
evaluations = evaluate(
    ["booster" => self_tuning_pipe, "dummy" => ConstantRegressor()],
    X,
    y;
    measures=[l1, rsquared],
    resampling=CV(nfolds=5, rng=123),
    acceleration=CPUThreads(),
)

describe.(evaluations) |> pretty
````

````
[ Info: Performing evaluations using 1 thread.
Evaluating over 5 folds:  40%[==========>              ]  ETA: 0:00:06[KEvaluating over 5 folds:  60%[===============>         ]  ETA: 0:00:05[KEvaluating over 5 folds:  80%[====================>    ]  ETA: 0:00:03[KEvaluating over 5 folds: 100%[=========================] Time: 0:00:12[K
[ Info: Performing evaluations using 1 thread.
Evaluating over 5 folds:  40%[==========>              ]  ETA: 0:00:01[KEvaluating over 5 folds: 100%[=========================] Time: 0:00:00[K
┌─────────┬──────────────────────┬──────────────────────┐
│ tag     │ LPLoss               │ RSquared             │
│ String  │ Measurement{Float64} │ Measurement{Float64} │
│ Textual │ Continuous           │ Continuous           │
├─────────┼──────────────────────┼──────────────────────┤
│ booster │ 0.092±0.03           │ 0.857±0.075          │
│ dummy   │ 0.292±0.054          │ -0.076±0.089         │
└─────────┴──────────────────────┴──────────────────────┘

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

