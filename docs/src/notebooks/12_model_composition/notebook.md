```@meta
EditURL = "notebook.jl"
```

# Lesson 2. Model Composition

Notebook supporting the video series "Using MLJ".

To run the code in this tutorial in a live Julia session, first follow the instructions
given [here](@ref instructions).

````@julia
using MLJ
````

load some model code:

````@julia
RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels
````

````
MLJLinearModels.RidgeRegressor
````

load some data and inspect schema:

````@julia
data = load_reduced_ames();
schema(data)
````

````
┌──────────────┬───────────────────┬──────────────────────────────────┐
│ names        │ scitypes          │ types                            │
├──────────────┼───────────────────┼──────────────────────────────────┤
│ target       │ Continuous        │ Float64                          │
│ OverallQual  │ OrderedFactor{10} │ CategoricalValue{Int64, UInt32}  │
│ GrLivArea    │ Continuous        │ Float64                          │
│ Neighborhood │ Multiclass{25}    │ CategoricalValue{String, UInt32} │
│ x1stFlrSF    │ Continuous        │ Float64                          │
│ TotalBsmtSF  │ Continuous        │ Float64                          │
│ BsmtFinSF1   │ Continuous        │ Float64                          │
│ LotArea      │ Continuous        │ Float64                          │
│ GarageCars   │ Count             │ Int64                            │
│ MSSubClass   │ Multiclass{15}    │ CategoricalValue{String, UInt32} │
│ GarageArea   │ Continuous        │ Float64                          │
│ YearRemodAdd │ Count             │ Int64                            │
│ YearBuilt    │ Count             │ Int64                            │
└──────────────┴───────────────────┴──────────────────────────────────┘

````

horizontally split with observation shuffling:

````@julia
y, X = unpack(data, ==(:target); rng=123);
schema(X)
````

````
┌──────────────┬───────────────────┬──────────────────────────────────┐
│ names        │ scitypes          │ types                            │
├──────────────┼───────────────────┼──────────────────────────────────┤
│ OverallQual  │ OrderedFactor{10} │ CategoricalValue{Int64, UInt32}  │
│ GrLivArea    │ Continuous        │ Float64                          │
│ Neighborhood │ Multiclass{25}    │ CategoricalValue{String, UInt32} │
│ x1stFlrSF    │ Continuous        │ Float64                          │
│ TotalBsmtSF  │ Continuous        │ Float64                          │
│ BsmtFinSF1   │ Continuous        │ Float64                          │
│ LotArea      │ Continuous        │ Float64                          │
│ GarageCars   │ Count             │ Int64                            │
│ MSSubClass   │ Multiclass{15}    │ CategoricalValue{String, UInt32} │
│ GarageArea   │ Continuous        │ Float64                          │
│ YearRemodAdd │ Count             │ Int64                            │
│ YearBuilt    │ Count             │ Int64                            │
└──────────────┴───────────────────┴──────────────────────────────────┘

````

defined a pipeline model:

````@julia
pipe = ContinuousEncoder() |> Standardizer() |> RidgeRegressor()
````

````
DeterministicPipeline(
  continuous_encoder = ContinuousEncoder(
        drop_last = false, 
        one_hot_ordered_factors = false), 
  standardizer = Standardizer(
        features = Symbol[], 
        ignore = false, 
        ordered_factor = false, 
        count = false), 
  ridge_regressor = RidgeRegressor(
        lambda = 1.0, 
        fit_intercept = true, 
        penalize_intercept = false, 
        scale_penalty_with_samples = true, 
        solver = nothing), 
  cache = true)
````

accessing a nested hyperparameter:

````@julia
pipe.ridge_regressor.fit_intercept
````

````
true
````

changing it:

````@julia
pipe.ridge_regressor.fit_intercept = false
````

````
false
````

evaluate the pipeline:

````@julia
evaluate(pipe, X, y; resampling=CV(nfolds=4, rng=123), repeats=2, measure=mav)
````

````
PerformanceEvaluation object with these fields:
  model, tag, measure, operation,
  measurement, uncertainty_radius_95, per_fold, per_observation,
  fitted_params_per_fold, report_per_fold,
  train_test_rows, resampling, repeats
Tag: DeterministicPipeline-245
Extract:
┌──────────┬───────────┬─────────────┐
│ measure  │ operation │ measurement │
├──────────┼───────────┼─────────────┤
│ LPLoss(  │ predict   │ 180000.0    │
│   p = 1) │           │             │
└──────────┴───────────┴─────────────┘
┌───────────────────────────────────────────────────────────────────────────────
│ per_fold                                                                     ⋯
├───────────────────────────────────────────────────────────────────────────────
│ [180000.0, 180000.0, 181000.0, 179000.0, 179000.0, 180000.0, 180000.0, 18100 ⋯
└───────────────────────────────────────────────────────────────────────────────
                                                               2 columns omitted

````

look at the target:

````@julia
@show mean(y) std(y)
````

````
76696.59253004662
````

wrap in target normalization:

````@julia
norm_pipe = TransformedTargetModel(pipe, transformer=Standardizer())
````

````
TransformedTargetModelDeterministic(
  model = DeterministicPipeline(
        continuous_encoder = ContinuousEncoder(drop_last = false, …), 
        standardizer = Standardizer(features = Symbol[], …), 
        ridge_regressor = RidgeRegressor(lambda = 1.0, …), 
        cache = true), 
  transformer = Standardizer(
        features = Symbol[], 
        ignore = false, 
        ordered_factor = false, 
        count = false), 
  inverse = nothing, 
  cache = true)
````

evaluate performance:

````@julia
evaluate(norm_pipe, X, y; resampling=CV(nfolds=4, rng=123), repeats=2, measure=mav)
````

````
PerformanceEvaluation object with these fields:
  model, tag, measure, operation,
  measurement, uncertainty_radius_95, per_fold, per_observation,
  fitted_params_per_fold, report_per_fold,
  train_test_rows, resampling, repeats
Tag: TransformedTargetModelDeterministic-843
Extract:
┌──────────┬───────────┬─────────────┐
│ measure  │ operation │ measurement │
├──────────┼───────────┼─────────────┤
│ LPLoss(  │ predict   │ 19800.0     │
│   p = 1) │           │             │
└──────────┴───────────┴─────────────┘
┌──────────────────────────────────────────────────────────────────────────┬────
│ per_fold                                                                 │ 1 ⋯
├──────────────────────────────────────────────────────────────────────────┼────
│ [18100.0, 19800.0, 21400.0, 19600.0, 20300.0, 19500.0, 18700.0, 20900.0] │ 8 ⋯
└──────────────────────────────────────────────────────────────────────────┴────
                                                                1 column omitted

````

horizontally split with observation shuffling:

````@julia
y, X = unpack(data, ==(:target); rng=123)
schema(X)
````

````
┌──────────────┬───────────────────┬──────────────────────────────────┐
│ names        │ scitypes          │ types                            │
├──────────────┼───────────────────┼──────────────────────────────────┤
│ OverallQual  │ OrderedFactor{10} │ CategoricalValue{Int64, UInt32}  │
│ GrLivArea    │ Continuous        │ Float64                          │
│ Neighborhood │ Multiclass{25}    │ CategoricalValue{String, UInt32} │
│ x1stFlrSF    │ Continuous        │ Float64                          │
│ TotalBsmtSF  │ Continuous        │ Float64                          │
│ BsmtFinSF1   │ Continuous        │ Float64                          │
│ LotArea      │ Continuous        │ Float64                          │
│ GarageCars   │ Count             │ Int64                            │
│ MSSubClass   │ Multiclass{15}    │ CategoricalValue{String, UInt32} │
│ GarageArea   │ Continuous        │ Float64                          │
│ YearRemodAdd │ Count             │ Int64                            │
│ YearBuilt    │ Count             │ Int64                            │
└──────────────┴───────────────────┴──────────────────────────────────┘

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

