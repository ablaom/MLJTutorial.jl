```@meta
EditURL = "notebook.jl"
```

# Tutorial 3. Transformers and Pipelines

### Transformers

Unsupervised models, which receive no target `y` during training, always have a
`transform` operation. They sometimes also support an `inverse_transform` operation,
with obvious meaning, and sometimes support a `predict` operation (see the clustering
example discussed
[here](https://juliaai.github.io/MLJ.jl/dev/transformers/#Transformers-that-also-predict-1)).
Otherwise, they are handled much like supervised models.

Here's a simple standardization example:

````@julia
using MLJ

x = rand(100);
@show mean(x) std(x);
````

````
mean(x) = 0.45242734888566294
std(x) = 0.30973127101002423

````

````@julia
model = Standardizer() # a built-in model
mach = machine(model, x)
fit!(mach)
xhat = transform(mach, x);
@show mean(xhat) std(xhat);
````

````
[ Info: Training machine(Standardizer(features = Symbol[], …), …).
mean(xhat) = 4.263256414560601e-16
std(xhat) = 1.0

````

This particular model has an `inverse_transform`:

````@julia
inverse_transform(mach, xhat) ≈ x
````

````
true
````

### Re-encoding the King County House data as continuous

For further illustrations of transformers, let's re-encode *all* of the King County
House input features (see Exercise 3) into a set of `Continuous` features. We do this
with the `ContinuousEncoder` model, which, by default, will:

- one-hot encode all `Multiclass` features
- coerce all `OrderedFactor` features to `Continuous` ones
- coerce all `Count` features to `Continuous` ones (there aren't any)
- drop any remaining non-Continuous features (none of these either)

First, we reload the data and fix the scitypes (Exercise 3):

````@julia
import Downloads, CSV
import DataFrames
url = "https://raw.githubusercontent.com/ablaom/"*
    "MachineLearningInJulia2020/for-MLJ-version-0.16/"*
    "data/house.csv"
csv_file = Downloads.download(url)
house = CSV.read(csv_file, DataFrames.DataFrame)
coerce!(house, autotype(house))
coerce!(house, Count => Continuous, :zipcode => Multiclass)
schema(house)
````

````
┌───────────────┬───────────────────┬───────────────────────────────────┐
│ names         │ scitypes          │ types                             │
├───────────────┼───────────────────┼───────────────────────────────────┤
│ price         │ Continuous        │ Float64                           │
│ bedrooms      │ OrderedFactor{13} │ CategoricalValue{Int64, UInt32}   │
│ bathrooms     │ OrderedFactor{30} │ CategoricalValue{Float64, UInt32} │
│ sqft_living   │ Continuous        │ Float64                           │
│ sqft_lot      │ Continuous        │ Float64                           │
│ floors        │ OrderedFactor{6}  │ CategoricalValue{Float64, UInt32} │
│ waterfront    │ OrderedFactor{2}  │ CategoricalValue{Int64, UInt32}   │
│ view          │ OrderedFactor{5}  │ CategoricalValue{Int64, UInt32}   │
│ condition     │ OrderedFactor{5}  │ CategoricalValue{Int64, UInt32}   │
│ grade         │ OrderedFactor{12} │ CategoricalValue{Int64, UInt32}   │
│ sqft_above    │ Continuous        │ Float64                           │
│ sqft_basement │ Continuous        │ Float64                           │
│ yr_built      │ Continuous        │ Float64                           │
│ zipcode       │ Multiclass{70}    │ CategoricalValue{Int64, UInt32}   │
│ lat           │ Continuous        │ Float64                           │
│ long          │ Continuous        │ Float64                           │
│ sqft_living15 │ Continuous        │ Float64                           │
│ sqft_lot15    │ Continuous        │ Float64                           │
│ is_renovated  │ OrderedFactor{2}  │ CategoricalValue{Bool, UInt32}    │
└───────────────┴───────────────────┴───────────────────────────────────┘

````

````@julia
y, X = unpack(house, ==(:price), rng=123);
````

Instantiate the unsupervised model (transformer):

````@julia
encoder = ContinuousEncoder() # a built-in model; no need to @load it
````

````
ContinuousEncoder(
  drop_last = false, 
  one_hot_ordered_factors = false)
````

Bind the model to the data and fit!

````@julia
mach = machine(encoder, X) |> fit!;
````

````
[ Info: Training machine(ContinuousEncoder(drop_last = false, …), …).

````

Transform and inspect the result:

````@julia
Xcont = transform(mach, X);
schema(Xcont)
````

````
┌────────────────┬────────────┬─────────┐
│ names          │ scitypes   │ types   │
├────────────────┼────────────┼─────────┤
│ bedrooms       │ Continuous │ Float64 │
│ bathrooms      │ Continuous │ Float64 │
│ sqft_living    │ Continuous │ Float64 │
│ sqft_lot       │ Continuous │ Float64 │
│ floors         │ Continuous │ Float64 │
│ waterfront     │ Continuous │ Float64 │
│ view           │ Continuous │ Float64 │
│ condition      │ Continuous │ Float64 │
│ grade          │ Continuous │ Float64 │
│ sqft_above     │ Continuous │ Float64 │
│ sqft_basement  │ Continuous │ Float64 │
│ yr_built       │ Continuous │ Float64 │
│ zipcode__98001 │ Continuous │ Float64 │
│ zipcode__98002 │ Continuous │ Float64 │
│ zipcode__98003 │ Continuous │ Float64 │
│ zipcode__98004 │ Continuous │ Float64 │
│ zipcode__98005 │ Continuous │ Float64 │
│ zipcode__98006 │ Continuous │ Float64 │
│ zipcode__98007 │ Continuous │ Float64 │
│ zipcode__98008 │ Continuous │ Float64 │
│ zipcode__98010 │ Continuous │ Float64 │
│ zipcode__98011 │ Continuous │ Float64 │
│ zipcode__98014 │ Continuous │ Float64 │
│ zipcode__98019 │ Continuous │ Float64 │
│ zipcode__98022 │ Continuous │ Float64 │
│ zipcode__98023 │ Continuous │ Float64 │
│ zipcode__98024 │ Continuous │ Float64 │
│ zipcode__98027 │ Continuous │ Float64 │
│ zipcode__98028 │ Continuous │ Float64 │
│ zipcode__98029 │ Continuous │ Float64 │
│ zipcode__98030 │ Continuous │ Float64 │
│ zipcode__98031 │ Continuous │ Float64 │
│ zipcode__98032 │ Continuous │ Float64 │
│ zipcode__98033 │ Continuous │ Float64 │
│ zipcode__98034 │ Continuous │ Float64 │
│ zipcode__98038 │ Continuous │ Float64 │
│ zipcode__98039 │ Continuous │ Float64 │
│ zipcode__98040 │ Continuous │ Float64 │
│ zipcode__98042 │ Continuous │ Float64 │
│ zipcode__98045 │ Continuous │ Float64 │
│ zipcode__98052 │ Continuous │ Float64 │
│ zipcode__98053 │ Continuous │ Float64 │
│ zipcode__98055 │ Continuous │ Float64 │
│ zipcode__98056 │ Continuous │ Float64 │
│ zipcode__98058 │ Continuous │ Float64 │
│ zipcode__98059 │ Continuous │ Float64 │
│ zipcode__98065 │ Continuous │ Float64 │
│ zipcode__98070 │ Continuous │ Float64 │
│ zipcode__98072 │ Continuous │ Float64 │
│ zipcode__98074 │ Continuous │ Float64 │
│ zipcode__98075 │ Continuous │ Float64 │
│ zipcode__98077 │ Continuous │ Float64 │
│ zipcode__98092 │ Continuous │ Float64 │
│ zipcode__98102 │ Continuous │ Float64 │
│ zipcode__98103 │ Continuous │ Float64 │
│ zipcode__98105 │ Continuous │ Float64 │
│ zipcode__98106 │ Continuous │ Float64 │
│ zipcode__98107 │ Continuous │ Float64 │
│ zipcode__98108 │ Continuous │ Float64 │
│ zipcode__98109 │ Continuous │ Float64 │
│ zipcode__98112 │ Continuous │ Float64 │
│ zipcode__98115 │ Continuous │ Float64 │
│ zipcode__98116 │ Continuous │ Float64 │
│ zipcode__98117 │ Continuous │ Float64 │
│ zipcode__98118 │ Continuous │ Float64 │
│ zipcode__98119 │ Continuous │ Float64 │
│ zipcode__98122 │ Continuous │ Float64 │
│ zipcode__98125 │ Continuous │ Float64 │
│ zipcode__98126 │ Continuous │ Float64 │
│ zipcode__98133 │ Continuous │ Float64 │
│ zipcode__98136 │ Continuous │ Float64 │
│ zipcode__98144 │ Continuous │ Float64 │
│ zipcode__98146 │ Continuous │ Float64 │
│ zipcode__98148 │ Continuous │ Float64 │
│ zipcode__98155 │ Continuous │ Float64 │
│ zipcode__98166 │ Continuous │ Float64 │
│ zipcode__98168 │ Continuous │ Float64 │
│ zipcode__98177 │ Continuous │ Float64 │
│ zipcode__98178 │ Continuous │ Float64 │
│ zipcode__98188 │ Continuous │ Float64 │
│ zipcode__98198 │ Continuous │ Float64 │
│ zipcode__98199 │ Continuous │ Float64 │
│ lat            │ Continuous │ Float64 │
│ long           │ Continuous │ Float64 │
│ sqft_living15  │ Continuous │ Float64 │
│ sqft_lot15     │ Continuous │ Float64 │
│ is_renovated   │ Continuous │ Float64 │
└────────────────┴────────────┴─────────┘

````

### More transformers

Here's how to list all of MLJ's unsupervised models:

````@julia
models(m->!m.is_supervised)
````

````
92-element Vector{NamedTuple{(:name, :package_name, :is_supervised, :abstract_type, :constructor, :deep_properties, :docstring, :fit_data_scitype, :human_name, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :inverse_transform_scitype, :is_pure_julia, :is_wrapper, :iteration_parameter, :load_path, :package_license, :package_url, :package_uuid, :predict_scitype, :prediction_type, :reporting_operations, :reports_feature_importances, :supports_class_weights, :supports_online, :supports_training_losses, :supports_weights, :tags, :target_in_fit, :transform_scitype, :input_scitype, :target_scitype, :output_scitype)}}:
 (name = ABODDetector, package_name = OutlierDetectionNeighbors, ... )
 (name = ABODDetector, package_name = OutlierDetectionPython, ... )
 (name = AffinityPropagation, package_name = Clustering, ... )
 (name = AffinityPropagation, package_name = MLJScikitLearnInterface, ... )
 (name = AgglomerativeClustering, package_name = MLJScikitLearnInterface, ... )
 (name = AutoEncoder, package_name = BetaML, ... )
 (name = BM25Transformer, package_name = MLJText, ... )
 (name = Birch, package_name = MLJScikitLearnInterface, ... )
 (name = BisectingKMeans, package_name = MLJScikitLearnInterface, ... )
 (name = BorderlineSMOTE1, package_name = Imbalance, ... )
 (name = CBLOFDetector, package_name = OutlierDetectionPython, ... )
 (name = CDDetector, package_name = OutlierDetectionPython, ... )
 (name = COFDetector, package_name = OutlierDetectionNeighbors, ... )
 (name = COFDetector, package_name = OutlierDetectionPython, ... )
 (name = COPODDetector, package_name = OutlierDetectionPython, ... )
 (name = CardinalityReducer, package_name = MLJTransforms, ... )
 (name = ClusterUndersampler, package_name = Imbalance, ... )
 (name = ContinuousEncoder, package_name = MLJTransforms, ... )
 (name = ContrastEncoder, package_name = MLJTransforms, ... )
 (name = CountTransformer, package_name = MLJText, ... )
 (name = DBSCAN, package_name = Clustering, ... )
 (name = DBSCAN, package_name = MLJScikitLearnInterface, ... )
 (name = DNNDetector, package_name = OutlierDetectionNeighbors, ... )
 (name = ECODDetector, package_name = OutlierDetectionPython, ... )
 (name = ENNUndersampler, package_name = Imbalance, ... )
 (name = FactorAnalysis, package_name = MultivariateStats, ... )
 (name = FeatureAgglomeration, package_name = MLJScikitLearnInterface, ... )
 (name = FeatureSelector, package_name = FeatureSelection, ... )
 (name = FillImputer, package_name = MLJTransforms, ... )
 (name = FrequencyEncoder, package_name = MLJTransforms, ... )
 (name = GMMDetector, package_name = OutlierDetectionPython, ... )
 (name = GaussianMixtureClusterer, package_name = BetaML, ... )
 (name = GaussianMixtureImputer, package_name = BetaML, ... )
 (name = GeneralImputer, package_name = BetaML, ... )
 (name = HBOSDetector, package_name = OutlierDetectionPython, ... )
 (name = HDBSCAN, package_name = MLJScikitLearnInterface, ... )
 (name = HierarchicalClustering, package_name = Clustering, ... )
 (name = ICA, package_name = MultivariateStats, ... )
 (name = IForestDetector, package_name = OutlierDetectionPython, ... )
 (name = INNEDetector, package_name = OutlierDetectionPython, ... )
 (name = InteractionTransformer, package_name = MLJTransforms, ... )
 (name = KDEDetector, package_name = OutlierDetectionPython, ... )
 (name = KMeans, package_name = Clustering, ... )
 (name = KMeans, package_name = MLJScikitLearnInterface, ... )
 (name = KMeans, package_name = ParallelKMeans, ... )
 (name = KMeansClusterer, package_name = BetaML, ... )
 (name = KMedoids, package_name = Clustering, ... )
 (name = KMedoidsClusterer, package_name = BetaML, ... )
 (name = KNNDetector, package_name = OutlierDetectionNeighbors, ... )
 (name = KNNDetector, package_name = OutlierDetectionPython, ... )
 (name = KernelPCA, package_name = MultivariateStats, ... )
 (name = LMDDDetector, package_name = OutlierDetectionPython, ... )
 (name = LOCIDetector, package_name = OutlierDetectionPython, ... )
 (name = LODADetector, package_name = OutlierDetectionPython, ... )
 (name = LOFDetector, package_name = OutlierDetectionNeighbors, ... )
 (name = LOFDetector, package_name = OutlierDetectionPython, ... )
 (name = MCDDetector, package_name = OutlierDetectionPython, ... )
 (name = MeanShift, package_name = MLJScikitLearnInterface, ... )
 (name = MiniBatchKMeans, package_name = MLJScikitLearnInterface, ... )
 (name = MissingnessEncoder, package_name = MLJTransforms, ... )
 (name = OCSVMDetector, package_name = OutlierDetectionPython, ... )
 (name = OPTICS, package_name = MLJScikitLearnInterface, ... )
 (name = OneClassSVM, package_name = LIBSVM, ... )
 (name = OneHotEncoder, package_name = MLJTransforms, ... )
 (name = OrdinalEncoder, package_name = MLJTransforms, ... )
 (name = PCA, package_name = MultivariateStats, ... )
 (name = PCADetector, package_name = OutlierDetectionPython, ... )
 (name = PPCA, package_name = MultivariateStats, ... )
 (name = RODDetector, package_name = OutlierDetectionPython, ... )
 (name = ROSE, package_name = Imbalance, ... )
 (name = RandomForestImputer, package_name = BetaML, ... )
 (name = RandomOversampler, package_name = Imbalance, ... )
 (name = RandomUndersampler, package_name = Imbalance, ... )
 (name = RandomWalkOversampler, package_name = Imbalance, ... )
 (name = SMOTE, package_name = Imbalance, ... )
 (name = SMOTEN, package_name = Imbalance, ... )
 (name = SMOTENC, package_name = Imbalance, ... )
 (name = SODDetector, package_name = OutlierDetectionPython, ... )
 (name = SOSDetector, package_name = OutlierDetectionPython, ... )
 (name = SelfOrganizingMap, package_name = SelfOrganizingMaps, ... )
 (name = SimpleImputer, package_name = BetaML, ... )
 (name = SpectralClustering, package_name = MLJScikitLearnInterface, ... )
 (name = Standardizer, package_name = MLJTransforms, ... )
 (name = TSVDTransformer, package_name = TSVD, ... )
 (name = TargetEncoder, package_name = MLJTransforms, ... )
 (name = TfidfTransformer, package_name = MLJText, ... )
 (name = TomekUndersampler, package_name = Imbalance, ... )
 (name = UnivariateBoxCoxTransformer, package_name = MLJTransforms, ... )
 (name = UnivariateDiscretizer, package_name = MLJTransforms, ... )
 (name = UnivariateFillImputer, package_name = MLJTransforms, ... )
 (name = UnivariateStandardizer, package_name = MLJTransforms, ... )
 (name = UnivariateTimeTypeToContinuous, package_name = MLJTransforms, ... )
````

Some commonly used ones are built-in (do not require `@load`ing):

model type                  | does what?
----------------------------|----------------------------------------------
ContinuousEncoder | transform input table to a table of `Continuous` features (see above)
FeatureSelector | retain or dump selected features
FillImputer | impute missing values
OneHotEncoder | one-hot encoder `Multiclass` (and optionally `OrderedFactor`) features
Standardizer | standardize (whiten) a vector or all `Continuous` features of a table
UnivariateBoxCoxTransformer | apply a learned Box-Cox transformation to a vector
UnivariateDiscretizer | discretize a `Continuous` vector, and hence render its elscitypw `OrderedFactor`

In addition to "dynamic" transformers (ones that learn something
from the data and must be `fit!`) users can wrap ordinary functions
as transformers, and such *static* transformers can depend on
parameters, like the dynamic ones. See
[here](https://juliaai.github.io/MLJ.jl/dev/transformers/#Static-transformers-1)
for how to define your own static transformers.

### Pipelines

````@julia
length(schema(Xcont).names)
````

````
87
````

Let's suppose that additionally we'd like to reduce the dimension of
our data.  A model that will do this is `PCA` from
`MultivariateStats.jl`:

````@julia
PCA = @load PCA
reducer = PCA()
````

````
PCA(
  maxoutdim = 0, 
  method = :auto, 
  variance_ratio = 0.99, 
  mean = nothing)
````

Now, rather simply repeating the work-flow above, applying the new
transformation to `Xcont`, we can combine both the encoding and the
dimension-reducing models into a single model, known as a
*pipeline*. While MLJ offers a powerful interface for composing
models in a variety of ways, we'll stick to these simplest class of
composite models for now. The simplest way to construct a pipeline
is using the Julia's `|>` syntax:

````@julia
pipe = encoder |> reducer
````

````
UnsupervisedPipeline(
  continuous_encoder = ContinuousEncoder(
        drop_last = false, 
        one_hot_ordered_factors = false), 
  pca = PCA(
        maxoutdim = 0, 
        method = :auto, 
        variance_ratio = 0.99, 
        mean = nothing), 
  cache = true)
````

Notice that the model `pipe` has other models as hyperparameters
(with names automatically generated based on the mode type
name). The hyperparameters of the component models are are now
*nested*, but we can still access them:

````@julia
@show pipe.pca.variance_ratio
````

````
0.99
````

````@julia
pipe.pca.variance_ratio = 0.9995
````

````
0.9995
````

The pipeline model behaves like any other transformer:

````@julia
mach = machine(pipe, X)
fit!(mach)
Xsmall = transform(mach, X)
schema(Xsmall)
````

````
┌───────┬────────────┬─────────┐
│ names │ scitypes   │ types   │
├───────┼────────────┼─────────┤
│ x1    │ Continuous │ Float64 │
│ x2    │ Continuous │ Float64 │
│ x3    │ Continuous │ Float64 │
└───────┴────────────┴─────────┘

````

Want to combine this pre-processing with ridge regression?

````@julia
RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels
rgs = RidgeRegressor()
pipe2 = pipe |> rgs
````

````
DeterministicPipeline(
  continuous_encoder = ContinuousEncoder(
        drop_last = false, 
        one_hot_ordered_factors = false), 
  pca = PCA(
        maxoutdim = 0, 
        method = :auto, 
        variance_ratio = 0.9995, 
        mean = nothing), 
  ridge_regressor = RidgeRegressor(
        lambda = 1.0, 
        fit_intercept = true, 
        penalize_intercept = false, 
        scale_penalty_with_samples = true, 
        solver = nothing), 
  cache = true)
````

Now our pipeline is a supervised model, instead of a transformer,
whose performance we can evaluate:

````@julia
mach = machine(pipe2, X, y)
evaluate!(mach, measure=mae, resampling=Holdout()) # `CV(nfolds=6)` is `resampling` default
````

````
PerformanceEvaluation object with these fields:
  model, tag, measure, operation,
  measurement, uncertainty_radius_95, per_fold, per_observation,
  fitted_params_per_fold, report_per_fold,
  train_test_rows, resampling, repeats
Tag: DeterministicPipeline-929
Extract:
┌──────────┬───────────┬─────────────┐
│ measure  │ operation │ measurement │
├──────────┼───────────┼─────────────┤
│ LPLoss(  │ predict   │ 176000.0    │
│   p = 1) │           │             │
└──────────┴───────────┴─────────────┘

````

### Training of composite models is "smart"

Now notice what happens if we train on all the data, then change a
regressor hyperparameter and retrain:

````@julia
fit!(mach);
````

````
[ Info: Training machine(DeterministicPipeline(continuous_encoder = ContinuousEncoder(drop_last = false, …), …), …).
[ Info: Training machine(:continuous_encoder, …).
[ Info: Training machine(:pca, …).
[ Info: Training machine(:ridge_regressor, …).
┌ Info: Solver: MLJLinearModels.Analytical
│   iterative: Bool false
└   max_inner: Int64 200

````

````@julia
pipe2.ridge_regressor.lambda = 0.1
fit!(mach);
````

````
[ Info: Updating machine(DeterministicPipeline(continuous_encoder = ContinuousEncoder(drop_last = false, …), …), …).
[ Info: Not retraining machine(:continuous_encoder, …). Use `force=true` to force.
[ Info: Not retraining machine(:pca, …). Use `force=true` to force.
[ Info: Updating machine(:ridge_regressor, …).
┌ Info: Solver: MLJLinearModels.Analytical
│   iterative: Bool false
└   max_inner: Int64 200

````

Second time only the ridge regressor is retrained!

Mutate a hyperparameter of the `PCA` model and every model except
the `ContinuousEncoder` (which comes before it will be retrained):

````@julia
pipe2.pca.variance_ratio = 0.9999
fit!(mach);
````

````
[ Info: Updating machine(DeterministicPipeline(continuous_encoder = ContinuousEncoder(drop_last = false, …), …), …).
[ Info: Not retraining machine(:continuous_encoder, …). Use `force=true` to force.
[ Info: Updating machine(:pca, …).
[ Info: Training machine(:ridge_regressor, …).
┌ Info: Solver: MLJLinearModels.Analytical
│   iterative: Bool false
└   max_inner: Int64 200

````

### Inspecting composite models

The dot syntax used above to change the values of *nested*
hyperparameters is also useful when inspecting the learned
parameters and report generated when training a composite model:

````@julia
fitted_params(mach).ridge_regressor
````

````
(coefs = [:x1 => -0.7328956348620418, :x2 => -0.1659056319707262, :x3 => 194.59514735965877, :x4 => 102.71298051855302], intercept = 540088.1417665293)
````

````@julia
report(mach).pca
````

````
(indim = 87, outdim = 4, tprincipalvar = 2.463215246230865e9, tresidualvar = 157533.26199674606, tvar = 2.4633727794928617e9, mean = [4.369869985656781, 8.45912182482765, 2079.8997362698374, 15106.967565816869, 1.988617961412113, 1.0075417572757137, 1.2343034284921113, 3.4094295100171195, 6.6569194466293435, 1788.3906907879516, 291.5090454818859, 1971.0051357978994, 0.01674917873502059, 0.009207421459306898, 0.012955165872391617, 0.01466709850552908, 0.007773099523434969, 0.023041687873039375, 0.006523851385740064, 0.013093971221024384, 0.004626844954425577, 0.009022347661129875, 0.005737287743487716, 0.008791005413408597, 0.010826817193355851, 0.02308795632258363, 0.0037477444130847174, 0.01906260121223338, 0.013093971221024384, 0.014852172303706102, 0.011844723083329478, 0.012677555175126082, 0.005783556193031971, 0.019987970203118495, 0.025216305001619397, 0.027298385231110906, 0.0023134224772127887, 0.013047702771480128, 0.025355110350252164, 0.010225327349280526, 0.02655809003840281, 0.018738722065423586, 0.012399944477860548, 0.018784990514967844, 0.021052144542636375, 0.021653634386711702, 0.01434321935871929, 0.005459677046222181, 0.012631286725581826, 0.020404386249016797, 0.016610373386387822, 0.009161153009762642, 0.016240225790033775, 0.0048581872021468565, 0.027853606625641975, 0.010595474945634571, 0.015499930597325684, 0.012307407578772035, 0.008605931615231573, 0.005043261000323879, 0.012446212927404802, 0.026974506084301113, 0.015268588349604404, 0.02558645259797344, 0.02350437236848193, 0.008513394716143062, 0.013417850367834173, 0.018970064313144866, 0.016379031138666542, 0.02285661407486235, 0.012168602230139268, 0.01587007819367973, 0.013325313468745662, 0.002637301624022579, 0.020635728496738073, 0.011752186184240966, 0.012446212927404802, 0.011798454633785222, 0.012122333780595013, 0.006292509138018785, 0.012955165872391617, 0.01466709850552908, 47.560052519317075, -122.21389640494147, 1986.552491556008, 12768.455651691113, 1.9577106371165502], principalvars = [2.1770715510450835e9, 2.841813972643042e8, 1.6850160830642448e6, 277281.83841332135], loadings = [-0.03094830022236769 -0.0028697042773974534 0.5095749351696549 0.14813207201762071; -0.28588792050256073 -0.051089050459670715 2.268889769352154 0.26924666347206383; -171.31152075330877 -44.87769073824155 875.5659689915565 168.66602408773593; -40575.715872989625 8322.862769828476 -3.2832135973351555 0.12584457836092125; 0.007778995904394385 0.01016264682125737 0.4583174913590085 -0.4052534780546669; -0.0022116437335711944 -0.0014772314887840708 0.008030553125353493 0.005029961404332988; -0.06029631790584287 -0.008960249584418841 0.19581563520128384 0.16895626991097012; 0.005151697385622876 -0.0039242463566077895 -0.06655971540979964 0.13542648473251875; -0.14375938327053597 -0.03578556490758178 0.929658283207064 -0.04708045076934042; -163.86359230271813 -42.27617110314842 759.840647517477 -258.1268732347526; -7.4479284505907195 -2.60151963509295 115.72532147407448 426.7928973224915; -1.807573225134045 -1.048498874913949 10.79613973242966 -7.02382617051309; 0.00032880415830767603 0.001260811574209029 -0.0029454763966958648 -0.003678227969477667; 0.0018081489212890885 0.0004172080378575817 -0.004581868056013744 -0.002939472146669705; 0.0015021831533923982 0.00031177511754655596 -0.0019320797726570533 -0.0002596561899147569; 0.0005396372791693265 -0.0008933464547728204 0.014177194128099499 0.0031706817296188856; -0.0011599327900621342 -0.001149620090173445 0.004972673322122869 0.0025203139371222875; 0.0008852219451363137 -0.00044827915357165474 0.0218744769418017 0.01072739667128161; 0.0007603631489364668 0.00022354851848729358 0.0011083720581940798 -0.00017177945017741878; 0.0016539107318789695 0.00032485058081102895 0.0005427567135433357 0.0038757212549929347; -0.005197066654766833 -0.0028999696129265807 -0.000661003487714097 -0.0016313631088372704; 0.0009483368558857951 0.0005124522329180576 0.0025176432648759727 -0.0004027030686681336; -0.012705747793516903 -0.005556238235984618 -0.0024627691630763914 -0.0017728103263237134; -0.0063081202230937765 -0.004629616612497435 9.875656548485024e-5 -0.0033864500235734193; -0.016336236551631553 -0.0028974171250481734 -0.00601235783094024 -0.0033676160018413664; 0.002736632130605568 0.00047782424871933247 -0.0015501314219455071 0.00013252160458356493; -0.008668652483283178 -0.003256324995816642 -0.0006435711877161058 -0.0011877774016417191; -0.009468251570438561 -0.005500626005433613 0.00751962676696177 0.002812613153945314; 0.0011409133177221335 0.0005725313074184388 0.001025452249258631 0.0013828145048178564; 0.0026777974153312863 0.0015871226964076068 0.005756335524300161 -0.007719851104655777; 0.001260248144732329 0.001406971297743995 -0.0006748299461089108 -0.003503868329023825; 0.0012878877889525957 0.001685432607653056 -0.0014589422671145967 -0.0015576793048323763; 0.0007239635803662438 0.00024346172904719001 -0.002409675064843868 0.0007940243060852954; 0.002320308225076744 0.0002356275937591245 0.007624840796732267 -0.0006306379558993107; 0.0036470315314978716 0.001621218188995684 -0.002513931577754477 0.0022896844855860264; -0.007335356565092857 -0.0020449897917740904 0.0034910802878925736 -0.0137413459236454; -0.0002104501319689006 -0.00038582341831445803 0.004505171274721999 -3.927192233145351e-5; 0.0003341834904519865 -0.000563929197760669 0.015379903941725274 0.006595063200323504; -0.002337079811132057 0.00032840854255970124 -0.0015943786174120002 -0.008914392100202817; -0.008765678603957622 0.00023187662391033187 -0.0015192995806335703 -0.003796430858632811; 0.0027305490507038123 0.0005997358990150078 0.011343201263716516 -0.004577608138018163; -0.010376916001404368 -0.004561623593330248 0.012105411259294334 -0.012788460257345712; 0.0018397756388539733 0.00147293526371216 -0.0035169570693630832 -0.0015998490258303671; 0.002921917520325911 0.0017392570344885144 -0.0005149817059620732 -0.0025909215858807635; -0.001965284760130324 0.0015785374825356606 -0.0006069857511605745 -0.002198218812759695; 0.00032033258247909783 -0.00025798020795028394 0.01082421581718729 -0.010753910919246145; -0.00029346026885911254 0.001948253357905583 0.010459732530058908 -0.008559046853871755; -0.012318617611505052 -0.006161809467829821 -0.004132795384379965 -0.0002959567526587337; -0.00518563786806465 -0.0015107896501708894 0.0040265456441314915 -0.0010191161617764501; 0.00013118014979860345 -0.0007879289716323829 0.01597756701191363 -0.007931362675308366; -0.001611376521234072 -7.33427424546316e-5 0.021051854769706675 -0.009419400935426411; -0.009956208196007196 -0.004810863662211462 0.007455511377400376 -0.004342862071439404; -0.008261126915915061 -0.006344276309120046 0.0015564118816394777 -0.0067107794851391; 0.00152668316506974 0.0007354422376225287 0.0004910771824210308 0.001054084874306144; 0.008777127223503491 0.0038805618162345954 -0.013479885793856668 0.0004302952271221392; 0.0028498175899847942 0.0011806267800323646 0.000853458794615733 0.003981869684337886; 0.004016070520843428 0.001783772425610229 -0.011326651734512477 0.0031597957001891; 0.003948858799778394 0.001632380611886704 -0.006892677209652694 0.0009724769062961222; 0.0022922985841207183 0.0009344403206429555 -0.004200692032859478 0.0033692966563264705; 0.001568706313788418 0.0006874861400337716 -0.00010724691357686938 0.0015500117520518383; 0.003382513606730997 0.0013635336536353781 0.006106632924822056 0.004364598499838753; 0.006995381784250757 0.002781678278337619 -0.008938140708671449 0.011020861304471016; 0.00415029865227607 0.0016515208124793282 -0.004769302492821611 0.006341303049227759; 0.007196359156943741 0.0028885154281336994 -0.01254328531796608 0.007229639439179905; 0.00588328598079176 0.0023378298090003828 -0.01121264064866027 0.0066371829664509255; 0.0026381438662044796 0.001105818155618633 -0.000768637027088225 0.002717124311161479; 0.00420854537777756 0.0018166341117659068 -0.00414140388614738 0.0011593383192914147; 0.003726764304357646 0.0012209525165751069 -0.007686277512764649 0.0034348177918449856; 0.004393939480701836 0.0017519372185008065 -0.0107429124910514 0.0028370467607628574; 0.0048836455528599035 0.0017652635795700087 -0.012242459606510561 0.001089447710983446; 0.00297371007280963 0.0011539226050820224 -0.005186313967598817 0.003790685973057083; 0.004743676806900216 0.0021237284763159316 -0.0036046729924441136 0.006333692127822992; 0.0021481035188034036 0.0007241479186689499 -0.0072658862687709925 0.0004436433404973951; 0.0004127307832832469 6.399730208638208e-5 -0.0012817768113535712 -0.0007018097016022439; 0.0025013572656103545 0.00023483865414236684 -0.0073822242438670894 0.0027604332546305077; 0.00033739547227819603 -0.0004701304360303754 -0.0008154621561259409 0.0021060499992515518; 0.0012898963743125237 0.0005466730464324961 -0.008997546594791518 0.0006904779808614117; 0.0008384737928569879 -0.0004514419383084871 0.002848274834533251 0.004722431074377284; 0.0021293453661839793 0.00047721395248930156 -0.0055756914474487185 0.004663496790067101; 0.0007853579603687858 6.195606481725652e-5 -0.002325134592502852 0.0008424639677204342; 0.0015579654177854262 0.00046250964566683624 -0.004963183688635761 0.0006397686382711239; 0.0038054217699936593 0.001509146176030285 0.00048796129399700174 0.009547467619653874; 0.0126240522900567 0.0024616116911250364 0.008168721684362343 0.014083208386218456; -0.035312401805926916 -0.01127856936336475 0.03676354955394956 -0.031109555124335907; -112.88077365642482 -56.8466201296859 572.2199960847387 -1.4956769541523358; -23035.0566009115 -14659.614997914883 -8.975922676497651 0.23006892666563467; 0.0016649390524045606 0.00035640148161713024 -0.00697232854453362 -0.012428541266700475])
````

### Incorporating target transformations

Next, suppose that instead of using the raw `:price` as the training target, we want to
use the log-price (a common practice in dealing with house price data). However, suppose
that we still want to report final *predictions* on the original linear scale (and use
these for evaluation purposes). Then we wrap our supervised model using
`TransformedTargetModel`, which has to key-word arguments `target` and `inverse`.

First we'll overload `log` and `exp` for broadcasting:

````@julia
Base.log(v::AbstractArray) = log.(v)
Base.exp(v::AbstractArray) = exp.(v)
````

Now for the new pipeline:

````@julia
rgs_log = TransformedTargetModel(rgs, transformer=log, inverse=exp)

pipe3 = pipe |> rgs_log
mach = machine(pipe3, X, y)
evaluate!(mach, measure=mae)
````

````
PerformanceEvaluation object with these fields:
  model, tag, measure, operation,
  measurement, uncertainty_radius_95, per_fold, per_observation,
  fitted_params_per_fold, report_per_fold,
  train_test_rows, resampling, repeats
Tag: DeterministicPipeline-884
Extract:
┌──────────┬───────────┬─────────────┐
│ measure  │ operation │ measurement │
├──────────┼───────────┼─────────────┤
│ LPLoss(  │ predict   │ 162000.0    │
│   p = 1) │           │             │
└──────────┴───────────┴─────────────┘
┌──────────────────────────────────────────────────────────────┬─────────┐
│ per_fold                                                     │ 1.96*SE │
├──────────────────────────────────────────────────────────────┼─────────┤
│ [160000.0, 170000.0, 163000.0, 156000.0, 163000.0, 162000.0] │ 4140.0  │
└──────────────────────────────────────────────────────────────┴─────────┘

````

MLJ will also allow you to insert *learned* target transformations. For example, we
might want to apply `Standardizer()` to the target, to standardize it, or
`UnivariateBoxCoxTransformer()` to make it look Gaussian. Then instead of specifying a
*function* for `target`, we specify a unsupervised *model* (or model type). One does not
specify `inverse` because only models implementing `inverse_transform` are allowed.

Let's see which of these two options results in a better outcome:

````@julia
box = UnivariateBoxCoxTransformer(n=20)
stand = Standardizer()

rgs_box = TransformedTargetModel(rgs, transformer=box)
pipe4 = pipe |> rgs_box
mach = machine(pipe4, X, y)
evaluate!(mach, measure=mae)
````

````
PerformanceEvaluation object with these fields:
  model, tag, measure, operation,
  measurement, uncertainty_radius_95, per_fold, per_observation,
  fitted_params_per_fold, report_per_fold,
  train_test_rows, resampling, repeats
Tag: DeterministicPipeline-145
Extract:
┌──────────┬───────────┬─────────────┐
│ measure  │ operation │ measurement │
├──────────┼───────────┼─────────────┤
│ LPLoss(  │ predict   │ 509000.0    │
│   p = 1) │           │             │
└──────────┴───────────┴─────────────┘
┌───────────────────────────────────────────────────────────┬──────────┐
│ per_fold                                                  │ 1.96*SE  │
├───────────────────────────────────────────────────────────┼──────────┤
│ [162000.0, 2.2e6, 181000.0, 161000.0, 176000.0, 172000.0] │ 728000.0 │
└───────────────────────────────────────────────────────────┴──────────┘

````

````@julia
pipe4.transformed_target_model_deterministic.transformer = stand
evaluate!(mach, measure=mae)
````

````
PerformanceEvaluation object with these fields:
  model, tag, measure, operation,
  measurement, uncertainty_radius_95, per_fold, per_observation,
  fitted_params_per_fold, report_per_fold,
  train_test_rows, resampling, repeats
Tag: DeterministicPipeline-287
Extract:
┌──────────┬───────────┬─────────────┐
│ measure  │ operation │ measurement │
├──────────┼───────────┼─────────────┤
│ LPLoss(  │ predict   │ 172000.0    │
│   p = 1) │           │             │
└──────────┴───────────┴─────────────┘
┌──────────────────────────────────────────────────────────────┬─────────┐
│ per_fold                                                     │ 1.96*SE │
├──────────────────────────────────────────────────────────────┼─────────┤
│ [171000.0, 172000.0, 173000.0, 170000.0, 173000.0, 171000.0] │ 1240.0  │
└──────────────────────────────────────────────────────────────┴─────────┘

````

### Tutorial 3 Resources

- From the MLJ manual:
    - [Transformers and other unsupervised models](https://juliaai.github.io/MLJ.jl/dev/transformers/)
    - [Linear pipelines](https://juliaai.github.io/MLJ.jl/dev/linear_pipelines/#Linear-Pipelines)
- From Data Science Tutorials:
    - [Composing models](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/composing-models/)

### Tutorial 3 Exercises

#### Exercise 7

Consider again the Horse Colic classification problem considered in
Exercise 6, but with all features, `Finite` and `Infinite`:

````@julia
import Downloads
import CSV
url = "https://raw.githubusercontent.com/ablaom/"*
    "MachineLearningInJulia2020/"*
    "for-MLJ-version-0.16/data/horse.csv"
csv_file = Downloads.download(url)
horse = CSV.read(csv_file, DataFrames.DataFrame)
coerce!(horse, autotype(horse));
coerce!(horse, Count => Continuous);
coerce!(
    horse,
    :surgery               => Multiclass,
    :age                   => Multiclass,
    :mucous_membranes      => Multiclass,
    :capillary_refill_time => Multiclass,
    :outcome               => Multiclass,
    :cp_data               => Multiclass,
)
schema(X)
````

````
┌───────────────┬───────────────────┬───────────────────────────────────┐
│ names         │ scitypes          │ types                             │
├───────────────┼───────────────────┼───────────────────────────────────┤
│ bedrooms      │ OrderedFactor{13} │ CategoricalValue{Int64, UInt32}   │
│ bathrooms     │ OrderedFactor{30} │ CategoricalValue{Float64, UInt32} │
│ sqft_living   │ Continuous        │ Float64                           │
│ sqft_lot      │ Continuous        │ Float64                           │
│ floors        │ OrderedFactor{6}  │ CategoricalValue{Float64, UInt32} │
│ waterfront    │ OrderedFactor{2}  │ CategoricalValue{Int64, UInt32}   │
│ view          │ OrderedFactor{5}  │ CategoricalValue{Int64, UInt32}   │
│ condition     │ OrderedFactor{5}  │ CategoricalValue{Int64, UInt32}   │
│ grade         │ OrderedFactor{12} │ CategoricalValue{Int64, UInt32}   │
│ sqft_above    │ Continuous        │ Float64                           │
│ sqft_basement │ Continuous        │ Float64                           │
│ yr_built      │ Continuous        │ Float64                           │
│ zipcode       │ Multiclass{70}    │ CategoricalValue{Int64, UInt32}   │
│ lat           │ Continuous        │ Float64                           │
│ long          │ Continuous        │ Float64                           │
│ sqft_living15 │ Continuous        │ Float64                           │
│ sqft_lot15    │ Continuous        │ Float64                           │
│ is_renovated  │ OrderedFactor{2}  │ CategoricalValue{Bool, UInt32}    │
└───────────────┴───────────────────┴───────────────────────────────────┘

````

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
   hyperparameters

(b) Evaluate the pipeline on all data, using 6-fold cross-validation
and `cross_entropy` loss.

(c) Plot a learning curve which examines the effect on this loss
as the tree booster parameter `max_depth` varies from 2 to 10.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

