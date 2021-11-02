# # Machine Learning in Julia (conclusion)

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


# ## Solutions to exercises

using MLJ, UrlDownload, CSV, DataFrames, Plots

# #### Exercise 2 solution

# From the question statememt:

quality = ["good", "poor", "poor", "excellent", missing, "good", "excellent"]

#-

quality = coerce(quality, OrderedFactor);
levels!(quality, ["poor", "good", "excellent"]);
elscitype(quality)


# #### Exercise 3 solution

# From the question statement:

house_csv = urldownload("https://raw.githubusercontent.com/ablaom/"*
                        "MachineLearningInJulia2020/for-MLJ-version-0.16/"*
                        "data/house.csv");
house = DataFrames.DataFrame(house_csv)

# First pass:

coerce!(house, autotype(house));
schema(house)

#-

# All the "sqft" fields refer to "square feet" so are
# really `Continuous`. We'll regard `:yr_built` (the other `Count`
# variable above) as `Continuous` as well. So:

coerce!(house, Count => Continuous);

# And `:zipcode` should not be ordered:

coerce!(house, :zipcode => Multiclass);
schema(house)

# `:bathrooms` looks like it has a lot of levels, but on further
# inspection we see why, and `OrderedFactor` remains appropriate:

import StatsBase.countmap
countmap(house.bathrooms)


# #### Exercise 4 solution

# From the question statement:

import Distributions
poisson = Distributions.Poisson

age = 18 .+ 60*rand(10);
salary = coerce(rand(["small", "big", "huge"], 10), OrderedFactor);
levels!(salary, ["small", "big", "huge"]);
small = salary[1]

X4 = DataFrames.DataFrame(age=age, salary=salary)

n_devices(salary) = salary > small ? rand(poisson(1.3)) : rand(poisson(2.9))
y4 = [n_devices(row.salary) for row in eachrow(X4)]

#-

# 4(a)

# There are *no* models that apply immediately:

models(matching(X4, y4))

# 4(b)

y4 = coerce(y4, Continuous);
models(matching(X4, y4))


# #### Exercise 6 solution

# From the question statement:

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

# 6(a)

y, X = unpack(horse,
              ==(:outcome),
              name -> elscitype(Tables.getcolumn(horse, name)) == Continuous);

# 6(b)(i)

train, test = partition(eachindex(y), 0.7)
model = (@load LogisticClassifier pkg=MLJLinearModels)();
model.lambda = 100
mach = machine(model, X, y)
fit!(mach, rows=train)
fitted_params(mach)

#-

coefs_given_feature = Dict(fitted_params(mach).coefs)
coefs_given_feature[:pulse]

#6(b)(ii)

yhat = predict(mach, rows=test); # or predict(mach, X[test,:])
err = cross_entropy(yhat, y[test]) |> mean

# 6(b)(iii)

# The predicted probabilities of the actual observations in the test
# are given by

p = broadcast(pdf, yhat, y[test]);

# The number of times this probability exceeds 50% is:
n50 = filter(x -> x > 0.5, p) |> length

# Or, as a proportion:

n50/length(test)

# 6(b)(iv)

misclassification_rate(mode.(yhat), y[test])

# 6(c)(i)

model = (@load RandomForestClassifier pkg=DecisionTree)()
mach = machine(model, X, y)
evaluate!(mach, resampling=CV(nfolds=6), measure=cross_entropy)

r = range(model, :n_trees, lower=10, upper=70, scale=:log10)

# Since random forests are inherently randomized, we generate multiple
# curves:

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
plt #!md

# ![](exercise_6ci.png) #md


# 6(c)(ii)

evaluate!(mach, resampling=CV(nfolds=9),
                measure=cross_entropy,
                rows=train).measurement[1]

model.n_trees = 90

# 6(c)(iii)

err_forest = evaluate!(mach, resampling=Holdout(),
                       measure=cross_entropy).measurement[1]

# #### Exercise 7

# (a)

KMeans = @load KMeans pkg=Clustering
EvoTreeClassifier = @load EvoTreeClassifier
pipe = @pipeline(Standardizer,
                 ContinuousEncoder,
                 KMeans(k=10),
                 EvoTreeClassifier(nrounds=50))

# (b)

mach = machine(pipe, X, y)
evaluate!(mach, resampling=CV(nfolds=6), measure=cross_entropy)

# (c)

r = range(pipe, :(evo_tree_classifier.max_depth), lower=1, upper=10)

curve = learning_curve(mach,
                       range=r,
                       resampling=CV(nfolds=6),
                       measure=cross_entropy)

plt = plot(curve.parameter_values, curve.measurements)
xlabel!(plt, "max_depth")
ylabel!(plt, "CV estimate of cross entropy")
savefig("exercise_7c.png")
plt #!md

# ![](exercise_7c.png) #md

# Here's a second curve using a different random seed for the booster:

using Random
pipe.evo_tree_classifier.rng = MersenneTwister(123)
curve = learning_curve(mach,
                       range=r,
                       resampling=CV(nfolds=6),
                       measure=cross_entropy)
plot!(curve.parameter_values, curve.measurements)
savefig("exercise_7c_2.png")
plt #!md

# ![](exercise_7c_2.png) #md

# One can automate the production of multiple curves with different
# seeds in the following way:
curves = learning_curve(mach,
                        range=r,
                        resampling=CV(nfolds=6),
                        measure=cross_entropy,
                        rng_name=:(evo_tree_classifier.rng),
                        rngs=6) # list of RNGs, or num to auto generate
plt = plot(curves.parameter_values, curves.measurements)
savefig("exercise_7c_3.png")
plt #!md

# ![](exercise_7c_3.png) #md

# If you have multiple threads available in your julia session, you
# can add the option `acceleration=CPUThreads()` to speed up this
# computation.

# #### Exercise 8

# From the question statement:

y, X = unpack(house, ==(:price), name -> true, rng=123); # from Exercise 3

EvoTreeRegressor = @load EvoTreeRegressor
tree_booster = EvoTreeRegressor(nrounds = 70)
model = @pipeline ContinuousEncoder tree_booster

r2 = range(model,
           :(evo_tree_regressor.nbins),
           lower = 2.5,
           upper= 7.5, scale=x->2^round(Int, x))

# (a)

r1 = range(model, :(evo_tree_regressor.max_depth), lower=1, upper=12)

# (c)

tuned_model = TunedModel(model=model,
                         ranges=[r1, r2],
                         resampling=Holdout(),
                         measures=mae,
                         tuning=RandomSearch(rng=123),
                         n=40)

tuned_mach = machine(tuned_model, X, y) |> fit!
plt = plot(tuned_mach)
savefig("exercise_8c.png")
plt #!md

# ![](exercise_8c.png) #md

# (d)

best_model = report(tuned_mach).best_model;
best_mach = machine(best_model, X, y);
best_err = evaluate!(best_mach, resampling=CV(nfolds=3), measure=mae)

#-

tuned_err = evaluate!(tuned_mach, resampling=CV(nfolds=3), measure=mae)
