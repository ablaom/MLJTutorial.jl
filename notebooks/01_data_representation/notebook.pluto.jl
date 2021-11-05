### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# ╔═╡ 24abafc3-0e26-4cd2-9bca-7c69b794f8ce
md"# Machine Learning in Julia"

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

# ╔═╡ a9cee8a6-dcfb-4d97-b1b3-dd47e367ba54
md"## Part 1 - Data Representation"

# ╔═╡ da747378-24c9-4b9c-8d7e-8e449afd1c48
md"""
> **Goals:**
> 1. Learn how MLJ specifies it's data requirements using "scientific" types
> 2. Understand the options for representing tabular data
> 3. Learn how to inspect and fix the representation of data to meet MLJ requirements
"""

# ╔═╡ fe0ead17-c3f5-4f3c-a949-7f3bd292fe31
md"### Scientific types"

# ╔═╡ 4ff58cc5-ea99-4e9d-8514-99938a2932db
md"""
To help you focus on the intended *purpose* or *interpretation* of
data, MLJ models specify data requirements using *scientific types*,
instead of machine types. An example of a scientific type is
`OrderedFactor`. The other basic "scalar" scientific types are
illustrated below:
"""

# ╔═╡ 675753b1-786e-4344-a0de-1721c1bc2df2
md"![](scitypes.png)"

# ╔═╡ 74e167cd-bc94-486c-bca8-7eec7950fd82
md"""
A scientific type is an ordinary Julia type (so it can be used for
method dispatch, for example) but it usually has no instances. The
`scitype` function is used to articulate MLJ's convention about how
different machine types will be interpreted by MLJ models:
"""

# ╔═╡ 45c7e17a-63c8-4fb3-9873-1b1430e635cc
begin
  using ScientificTypes
  scitype(3.141)
end

# ╔═╡ 1a742c64-61e8-4fbf-b43e-d0ebe87da176
begin
  time = [2.3, 4.5, 4.2, 1.8, 7.1]
  scitype(time)
end

# ╔═╡ f4ec6544-bdd3-4c58-9009-4cfb2012998f
md"""
To fix data which MLJ is interpreting incorrectly, we use the
`coerce` method:
"""

# ╔═╡ 6ec50b8f-9c57-4085-abd3-cb77d7a79683
begin
  height = [185, 153, 163, 114, 180]
  scitype(height)
end

# ╔═╡ 8e3fc777-327a-46e3-879e-dc128f3db7b3
height = coerce(height, Continuous)

# ╔═╡ 91abff52-6801-423d-a367-9aa3c6cf34d0
md"""
Here's an example of data we would want interpreted as
`OrderedFactor` but isn't:
"""

# ╔═╡ 8d7b3ba5-a840-4377-bf32-94657e65284e
begin
  exam_mark = ["rotten", "great", "bla",  missing, "great"]
  scitype(exam_mark)
end

# ╔═╡ 7424426a-cdab-4b1e-9afd-65a5b5facac9
exam_mark = coerce(exam_mark, OrderedFactor)

# ╔═╡ dd8cef99-8f55-47a8-b6c8-54f5ed90a964
levels(exam_mark)

# ╔═╡ 41012c9e-8e71-4a9d-9292-fe822525fc77
md"Use `levels!` to put the classes in the right order:"

# ╔═╡ c1c65dab-f82a-400f-ae5e-4b83dcbc9675
begin
  levels!(exam_mark, ["rotten", "bla", "great"])
  exam_mark[1] < exam_mark[2]
end

# ╔═╡ e33a8562-1689-43ad-8a29-32f11452654a
md"When sub-sampling, no levels are lost:"

# ╔═╡ c7135c01-4881-45f6-a5f1-151fcbe229a2
levels(exam_mark[1:2])

# ╔═╡ 0e1afb98-7b11-4eb8-81bc-2d5603785a09
md"""
**Note on binary data.** There is no separate scientific type for
binary data. Binary data is `OrderedFactor{2}` or
`Multiclass{2}`. If a binary measure like `truepositive` is a
applied to `OrderedFactor{2}` then the "positive" class is assumed
to appear *second* in the ordering. If such a measure is applied to
`Multiclass{2}` data, a warning is issued. A single `OrderedFactor`
can be coerced to a single `Continuous` variable, for models that
require this, while a `Multiclass` variable can only be one-hot
encoded.
"""

# ╔═╡ ffcd5bb0-b53e-4ac8-9d87-1d1fbb0e3997
md"### Two-dimensional data"

# ╔═╡ f4a155dc-f3f2-4c44-b950-fddef2a1fb10
md"""
Whenever it makes sense, MLJ Models generally expect two-dimensional
data to be *tabular*. All the tabular formats implementing the
[Tables.jl API](https://juliadata.github.io/Tables.jl/stable/) (see
this
[list](https://github.com/JuliaData/Tables.jl/blob/master/INTEGRATIONS.md))
have a scientific type of `Table` and can be used with such models.
"""

# ╔═╡ 09cd28d1-eabf-488a-951b-64cb2a36c8e3
md"""
Probably the simplest example of a table is the julia native *column
table*, which is just a named tuple of equal-length vectors:
"""

# ╔═╡ 9428e4a9-8e51-4d13-b0e6-441461cc8770
column_table = (h=height, e=exam_mark, t=time)

# ╔═╡ b0d23945-5f6d-489e-8cb0-de601961bc02
scitype(column_table)

# ╔═╡ ed5001c5-e5b9-4a84-a87d-950c50fb2955
md"""
Notice the `Table{K}` type parameter `K` encodes the scientific
types of the columns. (This is useful when comparing table scitypes
with `<:`). To inspect the individual column scitypes, we use the
`schema` method instead:
"""

# ╔═╡ 91c12549-e518-47d3-8448-4bcb089096cd
schema(column_table)

# ╔═╡ 1257966c-8043-49de-a012-a84240254fb6
md"Here are five other examples of tables:"

# ╔═╡ 3b776076-56c6-4215-bbdd-9799f7bb2e60
begin
  dict_table = Dict(:h => height, :e => exam_mark, :t => time)
  schema(dict_table)
end

# ╔═╡ 919fcf78-227b-43c4-97a7-88e3af4f10ee
md"""
(To control column order here, instead use `LittleDict` from
OrderedCollections.jl.)
"""

# ╔═╡ aa98759a-f43d-491e-b372-8c9866e51852
begin
  row_table = [(a=1, b=3.4),
               (a=2, b=4.5),
               (a=3, b=5.6)]
  schema(row_table)
end

# ╔═╡ 74321d51-1831-4822-8f3d-b82f1e7b6f7a
begin
  import DataFrames
  df = DataFrames.DataFrame(column_table)
end

# ╔═╡ 6971b1f8-d607-42ae-ab04-579e5608ae53
schema(df) == schema(column_table)

# ╔═╡ fa257731-0521-4ffe-86cf-35420d9e6995
begin
  using UrlDownload, CSV
  csv_file = urldownload("https://raw.githubusercontent.com/ablaom/"*
                     "MachineLearningInJulia2020/"*
                     "for-MLJ-version-0.16/data/horse.csv");
  schema(csv_file)
end

# ╔═╡ 8134229e-c93c-482a-a29a-8d0445351914
md"""
Most MLJ models do not accept matrix in lieu of a table, but you can
wrap a matrix as a table:
"""

# ╔═╡ 6c7d2c63-9bec-4439-be65-28d0fcca50a8
begin
  using Tables
  matrix_table = Tables.table(rand(2,3))
  schema(matrix_table)
end

# ╔═╡ 5584fdb0-7a69-4f1f-9a2e-c468345d87d1
md"""
The matrix is *not* copied, only wrapped. Some models may perform
better if one wraps the adjoint of the transpose - see
[here](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/#Observations-correspond-to-rows,-not-columns).
"""

# ╔═╡ 7cbaa6ba-e85a-4eba-b5fa-cb79ebf595ef
md"""
**Manipulating tabular data.** In this workshop we assume
familiarity with some kind of tabular data container (although it is
possible, in principle, to carry out the exercises without this.)
For a quick start introduction to `DataFrames`, see [this
tutorial](https://juliaai.github.io/DataScienceTutorials.jl/data/dataframe/).
"""

# ╔═╡ 9518f91d-655c-4e3c-91c5-524da38aa391
md"### Fixing scientific types in tabular data"

# ╔═╡ 03df98f9-b52b-4690-ad90-b4405b216771
md"""
To show how we can correct the scientific types of data in tables,
we introduce a cleaned up version of the UCI Horse Colic Data Set
(the cleaning work-flow is described
[here](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/horse/#dealing_with_missing_values)).
We already downloaded this data set immediately above.q
"""

# ╔═╡ 6dd3ebb2-cb17-4d0a-895b-997a92b731e0
begin
  horse = DataFrames.DataFrame(csv_file); # convert to data frame
  first(horse, 4)
end

# ╔═╡ d11f7c90-acf2-421b-a525-e9b04a4bd246
md"""
From [the UCI
docs](http://archive.ics.uci.edu/ml/datasets/Horse+Colic) we can
surmise how each variable ought to be interpreted (a step in our
work-flow that cannot reliably be left to the computer):
"""

# ╔═╡ 604f6d68-7c31-468c-80f0-ba7781e173cf
md"""
variable                    | scientific type (interpretation)
----------------------------|-----------------------------------
`:surgery`                  | Multiclass
`:age`                      | Multiclass
`:rectal_temperature`       | Continuous
`:pulse`                    | Continuous
`:respiratory_rate`         | Continuous
`:temperature_extremities`  | OrderedFactor
`:mucous_membranes`         | Multiclass
`:capillary_refill_time`    | Multiclass
`:pain`                     | OrderedFactor
`:peristalsis`              | OrderedFactor
`:abdominal_distension`     | OrderedFactor
`:packed_cell_volume`       | Continuous
`:total_protein`            | Continuous
`:outcome`                  | Multiclass
`:surgical_lesion`          | OrderedFactor
`:cp_data`                  | Multiclass
"""

# ╔═╡ ea376156-57e3-43e1-9cbc-9a2c39793333
md"""
Let's see how MLJ will actually interpret the data, as it is
currently encoded:
"""

# ╔═╡ f878b48f-bc60-4e3c-b887-2b16710e5502
schema(horse)

# ╔═╡ 6d08247f-b9b8-456d-9452-5298a8a4a401
md"""
As a first correction step, we can get MLJ to "guess" the
appropriate fix, using the `autotype` method:
"""

# ╔═╡ 181aad5d-e272-49ec-b017-96f1602f2cad
autotype(horse)

# ╔═╡ f8e9f01f-74f6-4612-8be2-2a4017c45345
md"""
Okay, this is not perfect, but a step in the right direction, which
we implement like this:
"""

# ╔═╡ 43498690-c1cb-4dd4-a7ad-80d84f5b0070
begin
  coerce!(horse, autotype(horse));
  schema(horse)
end

# ╔═╡ 22c0ce1a-6a07-4b7a-8378-5aa686f0b407
md"All remaining `Count` data should be `Continuous`:"

# ╔═╡ 69a45801-0a1c-4f47-9f43-49763e8691a1
begin
  coerce!(horse, Count => Continuous);
  schema(horse)
end

# ╔═╡ f4734743-a383-45a2-bb0e-3a45761c733a
md"We'll correct the remaining truant entries manually:"

# ╔═╡ 2e0c0c2b-b011-48f4-9979-7044b2f2df33
begin
  coerce!(horse,
          :surgery               => Multiclass,
          :age                   => Multiclass,
          :mucous_membranes      => Multiclass,
          :capillary_refill_time => Multiclass,
          :outcome               => Multiclass,
          :cp_data               => Multiclass);
  schema(horse)
end

# ╔═╡ 1fc47a9c-b126-489d-b2a1-fbbde543f620
md"""
### Resources for Part 1

- From the MLJ manual:
   - [A preview of data type specification in
  MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/#A-preview-of-data-type-specification-in-MLJ-1)
   - [Data containers and scientific types](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/#Data-containers-and-scientific-types-1)
   - [Working with Categorical Data](https://alan-turing-institute.github.io/MLJ.jl/dev/working_with_categorical_data/)
- [Summary](https://juliaai.github.io/ScientificTypes.jl/dev/#Summary-of-the-default-convention) of the MLJ convention for representing scientific types
- [ScientificTypes.jl](https://juliaai.github.io/ScientificTypes.jl/dev/)
- From Data Science Tutorials:
    - [Data interpretation: Scientific Types](https://juliaai.github.io/DataScienceTutorials.jl/data/scitype/)
    - [Horse colic data](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/horse/)
- [UCI Horse Colic Data Set](http://archive.ics.uci.edu/ml/datasets/Horse+Colic)
"""

# ╔═╡ c9ac414c-ca43-460f-910b-f8a9a217eff2
md"### Exercises for Part 1"

# ╔═╡ 84a75325-e427-45c5-aa36-d2e8546da46a
md"#### Exercise 1"

# ╔═╡ 1c47cc40-33f8-4552-88a1-a2bf91434413
md"Try to guess how each code snippet below will evaluate:"

# ╔═╡ 4da068ed-3935-4728-a1cb-54e34396a855
scitype(42)

# ╔═╡ 04c88aba-6c17-48eb-8037-0de5806e1a54
begin
  questions = ["who", "why", "what", "when"]
  scitype(questions)
end

# ╔═╡ dc8c4bd5-e753-4b7b-9961-a09632c33fb0
elscitype(questions)

# ╔═╡ 34c94131-ef2a-4f86-b7cc-46f4ef988c68
begin
  t = (3.141, 42, "how")
  scitype(t)
end

# ╔═╡ 8f5681ef-5432-4daf-90fb-7f2721f6fcc7
A = rand(2, 3)

# ╔═╡ 971d3226-0c9b-4c3b-af65-ccd25ecb9818
md"-"

# ╔═╡ 281afa2b-c539-4c61-8890-979691212d9b
scitype(A)

# ╔═╡ 67054d46-550d-40e6-a6fa-eaf14df5d44b
elscitype(A)

# ╔═╡ ec84d270-fcb9-4b4b-8025-5084804a9f6c
begin
  using SparseArrays
  Asparse = sparse(A)
end

# ╔═╡ d3c65fbf-1a6c-4779-9e90-09023d201062
scitype(Asparse)

# ╔═╡ 1c530c2e-f0b8-4e30-b7bb-2f33ef765cc0
C = coerce(A, Multiclass)

# ╔═╡ 09b804b7-30f7-4831-9625-01172c4a0081
scitype(C)

# ╔═╡ 1e774630-8eac-4394-af4f-11c7de9e21dd
elscitype(C)

# ╔═╡ 347d35af-60d3-4422-8dba-241d9b744683
begin
  v = [1, 2, missing, 4]
  scitype(v)
end

# ╔═╡ ed2aa555-53d0-4f61-a6e5-2f0b4dca5c59
elscitype(v)

# ╔═╡ 28a31668-333e-434a-8550-20498aa03ed0
scitype(v[1:2])

# ╔═╡ 2cc4182e-ffaa-47a6-9e7b-5b943cf6b560
md"""
Can you guess at the general behavior of
`scitype` with respect to tuples, abstract arrays and missing
values? The answers are
[here](https://github.com/juliaai/ScientificTypesBase.jl#2-the-scitype-and-scitype-methods)
(ignore "Property 1").
"""

# ╔═╡ 34a20115-36f5-47d5-bce6-0a1379cc1259
md"#### Exercise 2"

# ╔═╡ b5866134-32b9-4ad9-9608-b5032c116833
md"""
Coerce the following vector to make MLJ recognize it as a vector of
ordered factors (with an appropriate ordering):
"""

# ╔═╡ 7f9003b5-95c7-418b-b473-4aa968e6937a
quality = ["good", "poor", "poor", "excellent", missing, "good", "excellent"]

# ╔═╡ 58ed39f5-3c80-460d-8d9e-644a1b3cc6b6
md"#### Exercise 3 (fixing scitypes in a table)"

# ╔═╡ c8162baa-74b7-4046-ac0a-23ded81445da
md"""
Fix the scitypes for the [House Prices in King
County](https://mlr3gallery.mlr-org.com/posts/2020-01-30-house-prices-in-king-county/)
dataset:
"""

# ╔═╡ 2e797d14-c00b-4166-8535-1a088a6a3228
begin
  house_csv = urldownload("https://raw.githubusercontent.com/ablaom/"*
                          "MachineLearningInJulia2020/for-MLJ-version-0.16/"*
                          "data/house.csv");
  house = DataFrames.DataFrame(house_csv)
  first(house, 4)
end

# ╔═╡ 706005cd-a3e5-4b5b-a39f-7893c73eef39
md"""
(Two features in the original data set have been deemed uninformative
and dropped, namely `:id` and `:date`. The original feature
`:yr_renovated` has been replaced by the `Bool` feature `is_renovated`.)
"""

# ╔═╡ 7d15cc45-23a6-4a3d-bcca-51a27994a151
md"<a id='part-2-selecting-training-and-evaluating-models'></a>"

# ╔═╡ 135dac9b-0bd9-4e1d-9b33-09b4b6661170
md"""
---

*This notebook was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
"""

# ╔═╡ Cell order:
# ╟─24abafc3-0e26-4cd2-9bca-7c69b794f8ce
# ╟─b67c62c2-15ab-4328-b795-033f6f2a0674
# ╟─bc689638-fd19-4c9f-935f-ddf6a6bfbbdd
# ╟─197fd00e-9068-46ea-af2a-25235e544a31
# ╠═f6d4f8c4-e441-45c4-8af5-148d95ea2900
# ╟─45740c4d-b789-45dc-a6bf-47194d7e8e12
# ╟─42b0f1e1-16c9-4238-828a-4cc485149963
# ╠═d09256dd-6c0d-4e28-9e54-3f7b3ca87ecb
# ╟─499cbc31-83ba-4583-ba1f-6363f43ec697
# ╟─db5821e8-956a-4a46-95ea-2955abd45275
# ╟─a9cee8a6-dcfb-4d97-b1b3-dd47e367ba54
# ╟─da747378-24c9-4b9c-8d7e-8e449afd1c48
# ╟─fe0ead17-c3f5-4f3c-a949-7f3bd292fe31
# ╟─4ff58cc5-ea99-4e9d-8514-99938a2932db
# ╟─675753b1-786e-4344-a0de-1721c1bc2df2
# ╟─74e167cd-bc94-486c-bca8-7eec7950fd82
# ╠═45c7e17a-63c8-4fb3-9873-1b1430e635cc
# ╠═1a742c64-61e8-4fbf-b43e-d0ebe87da176
# ╟─f4ec6544-bdd3-4c58-9009-4cfb2012998f
# ╠═6ec50b8f-9c57-4085-abd3-cb77d7a79683
# ╠═8e3fc777-327a-46e3-879e-dc128f3db7b3
# ╟─91abff52-6801-423d-a367-9aa3c6cf34d0
# ╠═8d7b3ba5-a840-4377-bf32-94657e65284e
# ╠═7424426a-cdab-4b1e-9afd-65a5b5facac9
# ╠═dd8cef99-8f55-47a8-b6c8-54f5ed90a964
# ╟─41012c9e-8e71-4a9d-9292-fe822525fc77
# ╠═c1c65dab-f82a-400f-ae5e-4b83dcbc9675
# ╟─e33a8562-1689-43ad-8a29-32f11452654a
# ╠═c7135c01-4881-45f6-a5f1-151fcbe229a2
# ╟─0e1afb98-7b11-4eb8-81bc-2d5603785a09
# ╟─ffcd5bb0-b53e-4ac8-9d87-1d1fbb0e3997
# ╟─f4a155dc-f3f2-4c44-b950-fddef2a1fb10
# ╟─09cd28d1-eabf-488a-951b-64cb2a36c8e3
# ╠═9428e4a9-8e51-4d13-b0e6-441461cc8770
# ╠═b0d23945-5f6d-489e-8cb0-de601961bc02
# ╟─ed5001c5-e5b9-4a84-a87d-950c50fb2955
# ╠═91c12549-e518-47d3-8448-4bcb089096cd
# ╟─1257966c-8043-49de-a012-a84240254fb6
# ╠═3b776076-56c6-4215-bbdd-9799f7bb2e60
# ╟─919fcf78-227b-43c4-97a7-88e3af4f10ee
# ╠═aa98759a-f43d-491e-b372-8c9866e51852
# ╠═74321d51-1831-4822-8f3d-b82f1e7b6f7a
# ╠═6971b1f8-d607-42ae-ab04-579e5608ae53
# ╠═fa257731-0521-4ffe-86cf-35420d9e6995
# ╟─8134229e-c93c-482a-a29a-8d0445351914
# ╠═6c7d2c63-9bec-4439-be65-28d0fcca50a8
# ╟─5584fdb0-7a69-4f1f-9a2e-c468345d87d1
# ╟─7cbaa6ba-e85a-4eba-b5fa-cb79ebf595ef
# ╟─9518f91d-655c-4e3c-91c5-524da38aa391
# ╟─03df98f9-b52b-4690-ad90-b4405b216771
# ╠═6dd3ebb2-cb17-4d0a-895b-997a92b731e0
# ╟─d11f7c90-acf2-421b-a525-e9b04a4bd246
# ╟─604f6d68-7c31-468c-80f0-ba7781e173cf
# ╟─ea376156-57e3-43e1-9cbc-9a2c39793333
# ╠═f878b48f-bc60-4e3c-b887-2b16710e5502
# ╟─6d08247f-b9b8-456d-9452-5298a8a4a401
# ╠═181aad5d-e272-49ec-b017-96f1602f2cad
# ╟─f8e9f01f-74f6-4612-8be2-2a4017c45345
# ╠═43498690-c1cb-4dd4-a7ad-80d84f5b0070
# ╟─22c0ce1a-6a07-4b7a-8378-5aa686f0b407
# ╠═69a45801-0a1c-4f47-9f43-49763e8691a1
# ╟─f4734743-a383-45a2-bb0e-3a45761c733a
# ╠═2e0c0c2b-b011-48f4-9979-7044b2f2df33
# ╟─1fc47a9c-b126-489d-b2a1-fbbde543f620
# ╟─c9ac414c-ca43-460f-910b-f8a9a217eff2
# ╟─84a75325-e427-45c5-aa36-d2e8546da46a
# ╟─1c47cc40-33f8-4552-88a1-a2bf91434413
# ╠═4da068ed-3935-4728-a1cb-54e34396a855
# ╠═04c88aba-6c17-48eb-8037-0de5806e1a54
# ╠═dc8c4bd5-e753-4b7b-9961-a09632c33fb0
# ╠═34c94131-ef2a-4f86-b7cc-46f4ef988c68
# ╠═8f5681ef-5432-4daf-90fb-7f2721f6fcc7
# ╟─971d3226-0c9b-4c3b-af65-ccd25ecb9818
# ╠═281afa2b-c539-4c61-8890-979691212d9b
# ╠═67054d46-550d-40e6-a6fa-eaf14df5d44b
# ╠═ec84d270-fcb9-4b4b-8025-5084804a9f6c
# ╠═d3c65fbf-1a6c-4779-9e90-09023d201062
# ╠═1c530c2e-f0b8-4e30-b7bb-2f33ef765cc0
# ╠═09b804b7-30f7-4831-9625-01172c4a0081
# ╠═1e774630-8eac-4394-af4f-11c7de9e21dd
# ╠═347d35af-60d3-4422-8dba-241d9b744683
# ╠═ed2aa555-53d0-4f61-a6e5-2f0b4dca5c59
# ╠═28a31668-333e-434a-8550-20498aa03ed0
# ╟─2cc4182e-ffaa-47a6-9e7b-5b943cf6b560
# ╟─34a20115-36f5-47d5-bce6-0a1379cc1259
# ╟─b5866134-32b9-4ad9-9608-b5032c116833
# ╠═7f9003b5-95c7-418b-b473-4aa968e6937a
# ╟─58ed39f5-3c80-460d-8d9e-644a1b3cc6b6
# ╟─c8162baa-74b7-4046-ac0a-23ded81445da
# ╠═2e797d14-c00b-4166-8535-1a088a6a3228
# ╟─706005cd-a3e5-4b5b-a39f-7893c73eef39
# ╟─7d15cc45-23a6-4a3d-bcca-51a27994a151
# ╟─135dac9b-0bd9-4e1d-9b33-09b4b6661170
