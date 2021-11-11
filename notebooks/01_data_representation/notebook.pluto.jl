### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# ╔═╡ d09256dd-6c0d-4e28-9e54-3f7b3ca87ecb
begin
  	using ScientificTypes
	import DataFrames
	using UrlDownload, CSV
	using Tables
	using SparseArrays
	using PlutoUI
end

# ╔═╡ 7e85780a-bd7e-4b5e-88ff-6053d64456f5
html"""
<div style="
position: absolute;
width: calc(100% - 30px);
border: 50vw solid SteelBlue;
border-top: 500px solid SteelBlue;
border-bottom: none;
box-sizing: content-box;
left: calc(-50vw + 15px);
top: -500px;
height: 300px;
pointer-events: none;
"></div>

<div style="
height: 300px;
width: 100%;
background: SteelBlue;
color: #88BBD6;
padding-top: 68px;
padding-left: 5px;
">

<span style="
font-family: Vollkorn, serif;
font-weight: 700;
font-feature-settings: 'lnum', 'pnum';
"> 

<p style="
font-family: Alegreya sans;
font-size: 1.4rem;
font-weight: 300;
opacity: 1.0;
color: #CDCDCD;
">Tutorial - Part 1</p>
<p style="text-align: left; font-size: 2.5rem;">
Machine Learning in Julia
</p>
"""

# ╔═╡ d9536b16-bb2a-4f83-9666-9b9eeadc125b
PlutoUI.TableOfContents(title = "MLJ Tutorial - Part 1")

# ╔═╡ b67c62c2-15ab-4328-b795-033f6f2a0674
md"""
An introduction to the
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/stable/)
toolbox.
"""

# ╔═╡ bc689638-fd19-4c9f-935f-ddf6a6bfbbdd
md"## Set-up"

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

# ╔═╡ db5821e8-956a-4a46-95ea-2955abd45275
md"""
- [MLJ Cheatsheet](https://alan-turing-institute.github.io/MLJ.jl/dev/mlj_cheatsheet/)
- [Common MLJ Workflows](https://alan-turing-institute.github.io/MLJ.jl/dev/common_mlj_workflows/)
- [MLJ manual](https://alan-turing-institute.github.io/MLJ.jl/dev/)
- [Data Science Tutorials in Julia](https://juliaai.github.io/DataScienceTutorials.jl/)
"""

# ╔═╡ a9cee8a6-dcfb-4d97-b1b3-dd47e367ba54
md"# Part 1 - Data Representation"

# ╔═╡ da747378-24c9-4b9c-8d7e-8e449afd1c48
md"""
> **Goals:**
> 1. Learn how MLJ specifies it's data requirements using "scientific" types
> 2. Understand the options for representing tabular data
> 3. Learn how to inspect and fix the representation of data to meet MLJ requirements
"""

# ╔═╡ fe0ead17-c3f5-4f3c-a949-7f3bd292fe31
md"## Scientific types"

# ╔═╡ 4ff58cc5-ea99-4e9d-8514-99938a2932db
md"""
To help you focus on the intended *purpose* or *interpretation* of
data, MLJ models specify data requirements using *scientific types*,
instead of machine types. An example of a scientific type is
`OrderedFactor`. The other basic "scalar" scientific types are
illustrated below:
"""

# ╔═╡ 675753b1-786e-4344-a0de-1721c1bc2df2
html"""
<div style="text-align: left";>
	<img src="https://github.com/ablaom/MLJTutorial.jl/blob/dev/notebooks/01_data_representation/scitypes.png?raw=true"
</div>
"""

# ╔═╡ 74e167cd-bc94-486c-bca8-7eec7950fd82
md"""
A scientific type is an ordinary Julia type (so it can be used for
method dispatch, for example) but it usually has no instances. The
`scitype` function is used to articulate MLJ's convention about how
different machine types will be interpreted by MLJ models:
"""

# ╔═╡ 45c7e17a-63c8-4fb3-9873-1b1430e635cc
scitype(3.141)

# ╔═╡ 1a742c64-61e8-4fbf-b43e-d0ebe87da176
time = [2.3, 4.5, 4.2, 1.8, 7.1]

# ╔═╡ 44737448-20e7-464b-bb15-33279dbe6c3a
scitype(time)

# ╔═╡ f4ec6544-bdd3-4c58-9009-4cfb2012998f
md"""
To fix data which MLJ is interpreting incorrectly, we use the
`coerce` method:
"""

# ╔═╡ 5e3fc1a4-2e74-48c6-9864-152c603930ec
height0 = [185, 153, 163, 114, 180]

# ╔═╡ 6ec50b8f-9c57-4085-abd3-cb77d7a79683
scitype(height0)

# ╔═╡ 8e3fc777-327a-46e3-879e-dc128f3db7b3
height = coerce(height0, Continuous)

# ╔═╡ 91abff52-6801-423d-a367-9aa3c6cf34d0
md"""
Here's an example of data we would want interpreted as
`OrderedFactor` but isn't:
"""

# ╔═╡ 8d7b3ba5-a840-4377-bf32-94657e65284e
exam_mark0 = ["rotten", "great", "bla",  missing, "great"]

# ╔═╡ 7482fffc-7b40-4e84-9589-873d4bc96cca
scitype(exam_mark0)

# ╔═╡ 7424426a-cdab-4b1e-9afd-65a5b5facac9
exam_mark = coerce(exam_mark0, OrderedFactor)

# ╔═╡ dd8cef99-8f55-47a8-b6c8-54f5ed90a964
levels(exam_mark)

# ╔═╡ 41012c9e-8e71-4a9d-9292-fe822525fc77
md"Use `levels!` to put the classes in the right order:"

# ╔═╡ c1c65dab-f82a-400f-ae5e-4b83dcbc9675
levels!(exam_mark, ["rotten", "bla", "great"])

# ╔═╡ 0033787e-0796-41c6-a81d-fc52d08ea8e0
exam_mark[1] < exam_mark[2]

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
md"## Two-dimensional data"

# ╔═╡ f4a155dc-f3f2-4c44-b950-fddef2a1fb10
md"""
Whenever it makes sense, MLJ Models generally expect two-dimensional
data to be *tabular*. All the tabular formats implementing the
[Tables.jl API](https://juliadata.github.io/Tables.jl/stable/) (see
this
[list](https://github.com/JuliaData/Tables.jl/blob/master/INTEGRATIONS.md))
have a scientific type of `Table` and can be used with such models.
"""

# ╔═╡ 308f453a-767e-4742-bab9-13a4b5e22567
md"""
### Column table
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
sc = schema(column_table)

# ╔═╡ 62be5b63-a6a3-48b3-b23d-a60990b06fec
md"""
Specific information about a schema can be obtained by accessing the attributes `names`, `types` and `scitypes` directly:
"""

# ╔═╡ 4eaac509-d637-41bf-85b6-a8154be0f959
sc.names

# ╔═╡ fef11f2a-da56-430d-b56d-ac411bbca555
sc.types

# ╔═╡ c37d5a96-29a7-465e-9ed4-cd7542e9065a
sc.scitypes

# ╔═╡ 1257966c-8043-49de-a012-a84240254fb6
md"Here are five other examples of tables:"

# ╔═╡ 27415a96-42ec-4dc9-9428-713a24ceeda9
md"""
### Dictionary
"""

# ╔═╡ 3b776076-56c6-4215-bbdd-9799f7bb2e60
dict_table = Dict(:h => height, :e => exam_mark, :t => time)

# ╔═╡ 0cf291d4-4e7d-4b28-8562-56d54658ca00
schema(dict_table)

# ╔═╡ 919fcf78-227b-43c4-97a7-88e3af4f10ee
md"""
(To control column order here, instead use `LittleDict` from
OrderedCollections.jl.)
"""

# ╔═╡ e72b4214-b18f-40ba-a0c0-a78f67d659a0
md"""
### Row Table
"""

# ╔═╡ aa98759a-f43d-491e-b372-8c9866e51852
row_table = [(a=1, b=3.4),
             (a=2, b=4.5),
             (a=3, b=5.6)]

# ╔═╡ cfb975ac-d946-4204-8427-bccb0a0e1ae7
schema(row_table)

# ╔═╡ eba6d196-f627-49be-8f80-51fef8c5aa59
md"""
### DataFrame
"""

# ╔═╡ 74321d51-1831-4822-8f3d-b82f1e7b6f7a
df = DataFrames.DataFrame(column_table)

# ╔═╡ 6971b1f8-d607-42ae-ab04-579e5608ae53
schema(df) == schema(column_table)

# ╔═╡ ee1c9b0f-62f5-4dfb-89b8-6a303aca22b1
md"""
### CSV File
"""

# ╔═╡ fa257731-0521-4ffe-86cf-35420d9e6995
csv_file = urldownload("https://raw.githubusercontent.com/ablaom/"*
                     "MachineLearningInJulia2020/"*
                     "for-MLJ-version-0.16/data/horse.csv")

# ╔═╡ e6f680e3-3dd7-421c-bb76-0ab026336d20
 schema(csv_file)

# ╔═╡ 75f109af-7651-477a-a6f7-2e3eb635a58d
md"""
### Matrix
"""

# ╔═╡ 8134229e-c93c-482a-a29a-8d0445351914
md"""
Most MLJ models do not accept matrix in lieu of a table, but you can
wrap a matrix as a table:
"""

# ╔═╡ 6c7d2c63-9bec-4439-be65-28d0fcca50a8
matrix_table = Tables.table(rand(2,3))

# ╔═╡ ac973148-6168-473a-adb7-3880e5bd45b6
schema(matrix_table)

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
md"## Fixing scientific types in tabular data"

# ╔═╡ 03df98f9-b52b-4690-ad90-b4405b216771
md"""
To show how we can correct the scientific types of data in tables,
we introduce a cleaned up version of the UCI Horse Colic Data Set
(the cleaning work-flow is described
[here](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/horse/#dealing_with_missing_values)).
We already downloaded this data set immediately above.q
"""

# ╔═╡ 6dd3ebb2-cb17-4d0a-895b-997a92b731e0
horse = DataFrames.DataFrame(csv_file) # convert to data frame

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
# Resources for Part 1

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
md"# Exercises for Part 1"

# ╔═╡ 84a75325-e427-45c5-aa36-d2e8546da46a
md"## Exercise 1"

# ╔═╡ 1c47cc40-33f8-4552-88a1-a2bf91434413
md"""
Try to guess how each code snippet below will evaluate (uncomment to see the solutions):
"""

# ╔═╡ 4da068ed-3935-4728-a1cb-54e34396a855
#scitype(42)

# ╔═╡ 04c88aba-6c17-48eb-8037-0de5806e1a54
questions = ["who", "why", "what", "when"]

# ╔═╡ e5c60366-e03c-490f-9475-e53e5b3347d4
#scitype(questions)

# ╔═╡ dc8c4bd5-e753-4b7b-9961-a09632c33fb0
#elscitype(questions)

# ╔═╡ 34c94131-ef2a-4f86-b7cc-46f4ef988c68
t = (3.141, 42, "how")

# ╔═╡ a9238f1a-2751-46a8-9d69-62669915ed91
#scitype(t)

# ╔═╡ 8f5681ef-5432-4daf-90fb-7f2721f6fcc7
A = rand(2, 3)

# ╔═╡ 281afa2b-c539-4c61-8890-979691212d9b
#scitype(A)

# ╔═╡ 67054d46-550d-40e6-a6fa-eaf14df5d44b
#elscitype(A)

# ╔═╡ ec84d270-fcb9-4b4b-8025-5084804a9f6c
Asparse = sparse(A)

# ╔═╡ d3c65fbf-1a6c-4779-9e90-09023d201062
#scitype(Asparse)

# ╔═╡ 1c530c2e-f0b8-4e30-b7bb-2f33ef765cc0
C = coerce(A, Multiclass)

# ╔═╡ 09b804b7-30f7-4831-9625-01172c4a0081
#scitype(C)

# ╔═╡ 1e774630-8eac-4394-af4f-11c7de9e21dd
#elscitype(C)

# ╔═╡ 347d35af-60d3-4422-8dba-241d9b744683
v = [1, 2, missing, 4]

# ╔═╡ 3aac62ed-9e28-4db9-a172-d57784b35656
#scitype(v)

# ╔═╡ ed2aa555-53d0-4f61-a6e5-2f0b4dca5c59
#elscitype(v)

# ╔═╡ 28a31668-333e-434a-8550-20498aa03ed0
#scitype(v[1:2])

# ╔═╡ 2cc4182e-ffaa-47a6-9e7b-5b943cf6b560
md"""
Can you guess at the general behavior of
`scitype` with respect to tuples, abstract arrays and missing
values? The answers are
[here](https://github.com/juliaai/ScientificTypesBase.jl#2-the-scitype-and-scitype-methods)
(ignore "Property 1").
"""

# ╔═╡ 34a20115-36f5-47d5-bce6-0a1379cc1259
md"## Exercise 2"

# ╔═╡ b5866134-32b9-4ad9-9608-b5032c116833
md"""
Coerce the following vector to make MLJ recognize it as a vector of
ordered factors (with an appropriate ordering):
"""

# ╔═╡ 7f9003b5-95c7-418b-b473-4aa968e6937a
quality0 = ["good", "poor", "poor", "excellent", missing, "good", "excellent"]

# ╔═╡ 58ed39f5-3c80-460d-8d9e-644a1b3cc6b6
md"## Exercise 3 (fixing scitypes in a table)"

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
html"""
<a id='part-2-selecting-training-and-evaluating-models'></a>
"""

# ╔═╡ 135dac9b-0bd9-4e1d-9b33-09b4b6661170
md"""
---

*This notebook was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ScientificTypes = "321657f4-b219-11e9-178b-2701a2544e81"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
UrlDownload = "856ac37a-3032-4c1c-9122-f86d88358c8b"

[compat]
CSV = "~0.9.10"
DataFrames = "~1.2.2"
PlutoUI = "~0.7.19"
ScientificTypes = "~2.3.3"
Tables = "~1.6.0"
UrlDownload = "~1.0.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0bc60e3006ad95b4bb7497698dd7c6d649b9bc06"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "74147e877531d7c172f70b492995bc2b5ca3a843"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.9.10"

[[CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "c308f209870fdbd84cb20332b6dfaf14bf3387f8"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.2"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f885e7e7c124f8c92650d61b9477b9ac2ee607dd"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.1"

[[ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "9a1d594397670492219635b35a3d830b04730d62"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.1"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "794daf62dce7df839b8ed446fc59c68db4b5182f"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.3.3"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "2ea02796c118368c3eda414fc11f5a39259fa3d9"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.27"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[FilePathsBase]]
deps = ["Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "5440c1d26aa29ca9ea848559216e5ee5f16a8627"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.14"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "14eece7a3308b4d8be910e265c724a6ba51a9798"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.16"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "19cb49649f8c41de7fea32d089d37de917b553da"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.0.1"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "be9eef9f9d78cecb6f262f3c10da151a6c5ab827"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "c8b8775b2f242c80ea85c83714c64ecfa3c53355"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.3"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "e071adf21e165ea0d904b595544a8e514c8bb42c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.19"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a193d6ad9c45ada72c14b731a318bedd3c2f00cf"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.3.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "d940010be611ee9d67064fe559edbb305f8cc0eb"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.2.3"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[ScientificTypes]]
deps = ["CategoricalArrays", "ColorTypes", "Dates", "Distributions", "PrettyTables", "Reexport", "ScientificTypesBase", "StatisticalTraits", "Tables"]
git-tree-sha1 = "7a3efcacd212801a8cf2f961e8238ffb2109b30d"
uuid = "321657f4-b219-11e9-178b-2701a2544e81"
version = "2.3.3"

[[ScientificTypesBase]]
git-tree-sha1 = "185e373beaf6b381c1e7151ce2c2a722351d6637"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "2.3.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "f45b34656397a1f6e729901dc9ef679610bd12b5"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.8"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f0bccf98e16759818ffc5d97ac3ebf87eb950150"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.1"

[[StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "730732cae4d3135e2f2182bd47f8d8b795ea4439"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "2.1.0"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "eb35dcc66558b2dda84079b9a1be17557d32091a"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.12"

[[StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "385ab64e64e79f0cd7cfcf897169b91ebbb2d6c8"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.13"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UrlDownload]]
deps = ["HTTP", "ProgressMeter"]
git-tree-sha1 = "05f86730c7a53c9da603bd506a4fc9ad0851171c"
uuid = "856ac37a-3032-4c1c-9122-f86d88358c8b"
version = "1.0.0"

[[WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "c69f9da3ff2f4f02e811c3323c22e5dfcb584cfa"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.1"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─7e85780a-bd7e-4b5e-88ff-6053d64456f5
# ╟─d9536b16-bb2a-4f83-9666-9b9eeadc125b
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
# ╠═44737448-20e7-464b-bb15-33279dbe6c3a
# ╟─f4ec6544-bdd3-4c58-9009-4cfb2012998f
# ╠═5e3fc1a4-2e74-48c6-9864-152c603930ec
# ╠═6ec50b8f-9c57-4085-abd3-cb77d7a79683
# ╠═8e3fc777-327a-46e3-879e-dc128f3db7b3
# ╟─91abff52-6801-423d-a367-9aa3c6cf34d0
# ╠═8d7b3ba5-a840-4377-bf32-94657e65284e
# ╠═7482fffc-7b40-4e84-9589-873d4bc96cca
# ╠═7424426a-cdab-4b1e-9afd-65a5b5facac9
# ╠═dd8cef99-8f55-47a8-b6c8-54f5ed90a964
# ╟─41012c9e-8e71-4a9d-9292-fe822525fc77
# ╠═c1c65dab-f82a-400f-ae5e-4b83dcbc9675
# ╠═0033787e-0796-41c6-a81d-fc52d08ea8e0
# ╟─e33a8562-1689-43ad-8a29-32f11452654a
# ╠═c7135c01-4881-45f6-a5f1-151fcbe229a2
# ╟─0e1afb98-7b11-4eb8-81bc-2d5603785a09
# ╟─ffcd5bb0-b53e-4ac8-9d87-1d1fbb0e3997
# ╟─f4a155dc-f3f2-4c44-b950-fddef2a1fb10
# ╟─308f453a-767e-4742-bab9-13a4b5e22567
# ╟─09cd28d1-eabf-488a-951b-64cb2a36c8e3
# ╠═9428e4a9-8e51-4d13-b0e6-441461cc8770
# ╠═b0d23945-5f6d-489e-8cb0-de601961bc02
# ╟─ed5001c5-e5b9-4a84-a87d-950c50fb2955
# ╠═91c12549-e518-47d3-8448-4bcb089096cd
# ╟─62be5b63-a6a3-48b3-b23d-a60990b06fec
# ╠═4eaac509-d637-41bf-85b6-a8154be0f959
# ╠═fef11f2a-da56-430d-b56d-ac411bbca555
# ╠═c37d5a96-29a7-465e-9ed4-cd7542e9065a
# ╟─1257966c-8043-49de-a012-a84240254fb6
# ╟─27415a96-42ec-4dc9-9428-713a24ceeda9
# ╠═3b776076-56c6-4215-bbdd-9799f7bb2e60
# ╠═0cf291d4-4e7d-4b28-8562-56d54658ca00
# ╟─919fcf78-227b-43c4-97a7-88e3af4f10ee
# ╟─e72b4214-b18f-40ba-a0c0-a78f67d659a0
# ╠═aa98759a-f43d-491e-b372-8c9866e51852
# ╠═cfb975ac-d946-4204-8427-bccb0a0e1ae7
# ╟─eba6d196-f627-49be-8f80-51fef8c5aa59
# ╠═74321d51-1831-4822-8f3d-b82f1e7b6f7a
# ╠═6971b1f8-d607-42ae-ab04-579e5608ae53
# ╟─ee1c9b0f-62f5-4dfb-89b8-6a303aca22b1
# ╠═fa257731-0521-4ffe-86cf-35420d9e6995
# ╠═e6f680e3-3dd7-421c-bb76-0ab026336d20
# ╟─75f109af-7651-477a-a6f7-2e3eb635a58d
# ╟─8134229e-c93c-482a-a29a-8d0445351914
# ╠═6c7d2c63-9bec-4439-be65-28d0fcca50a8
# ╠═ac973148-6168-473a-adb7-3880e5bd45b6
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
# ╠═e5c60366-e03c-490f-9475-e53e5b3347d4
# ╠═dc8c4bd5-e753-4b7b-9961-a09632c33fb0
# ╠═34c94131-ef2a-4f86-b7cc-46f4ef988c68
# ╠═a9238f1a-2751-46a8-9d69-62669915ed91
# ╠═8f5681ef-5432-4daf-90fb-7f2721f6fcc7
# ╠═281afa2b-c539-4c61-8890-979691212d9b
# ╠═67054d46-550d-40e6-a6fa-eaf14df5d44b
# ╠═ec84d270-fcb9-4b4b-8025-5084804a9f6c
# ╠═d3c65fbf-1a6c-4779-9e90-09023d201062
# ╠═1c530c2e-f0b8-4e30-b7bb-2f33ef765cc0
# ╠═09b804b7-30f7-4831-9625-01172c4a0081
# ╠═1e774630-8eac-4394-af4f-11c7de9e21dd
# ╠═347d35af-60d3-4422-8dba-241d9b744683
# ╠═3aac62ed-9e28-4db9-a172-d57784b35656
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
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
