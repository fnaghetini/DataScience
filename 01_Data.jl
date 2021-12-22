### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ c730d000-5ed6-11ec-36b6-39741ccad6c7
begin
	using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
	Pkg.precompile()

	using PlutoUI
	using BenchmarkTools
	using DataFrames
	using DelimitedFiles
	using CSV
	using XLSX
	using Downloads
	using JLD, NPZ, MAT, RData
	using RCall
end;

# ╔═╡ 62c5d4f2-9776-43ae-8125-0d429d6cde42
PlutoUI.TableOfContents(aside=true, indent=true, depth=3)

# ╔═╡ 34f92fb1-6c1e-4db6-b795-8a526007ea92
html"""
<p align="center">
	<img src="https://github.com/JuliaAcademy/DataScience/blob/main/datascience.png?raw=true" alt="Course-Logo" width="450px">
</p>
"""

# ╔═╡ 13a10dac-8c7e-4e72-9fa9-d58ca8dd0202
md"""
# 💾 Data

Being able to easily load and process data is a crucial task that can make any data science more pleasant. In this notebook, we will cover most common types often encountered in data science tasks, and we will be using this data throughout the rest of this tutorial.

[YouTube link](https://www.youtube.com/watch?v=iG1dZBaxS-U)
"""

# ╔═╡ 5100df3f-c5d3-4346-bcd9-8afc92bc559c
md"""
## Get some data

In Julia, it's pretty easy to dowload a file from the web using the `download` function. But also, you can use your favorite command line commad to download files by easily switching from Julia via the `;` key. Let's try both.

**Note:** `download` depends on external tools such as curl, wget or fetch. So you must have one of these.
"""

# ╔═╡ abe62085-b630-4816-af11-b39133892849
data₁ = Downloads.download("https://raw.githubusercontent.com/nassarhuda/easy_data/master/programming_languages.csv", "programminglanguages.csv")

# ╔═╡ 51c909b1-1d37-458f-acd6-68f4041bc92f
md"""
## Read your data from text files

The key question here is to load data from files such as `csv` files, `xlsx` files, or just raw text files. We will go over some Julia packages that will allow us to read such files very easily.
"""

# ╔═╡ 6483858c-2e58-4930-b4e9-76b47032eac4
md"""
### DelimitedFiles.jl

Let's start with the package `DelimitedFiles` which is in the standard library. When loaded, the file is stored as a `Matrix`.

> **readdlm**(source, 
> delim::AbstractChar, 
> T::Type, 
> eol::AbstractChar; 
> header=false, 
> skipstart=0, 
> skipblanks=true, 
> use_mmap, 
> quotes=true, 
> dims, 
> comments=false, 
> comment_char='#')
"""

# ╔═╡ 89a5a316-5058-4ff2-8d71-fc832a84c58d
begin
	data₂, header = readdlm("programming_languages.csv",',';header=true)
	header
end

# ╔═╡ 853a4e90-ac39-4d73-9953-c7b70e8f78ef
data₂

# ╔═╡ 3a22c018-2ca7-41df-9ee4-91dc8b64d940
md"""
> **Note:** the output table is a Matrix (2D Array).

We can use the function `writedlm`, from `DelimitedFiles.jl` package, to write to a text file...
"""

# ╔═╡ b4fcdcf9-793f-4d4a-a5fb-950108cb8188
writedlm("output/programminglanguages_dlm.txt", data₂, '-')

# ╔═╡ cfa37c7c-810c-4095-8887-262dee4a9d04
md"""
### CSV.jl

A more powerful package to use here is the `CSV.jl` package. By default, this package imports the data to a `DataFrame`, which can have several advantages as we will see below.

_In general, `CSV.jl` is the recommended way to load CSVs in Julia_. Only use `DelimitedFiles.jl` when you have a more complicated file where you want to specify several things.
"""

# ╔═╡ 4b651b5a-e562-46ef-a3b5-8ee85b5a61ed
df = CSV.read("programming_languages.csv", DataFrame)

# ╔═╡ 11471bbf-b7b4-4b6e-90cb-9e9ff47bbe4a
md"""
We can use the `typeof` function to make sure that the output is a DataFrame.
"""

# ╔═╡ 2652dc01-d80d-46d9-ae7a-9afa534e88cb
typeof(df)

# ╔═╡ b2f35e6b-8c19-4729-87a3-292858bfbcb5
md"""
We can use the macro `@btime`, from `BenchMarkTools.jl`, to prove that `CSV.jl` is faster than `DelimitedFiles.jl`...

**Note:** Check Julia REPL to see the processing time.
"""

# ╔═╡ e1210609-a216-4d54-be63-04f10c5c15bc
@btime df₁, h₁ = readdlm("programming_languages.csv",',';header=true);

# ╔═╡ 008b635f-c4e5-475a-83af-4bc3fc722344
@btime df₂ = CSV.read("programming_languages.csv", DataFrame);

# ╔═╡ 934c8549-c86c-45e6-a9f5-56297c73513d
md"""
We can use the function `write`, from `CSV.jl` package, to write to a text file...

**Note:** if we want to write a Matrix, we could first convert it into a DataFrame (using the `DataFrame` function) and, then, write it using the `write` function.
"""

# ╔═╡ 92708ece-3620-4cf5-b09d-098002044623
CSV.write("output/programminglanguages_CSV.csv", df)

# ╔═╡ cac4f1b5-f9e3-469e-9576-553a1cf1b363
md"""
### Some DataFrame Operations

There are different ways to slice a DataFrame. Suppose that we want just the first 10 rows of a DataFrame...
"""

# ╔═╡ 3d378ec4-659f-4032-a5b7-6d6b6d402c2f
df[1:10,:]

# ╔═╡ e50ae64c-ff64-4597-bf75-d7c921d23225
md"""
Suppose that we want all the instances of a given feature (column)...
"""

# ╔═╡ d9ea3574-2ffb-4fdb-9ced-db1dc669ecd9
df[!,"year"]

# ╔═╡ 09ff6d8b-15f2-4d8e-a775-dee357a417e4
df.year

# ╔═╡ 843a668f-7b08-4bce-b02a-3153d0eb7d0d
df[!,"year"] == df.year

# ╔═╡ 159a35f4-09fd-47bd-9935-88668a7e7f27
typeof(df.year)

# ╔═╡ 494ebede-80d0-430e-91aa-0a9050a1bb29
md"""
We can use the `names` function to get an array of all column names...
"""

# ╔═╡ 00d12274-3057-4a52-8400-0c8c3c358e6b
names(df)

# ╔═╡ fede3ce2-da03-4ff9-a13e-7ce55ec05bf6
md"""
We can use the `describe` function to check some data stats...
"""

# ╔═╡ 74783bbb-3c9d-4e15-bc54-b3d8e7c2d230
describe(df)

# ╔═╡ e4b3a444-c487-4fe8-8048-873b8a3affb7
md"""
Let's create two DataFrames...
"""

# ╔═╡ 0becba29-d5ff-4595-86ab-6b0575674047
begin
	foods = ["Apple","Cucumber","Tomato","Banana"]
	calories = [105,47,22,105]

	df_cal = DataFrame(Item=foods, Calories=calories)
end

# ╔═╡ 9d522bdc-d066-40ac-84c4-47287d9e07e0
begin
	prices = [0.85,1.6,0.8,0.6]

	df_price = DataFrame(Item=foods, Price=prices)
end

# ╔═╡ 79de9741-c913-4c00-8833-4f1b68f80ea5
md"""
Now, we can use the `innerjoin` function to join both DataFrames on `Item` column...
"""

# ╔═╡ fd724d83-41bb-422c-860e-2bbeff17e0dc
df_comb = innerjoin(df_cal, df_price, on=:Item)

# ╔═╡ 51301eeb-709a-4ea9-a1a5-d7967f8dd55d
md"""
### XLSX.jl

Another type of files that we may often need to read is XLSX files. Let's try to read this type of file using the `XLSX.jl` package...

> **readdata**(filepath, sheetname, cellrange)
"""

# ╔═╡ 64e7e90b-cd7d-4569-a715-4f27f1787e40
xlsx₁ = XLSX.readdata("data/zillow_data_download_april2020.xlsx",
    			  	  "Sale_counts_city",
    			  	  "A1:F9")

# ╔═╡ e5f2c69b-fbd8-4048-aaed-792a64a4d66c
md"""
If you don't want to specify cell ranges, you can use the `readtable` function. However, this will take a little longer...

> **readtable**(filepath, sheetname)
"""

# ╔═╡ 26047637-1b77-4203-b705-e34f74137295
xlsx₂ = XLSX.readtable("data/zillow_data_download_april2020.xlsx",
					   "Sale_counts_city")

# ╔═╡ 0f3d505c-a575-419f-8155-fd071f510714
typeof(xlsx₂)

# ╔═╡ c2f1eab2-862a-4eb8-b233-71567cdc9963
md"""
Here, `xlsx₂` is a tuple of two items. The first item is an vector of vectors where each vector corresponds to a column in the excel file...
"""

# ╔═╡ 3a924b3e-3dbe-49b6-9e5e-5a48b0ffd385
xlsx₂[1]

# ╔═╡ 58bad348-0586-4d0f-8d54-7c3e1a7a2c40
md"""
We can access the first column by typing...
"""

# ╔═╡ 036de982-36ad-431a-8445-517178626911
xlsx₂[1][1]

# ╔═╡ 4ead9039-f4c2-4f82-bf1e-2146718014a3
md"""
We can access the third column first 10 elements...
"""

# ╔═╡ bb04dc6c-e62e-4b2c-a247-ebe4836d8013
xlsx₂[1][3][begin:10]

# ╔═╡ d9550872-fa7c-4ebd-9a4d-828182c8e49d
md"""
On the other hand, the second is the header with the column names...
"""

# ╔═╡ e01a65ab-394a-4718-b9ae-ca2bb30e20b5
xlsx₂[2]

# ╔═╡ 1eb26b3c-8a5a-4648-bb73-f8291b656edc
md"""
We could check the names of the last 10 columns...
"""

# ╔═╡ cb027055-2cf6-4f74-b265-9d01c23c2967
xlsx₂[2][end-9:end]

# ╔═╡ 12dccfc3-e976-46e6-81b0-dc51b48309f4
md"""
And we can easily store this data in a `DataFrame`. We can use the "splat" operator to unwrap these arrays and pass them to the DataFrame constructor...
"""

# ╔═╡ 403ac783-9c02-4fd1-8506-afe0af36a14b
df_xlsx = DataFrame(xlsx₂...)
# df_xlsx = DataFrame(xlsx₂[1], xlsx₂[2])

# ╔═╡ 36d8898e-980c-4f54-bbff-277ded2229ea
md"""
**Note:** the first argument is the actual data and the second one comprises the column names.

We can also easily write data to an XLSX file (takes several amount of time)...
"""

# ╔═╡ 8d769d53-1902-4a09-8228-ce60174b22bb
# XLSX.writetable("output/writefile_using_XLSX.xlsx",xlsx₂[1],xlsx₂[2])

# ╔═╡ bc29d2b5-e6c2-4b45-8b79-cbb012659be1
md"""
## Importing your data

Often, the data you want to import is not stored in plain text, and you might want to import different kinds of types. Here we will go over importing `jld`, `npz`, `rda`, and `mat` files. Hopefully, these four will capture the types from four common programming languages used in Data Science (Julia, Python, R and Matlab).

We will use a toy example here of a very small matrix. But the same syntax will hold for bigger files.
"""

# ╔═╡ 7c2fef9c-4e9a-4637-bc8e-6636af5f98ae
md"""
### Julia Data (JLD)

JLD, for which files conventionally have the extension `.jld`, is a widely-used format for data storage with the Julia programming language.

JLD is a specific "dialect" of HDF5, a cross-platform, multi-language data storage format most frequently used for scientific data. By comparison with "plain" HDF5, JLD files automatically add attributes and naming conventions to preserve type information for each object.
"""

# ╔═╡ 8f51aeb5-3e75-4e5a-8c8f-ed0201eb8ce0
# load JLD file
jld_data = JLD.load("data/mytempdata.jld")

# ╔═╡ 879ac394-8598-4399-957f-1280ecd99478
# see the data
jld_data["tempdata"]

# ╔═╡ 0d27e3b3-79e3-424d-bc6b-94252900f7fb
# save JLD file
save("output/mywrite.jld", "A", jld_data)

# ╔═╡ 13053d54-8680-41e2-b416-8a7bc50f6c92
# JLD type
typeof(jld_data)

# ╔═╡ e0f49c77-0c0d-4740-9227-5619d14748b6
md"""
### Numpy Data (NPZ)

Several arrays into a single file in uncompressed Numpy `.npz` format.
"""

# ╔═╡ f4ddf1bf-0cbf-47c5-bb5f-f8abf0fb8ac2
# load NPZ file
npz_data = npzread("data/mytempdata.npz")

# ╔═╡ a979107e-f2cf-42da-aefa-5cb249bc7073
# save NPZ file
npzwrite("output/mywrite.npz", npz_data)

# ╔═╡ edd2093e-f84a-4563-acb7-219f2bd07399
# NPZ type
typeof(npz_data)

# ╔═╡ be91882d-fe73-40d0-8e75-30c2e8ddd564
md"""
### R Data (RDA)

The RData format (usually with extension `.rdata` or `.rda`) is a format designed for use with R, a system for statistical computation and related graphics, for storing a complete R workspace or selected "objects" from a workspace in a form that can be loaded back by R.
"""

# ╔═╡ 1f3092e8-3f3a-4d29-a9f5-e64ef96c13a8
# load RDA file
R_data = RData.load("data/mytempdata.rda")

# ╔═╡ cf9a1b00-79dd-4305-93ad-69f229319ce2
# see the data
R_data["tempdata"]

# ╔═╡ 1b49419d-18cd-4617-9442-15080e3ef588
begin
	# save RDA file
	@rput R_data
	R"save(R_data, file=\"output/mywrite.rda\")"
end

# ╔═╡ c3bce6db-d503-4a68-b83f-278c6d603f07
md"""
**Note:** we must use the `RCall.jl` package to save `.rda` files.
"""

# ╔═╡ 7131ac1c-984e-4ad9-a1cd-b1d46a4f240e
# RDA type
typeof(R_data)

# ╔═╡ 4f70b5df-0026-43cd-b769-d4ca50145fbd
md"""
### MATLAB® Data (MAT)

MAT-files are binary MATLAB® `.mat` files that store workspace variables.
"""

# ╔═╡ 0c1201b9-6acb-4206-a806-e09e04c6a65e
# load MAT file
matlab_data = matread("data/mytempdata.mat")

# ╔═╡ cc555ffc-0eea-4e35-b0d1-36d3a734f9f2
# see the data
matlab_data["tempdata"]

# ╔═╡ 5516dd98-3a87-4801-a62e-e91a95e4923a
# save MAT file
matwrite("output/mywrite.mat", matlab_data)

# ╔═╡ 53be8986-1f06-45a5-9303-5018a3ddf06f
# MAT type
typeof(matlab_data)

# ╔═╡ ecefa934-49a8-4abf-880d-d91d3f9f9d05
md"""
All files, when loaded, are `Dict`. The only exception is NPZ data, which is loaded as an `Matrix`.
"""

# ╔═╡ 01be19ba-2aaa-4746-af5d-8ce7f1cf3a55
md"""
## Processing different types of data

We will mainly cover `Vector` (`Matrix` included), `DataFrame`, and `Dict`. Let's bring back our programming languages dataset `data₂` and start playing it the matrix it's stored in.
"""

# ╔═╡ 3a6f9dc0-2186-440a-9d78-5da6acabcdae
md"""
### Matrix
"""

# ╔═╡ 4f9dca90-9f1a-4c0e-a510-fa640d109b0d
data₂

# ╔═╡ 57b8265f-be5c-4f05-91c6-6c4fec4f0a6b
md"""
Here are some quick questions we might want to ask about this simple data.

> **Q1:** Which year was was a given language invented?
"""

# ╔═╡ 157e9e3d-2e10-418e-997e-0178c14d7b3d
function year_created_mtx(data, language::String)
    loc = findfirst(data[:,2] .== language)
    !isnothing(loc) && return data[loc,1]
	error("Error: Language not found!")
end

# ╔═╡ e7bd695e-c7d6-4b38-8520-7d915e740015
md"""
**Note:** this function return the year just when `loc` local variable is not `nothing`. Otherwise, it returns an error message.
"""

# ╔═╡ c9ead4e0-aa01-43bc-8107-72d8ec4c073b
begin
	julia_year1 = year_created_mtx(data₂, "Julia")

	"Julia was created in $julia_year1."
end

# ╔═╡ 92b4a2d9-b271-42b3-97d1-647300a9439a
begin
	cobol_year1 = year_created_mtx(data₂, "COBOL")

	"COBOL was created in $cobol_year1."
end

# ╔═╡ 01a73335-69b9-4297-9910-a441848fe2c5
md"""
> **Q2:** How many languages were created in a given year?
"""

# ╔═╡ fbf69873-af41-428c-9bba-d63cdb20149b
function langs_per_year_mtx(data, year::Int64)
	count = length(findall(data[:,1] .== year))

	return count
end

# ╔═╡ 1ea19d7e-5748-4559-b2d2-fd4f4d449516
begin
	langs_1988₁ = langs_per_year_mtx(data₂, 1988)

	"In 1988, $langs_1988₁ language(s) was/were created."
end

# ╔═╡ 2592f7cd-95ec-493c-93e1-bbedf433367f
begin
	langs_2006₁ = langs_per_year_mtx(data₂, 2006)

	"In 2006, $langs_2006₁ language(s) was/were created."
end

# ╔═╡ 12121f45-8a7b-4709-bc85-b2fde7313a21
md"""
### DataFrame

Now let's try to store this data in a DataFrame...
"""

# ╔═╡ d455f80e-c5b5-4b1a-9667-86fade5ee3d3
# anonymous column names
data₃ = DataFrame(data₂, :auto)

# ╔═╡ 42c72fde-8331-4e40-b761-c50d0e2e8486
# specifying column names
data₄ = DataFrame(Year = data₂[:,1], Language = data₂[:,2])

# ╔═╡ 201a8534-91ee-4ac9-a120-51b29f196c23
# specifying both column names and data types
data₅ = DataFrame(Year = Int.(data₂[:,1]), Language = string.(data₂[:,2]))

# ╔═╡ 0b14e9fc-62e6-4ad7-ab67-eb36e340749c
md"""
> **Q1:** Which year was was a given language invented?
"""

# ╔═╡ 679b2b83-07f9-40bd-bdd1-89e7233bfade
function year_created_df(df,language::String)
    loc = findfirst(df.Language .== language)
	!isnothing(loc) && return df.Year[loc]
	error("Error: Language not found!")
end

# ╔═╡ f4738476-8b05-4dbc-a15e-6b3d59c90c50
md"""
**Note:** this function return the year just when `loc` local variable is not `nothing`. Otherwise, it returns an error message.
"""

# ╔═╡ c6b170be-6576-480e-b2ba-65d3b853871a
begin
	julia_year2 = year_created_df(data₅, "Julia")

	"Julia was created in $julia_year2."
end

# ╔═╡ 9977fdaa-db30-43a5-bd58-14087c2163e4
begin
	cobol_year2 = year_created_df(data₅, "COBOL")

	"COBOL was created in $cobol_year2."
end

# ╔═╡ a137ebd7-9db4-4c29-a694-b1acad5214a3
md"""
> **Q2:** How many languages were created in a given year?
"""

# ╔═╡ 2cc2938b-58d5-4c27-bdab-dddf22231bc8
function langs_per_year_df(df, year::Int64)
    count = length(findall(df.Year.==year))
    return count
end

# ╔═╡ ae7a1714-2245-4d5a-89bb-10b3adff787a
begin
	langs_1988₂ = langs_per_year_mtx(data₅, 1988)

	"In 1988, $langs_1988₂ language(s) was/were created."
end

# ╔═╡ ce4ac86d-5e06-4c3e-93d6-ebdee46439bd
begin
	langs_2006₂ = langs_per_year_mtx(data₅, 2006)

	"In 2006, $langs_2006₂ language(s) was/were created."
end

# ╔═╡ 515b7479-97d8-464b-8245-1034f3cd8bdc
md"""
### Dictionary

Next, we'll use dictionaries. A quick way to create a dictionary is with the `Dict()` command. But this creates a dictionary without types. Here, we will specify the types of this dictionary.
"""

# ╔═╡ 3b08a2c8-546b-4a6c-8ae0-2a1e64578af8
# dict from a list of tuples
Dict([("A", 1), ("B", 2)])

# ╔═╡ 21551a38-57fc-42c2-81a2-b314bb06250b
md"""
**Note:** the key type is `String`, while the value type is `Int64`.
"""

# ╔═╡ 6da1f5a7-56a2-4a7a-abfc-25ec3f44f892
# dictionary from a list of tuples
Dict([("A", 1), ("B", 2), ("C", [1,2,3])])

# ╔═╡ 43ad8297-d94c-45a0-941f-ab0d5bfd2ddb
md"""
**Note:** the key type is `String`, while the value type is `Any`, since we have integers and arrays as values.
"""

# ╔═╡ 9390d98f-94eb-48bb-ba9c-e6b3f67f762f
# empty dict
Dict()

# ╔═╡ 30718df7-5085-4d0d-8a86-7b4a08ff20b2
# empty dict, specifying that keys are Integers and values are Vectors of Strings
dict₁ = Dict{Integer, Vector{String}}()

# ╔═╡ f77a5189-19b4-4f99-b394-72190253912f
# appending a key-value pair to dict
dict₁[2012] = ["Julia", "Programming", "Language"]

# ╔═╡ 50ba8f3b-5104-4d4f-85ea-831d4afe9164
# updated dict
dict₁

# ╔═╡ bb814af7-62e3-4744-955f-31b557540057
md"""
Now, let's populate the dictionary with years as keys and vectors that hold all the programming languages created in each year as their values. Even though this looks like more work, we often need to do it just once.
"""

# ╔═╡ 9193df28-7ed1-4853-a97b-aa5a1232e07b
begin
	lang_dict = Dict{Integer, Vector{String}}()

	for i in 1:size(data₂,1)
		year, lang = data₂[i,:]
		if year ∈ keys(lang_dict)
			lang_dict[year] = push!(lang_dict[year], lang)
		else
			lang_dict[year] = [lang]
		end
	end
end

# ╔═╡ c5aca233-9a26-4800-9b8d-7f0f03852626
lang_dict

# ╔═╡ 7b2d5919-9629-4ca9-8195-f26183734044
md"""
> **Note:** there is an `enumerate()` method in Julia which is useful when you need not only the values `x` over which you are iterating, but also the number of iterations (index) so far.
**Syntax:**
```julia
for (index, value) in enumerate(df.column)
```
"""

# ╔═╡ d8054919-1554-4c73-a4a2-09383ee09224
md"""
We can check the size of the dictionary...
"""

# ╔═╡ 7d57f6cc-ab44-4658-a73e-92c873111155
length(keys(lang_dict))

# ╔═╡ a322a0e7-3984-4c00-95a5-7b23fc6ec06f
md"""
Or we can check the size of the vector made up by unique `Year` values...
"""

# ╔═╡ e0a28251-a7d6-4c17-b49e-84ffc91c60b6
length(unique(data₂[:,1]))

# ╔═╡ 9073e71f-b922-44a4-a470-5cb081588fcd
md"""
> **Q1:** Which year was was a given language invented?
"""

# ╔═╡ f9325e7b-8df8-448a-98b2-9ea8add93da6
function year_created_dict(dict,language::String)
    keys_vec = collect(keys(dict))
    lookup = map(keyid -> findfirst(dict[keyid] .== language), keys_vec)
    
    return keys_vec[findfirst((!isnothing).(lookup))]
end

# ╔═╡ c33df74d-dd81-4efa-81d6-a5deb4685f1d
begin
	julia_year3 = year_created_dict(lang_dict, "Julia")

	"Julia was created in $julia_year3."
end

# ╔═╡ a08ad2fd-5d3c-4e76-a059-0608de3e36aa
md"""
> **Q2:** How many languages were created in a given year?
"""

# ╔═╡ fb6df54d-0765-4493-b889-17707ff0cf14
langs_per_year_dict(dict, year::Int64) = length(dict[year])

# ╔═╡ d5e60574-5afd-498c-aedd-1f5ffc4c6bd7
begin
	langs_1988₃ = langs_per_year_dict(lang_dict, 1988)

	"In 1988, $langs_1988₃ language(s) was/were created."
end

# ╔═╡ ee8bcf97-9efa-47f0-a82e-cfaa2efd68e6
md"""
## Missing data

Let's remove the first year value from our data (Matrix) and, after that, create a DataFrame...
"""

# ╔═╡ b3927e97-7516-4a80-bcc1-49614cf67166
begin
	data₂[1,1] = missing

	data₆ = DataFrame(Year=data₂[:,1], Language=data₂[:,2])
end

# ╔═╡ dc7ed277-b044-464e-8ad3-998a83b00d1b
md"""
We can now use the `dropmissing!` function to remove (inplace) the line with missing value...
"""

# ╔═╡ e2b6b920-2e05-4168-af16-07e9611d895d
dropmissing!(data₆)

# ╔═╡ Cell order:
# ╟─c730d000-5ed6-11ec-36b6-39741ccad6c7
# ╟─62c5d4f2-9776-43ae-8125-0d429d6cde42
# ╟─34f92fb1-6c1e-4db6-b795-8a526007ea92
# ╟─13a10dac-8c7e-4e72-9fa9-d58ca8dd0202
# ╟─5100df3f-c5d3-4346-bcd9-8afc92bc559c
# ╠═abe62085-b630-4816-af11-b39133892849
# ╟─51c909b1-1d37-458f-acd6-68f4041bc92f
# ╟─6483858c-2e58-4930-b4e9-76b47032eac4
# ╠═89a5a316-5058-4ff2-8d71-fc832a84c58d
# ╠═853a4e90-ac39-4d73-9953-c7b70e8f78ef
# ╟─3a22c018-2ca7-41df-9ee4-91dc8b64d940
# ╠═b4fcdcf9-793f-4d4a-a5fb-950108cb8188
# ╟─cfa37c7c-810c-4095-8887-262dee4a9d04
# ╠═4b651b5a-e562-46ef-a3b5-8ee85b5a61ed
# ╟─11471bbf-b7b4-4b6e-90cb-9e9ff47bbe4a
# ╠═2652dc01-d80d-46d9-ae7a-9afa534e88cb
# ╟─b2f35e6b-8c19-4729-87a3-292858bfbcb5
# ╠═e1210609-a216-4d54-be63-04f10c5c15bc
# ╠═008b635f-c4e5-475a-83af-4bc3fc722344
# ╟─934c8549-c86c-45e6-a9f5-56297c73513d
# ╠═92708ece-3620-4cf5-b09d-098002044623
# ╟─cac4f1b5-f9e3-469e-9576-553a1cf1b363
# ╠═3d378ec4-659f-4032-a5b7-6d6b6d402c2f
# ╟─e50ae64c-ff64-4597-bf75-d7c921d23225
# ╠═d9ea3574-2ffb-4fdb-9ced-db1dc669ecd9
# ╠═09ff6d8b-15f2-4d8e-a775-dee357a417e4
# ╠═843a668f-7b08-4bce-b02a-3153d0eb7d0d
# ╠═159a35f4-09fd-47bd-9935-88668a7e7f27
# ╟─494ebede-80d0-430e-91aa-0a9050a1bb29
# ╠═00d12274-3057-4a52-8400-0c8c3c358e6b
# ╟─fede3ce2-da03-4ff9-a13e-7ce55ec05bf6
# ╠═74783bbb-3c9d-4e15-bc54-b3d8e7c2d230
# ╟─e4b3a444-c487-4fe8-8048-873b8a3affb7
# ╠═0becba29-d5ff-4595-86ab-6b0575674047
# ╠═9d522bdc-d066-40ac-84c4-47287d9e07e0
# ╟─79de9741-c913-4c00-8833-4f1b68f80ea5
# ╠═fd724d83-41bb-422c-860e-2bbeff17e0dc
# ╟─51301eeb-709a-4ea9-a1a5-d7967f8dd55d
# ╠═64e7e90b-cd7d-4569-a715-4f27f1787e40
# ╟─e5f2c69b-fbd8-4048-aaed-792a64a4d66c
# ╠═26047637-1b77-4203-b705-e34f74137295
# ╠═0f3d505c-a575-419f-8155-fd071f510714
# ╟─c2f1eab2-862a-4eb8-b233-71567cdc9963
# ╠═3a924b3e-3dbe-49b6-9e5e-5a48b0ffd385
# ╟─58bad348-0586-4d0f-8d54-7c3e1a7a2c40
# ╠═036de982-36ad-431a-8445-517178626911
# ╟─4ead9039-f4c2-4f82-bf1e-2146718014a3
# ╠═bb04dc6c-e62e-4b2c-a247-ebe4836d8013
# ╟─d9550872-fa7c-4ebd-9a4d-828182c8e49d
# ╠═e01a65ab-394a-4718-b9ae-ca2bb30e20b5
# ╟─1eb26b3c-8a5a-4648-bb73-f8291b656edc
# ╠═cb027055-2cf6-4f74-b265-9d01c23c2967
# ╟─12dccfc3-e976-46e6-81b0-dc51b48309f4
# ╠═403ac783-9c02-4fd1-8506-afe0af36a14b
# ╟─36d8898e-980c-4f54-bbff-277ded2229ea
# ╠═8d769d53-1902-4a09-8228-ce60174b22bb
# ╟─bc29d2b5-e6c2-4b45-8b79-cbb012659be1
# ╟─7c2fef9c-4e9a-4637-bc8e-6636af5f98ae
# ╠═8f51aeb5-3e75-4e5a-8c8f-ed0201eb8ce0
# ╠═879ac394-8598-4399-957f-1280ecd99478
# ╠═0d27e3b3-79e3-424d-bc6b-94252900f7fb
# ╠═13053d54-8680-41e2-b416-8a7bc50f6c92
# ╟─e0f49c77-0c0d-4740-9227-5619d14748b6
# ╠═f4ddf1bf-0cbf-47c5-bb5f-f8abf0fb8ac2
# ╠═a979107e-f2cf-42da-aefa-5cb249bc7073
# ╠═edd2093e-f84a-4563-acb7-219f2bd07399
# ╟─be91882d-fe73-40d0-8e75-30c2e8ddd564
# ╠═1f3092e8-3f3a-4d29-a9f5-e64ef96c13a8
# ╠═cf9a1b00-79dd-4305-93ad-69f229319ce2
# ╠═1b49419d-18cd-4617-9442-15080e3ef588
# ╟─c3bce6db-d503-4a68-b83f-278c6d603f07
# ╠═7131ac1c-984e-4ad9-a1cd-b1d46a4f240e
# ╟─4f70b5df-0026-43cd-b769-d4ca50145fbd
# ╠═0c1201b9-6acb-4206-a806-e09e04c6a65e
# ╠═cc555ffc-0eea-4e35-b0d1-36d3a734f9f2
# ╠═5516dd98-3a87-4801-a62e-e91a95e4923a
# ╠═53be8986-1f06-45a5-9303-5018a3ddf06f
# ╟─ecefa934-49a8-4abf-880d-d91d3f9f9d05
# ╟─01be19ba-2aaa-4746-af5d-8ce7f1cf3a55
# ╟─3a6f9dc0-2186-440a-9d78-5da6acabcdae
# ╠═4f9dca90-9f1a-4c0e-a510-fa640d109b0d
# ╟─57b8265f-be5c-4f05-91c6-6c4fec4f0a6b
# ╠═157e9e3d-2e10-418e-997e-0178c14d7b3d
# ╟─e7bd695e-c7d6-4b38-8520-7d915e740015
# ╠═c9ead4e0-aa01-43bc-8107-72d8ec4c073b
# ╠═92b4a2d9-b271-42b3-97d1-647300a9439a
# ╟─01a73335-69b9-4297-9910-a441848fe2c5
# ╠═fbf69873-af41-428c-9bba-d63cdb20149b
# ╠═1ea19d7e-5748-4559-b2d2-fd4f4d449516
# ╠═2592f7cd-95ec-493c-93e1-bbedf433367f
# ╟─12121f45-8a7b-4709-bc85-b2fde7313a21
# ╠═d455f80e-c5b5-4b1a-9667-86fade5ee3d3
# ╠═42c72fde-8331-4e40-b761-c50d0e2e8486
# ╠═201a8534-91ee-4ac9-a120-51b29f196c23
# ╟─0b14e9fc-62e6-4ad7-ab67-eb36e340749c
# ╠═679b2b83-07f9-40bd-bdd1-89e7233bfade
# ╟─f4738476-8b05-4dbc-a15e-6b3d59c90c50
# ╠═c6b170be-6576-480e-b2ba-65d3b853871a
# ╠═9977fdaa-db30-43a5-bd58-14087c2163e4
# ╟─a137ebd7-9db4-4c29-a694-b1acad5214a3
# ╠═2cc2938b-58d5-4c27-bdab-dddf22231bc8
# ╠═ae7a1714-2245-4d5a-89bb-10b3adff787a
# ╠═ce4ac86d-5e06-4c3e-93d6-ebdee46439bd
# ╟─515b7479-97d8-464b-8245-1034f3cd8bdc
# ╠═3b08a2c8-546b-4a6c-8ae0-2a1e64578af8
# ╠═21551a38-57fc-42c2-81a2-b314bb06250b
# ╠═6da1f5a7-56a2-4a7a-abfc-25ec3f44f892
# ╟─43ad8297-d94c-45a0-941f-ab0d5bfd2ddb
# ╠═9390d98f-94eb-48bb-ba9c-e6b3f67f762f
# ╠═30718df7-5085-4d0d-8a86-7b4a08ff20b2
# ╠═f77a5189-19b4-4f99-b394-72190253912f
# ╠═50ba8f3b-5104-4d4f-85ea-831d4afe9164
# ╟─bb814af7-62e3-4744-955f-31b557540057
# ╠═9193df28-7ed1-4853-a97b-aa5a1232e07b
# ╠═c5aca233-9a26-4800-9b8d-7f0f03852626
# ╟─7b2d5919-9629-4ca9-8195-f26183734044
# ╟─d8054919-1554-4c73-a4a2-09383ee09224
# ╠═7d57f6cc-ab44-4658-a73e-92c873111155
# ╟─a322a0e7-3984-4c00-95a5-7b23fc6ec06f
# ╠═e0a28251-a7d6-4c17-b49e-84ffc91c60b6
# ╟─9073e71f-b922-44a4-a470-5cb081588fcd
# ╠═f9325e7b-8df8-448a-98b2-9ea8add93da6
# ╠═c33df74d-dd81-4efa-81d6-a5deb4685f1d
# ╟─a08ad2fd-5d3c-4e76-a059-0608de3e36aa
# ╠═fb6df54d-0765-4493-b889-17707ff0cf14
# ╠═d5e60574-5afd-498c-aedd-1f5ffc4c6bd7
# ╟─ee8bcf97-9efa-47f0-a82e-cfaa2efd68e6
# ╠═b3927e97-7516-4a80-bcc1-49614cf67166
# ╟─dc7ed277-b044-464e-8ad3-998a83b00d1b
# ╠═e2b6b920-2e05-4168-af16-07e9611d895d
