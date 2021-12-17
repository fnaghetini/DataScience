### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ c730d000-5ed6-11ec-36b6-39741ccad6c7
begin
	using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
	Pkg.instantiate()
	
	using BenchmarkTools
	using DataFrames
	using DelimitedFiles
	using CSV
	using XLSX
	using Downloads
end;

# ╔═╡ 13a10dac-8c7e-4e72-9fa9-d58ca8dd0202
md"""
# Data

Being able to easily load and process data is a crucial task that can make any data science more pleasant. In this notebook, we will cover most common types often encountered in data science tasks, and we will be using this data throughout the rest of this tutorial.
"""

# ╔═╡ 5100df3f-c5d3-4346-bcd9-8afc92bc559c
md"""
## 🗃️ Get some data

In Julia, it's pretty easy to dowload a file from the web using the `download` function. But also, you can use your favorite command line commad to download files by easily switching from Julia via the `;` key. Let's try both.

**Note:** `download` depends on external tools such as curl, wget or fetch. So you must have one of these.
"""

# ╔═╡ abe62085-b630-4816-af11-b39133892849
data₁ = Downloads.download("https://raw.githubusercontent.com/nassarhuda/easy_data/master/programming_languages.csv", "programminglanguages.csv")

# ╔═╡ 51c909b1-1d37-458f-acd6-68f4041bc92f
md"""
## 📂 Read your data from text files

The key question here is to load data from files such as `csv` files, `xlsx` files, or just raw text files. We will go over some Julia packages that will allow us to read such files very easily.

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


# ╔═╡ Cell order:
# ╟─c730d000-5ed6-11ec-36b6-39741ccad6c7
# ╟─13a10dac-8c7e-4e72-9fa9-d58ca8dd0202
# ╟─5100df3f-c5d3-4346-bcd9-8afc92bc559c
# ╠═abe62085-b630-4816-af11-b39133892849
# ╟─51c909b1-1d37-458f-acd6-68f4041bc92f
# ╠═89a5a316-5058-4ff2-8d71-fc832a84c58d
# ╠═853a4e90-ac39-4d73-9953-c7b70e8f78ef
# ╠═3a22c018-2ca7-41df-9ee4-91dc8b64d940
