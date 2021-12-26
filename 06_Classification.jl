### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ 3247c742-65df-11ec-0eed-7d76c7786941
begin
	using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
	Pkg.precompile()

	using PlutoUI
	using GLMNet
	using RDatasets
	using MLBase
	using Plots
	using DecisionTree
	using Distances
	using NearestNeighbors
	using Random
	using LinearAlgebra
	using DataStructures
	using LIBSVM
end;

# ╔═╡ ff42960a-f27f-41b8-8eee-14fb5b0454ba
PlutoUI.TableOfContents(aside=true, indent=true, depth=3)

# ╔═╡ b4c7b20b-ef19-4684-933b-7a00bfb69dce
html"""
<p align="center">
	<img src="https://github.com/JuliaAcademy/DataScience/blob/main/datascience.png?raw=true" alt="Course-Logo" width="450px">
</p>
"""

# ╔═╡ 73a1eed0-af5b-42b3-a4b8-a6f191f49daa
md"""
# 🔧 Classification

Put simply, classification is the task of predicting a label for a given observation. For example: you are given certain physical descriptions of an animal, and your taks is to classify them as either a dog or a cat. Here, we will classify iris flowers.
"""

# ╔═╡ Cell order:
# ╟─3247c742-65df-11ec-0eed-7d76c7786941
# ╟─ff42960a-f27f-41b8-8eee-14fb5b0454ba
# ╟─b4c7b20b-ef19-4684-933b-7a00bfb69dce
# ╟─73a1eed0-af5b-42b3-a4b8-a6f191f49daa
