### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ efff10ac-824b-422b-b2f9-4074ea3437fb
begin
	using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
	Pkg.precompile()

	using PlutoUI
	using UMAP
	using Makie
	using GLMakie
	using XLSX
	using VegaDatasets
	using DataFrames
	using MultivariateStats
	using RDatasets
	using StatsBase
	using Statistics
	using LinearAlgebra
	using Plots
	using ScikitLearn
	using MLBase
	using Distances
end;

# ╔═╡ 85e2d832-542c-49b0-a26d-b57a7a3d4404
PlutoUI.TableOfContents(aside=true, indent=true, depth=3)

# ╔═╡ 6e8c97d2-4d9a-4f00-99e1-69054903635c
html"""
<p align="center">
	<img src="https://github.com/JuliaAcademy/DataScience/blob/main/datascience.png?raw=true" alt="Course-Logo" width="450px">
</p>
"""

# ╔═╡ 133df7ea-d3ec-41fa-bcb8-5eae4c853a55
md"""
# 📝 Dimensionality Reduction

As the name says, dimensionality reduction is the idea of reducing your feature set to a much smaller number. Dimensionality reduction is often used in visualization of datasets to try and detect samples that are similar. Usually, the goal is to come up with a 2D visualization of your data, where every dot in your plot is representing an observation. We can use these plots to look for patterns in data. We will cover three dimensionality reduction techniques here:

1. t-SNE
2. PCA
3. UMAP

[YouTube Link](https://www.youtube.com/watch?v=hIsYy04zO7U)
"""

# ╔═╡ 5271ffe0-f73a-430e-8ccc-5c6bd0937ce0
md"""
## Load dataset

We will use a dataset from the [VegaDatasets.jl](https://github.com/queryverse/VegaDatasets.jl) package. The dataset is about car specifications of over 400 car models.
"""

# ╔═╡ 6a53752f-3101-4b62-a661-c09c9df27f0f
data = VegaDatasets.dataset("cars") |> DataFrame

# ╔═╡ 9a491a1d-32a0-496c-887a-cdfd49f01836
md"""
Let's check the available variables with the `names` function...
"""

# ╔═╡ dcc6939b-ba8e-4f4a-abcf-aa90efa60682
names(data)

# ╔═╡ 56529677-8b5c-4912-b25a-728847b4f374
md"""
## Data wrangling

"""

# ╔═╡ 59d179aa-a19c-497d-ae56-6e0b0b0007dc
md"""
We will first drop missing data with the `dropmissing!` inplace function...
"""

# ╔═╡ 2f0ba288-19b9-49c8-845c-bdc2317328fc
dropmissing!(data)

# ╔═╡ 182b795f-c069-4ecd-aab6-89c66c93ee6e
md"""
Now, we need to save the numeric variables (columns 2 through 7)  as a `Matrix`. We will use this data to perform dimensionality reduction methods later...
"""

# ╔═╡ 56e3357f-a692-4b29-8ca2-390ba62b9679
M = data[:,2:7] |> Matrix

# ╔═╡ 53bae393-698c-4610-b18f-073a67d9bdc8
md"""
We are also going to use the `Origin` feature. However, since it is a vector of strings, we will need to encode it.

First, we will use the [MLBase.jl](https://github.com/JuliaStats/MLBase.jl) `labelmap` function to map `Origin` labels and, then, we will encode it using the `labelencode` function...
"""

# ╔═╡ 193d7109-5499-40c3-a25f-b7b8560db5b9
origin = data[!,:Origin]

# ╔═╡ c3d8e866-c754-4ee6-8c1c-fab7f8048181
origin_map = labelmap(origin)

# ╔═╡ 231955b0-5ccd-46ad-b78d-96f861f1938a
origin_codes = labelencode(origin_map, origin)

# ╔═╡ 841ef716-8dbf-4aec-947e-6c0409a7fd64
md"""
**Note:** after encoding, we have:
- USA → 1
- Japan → 2
- Europe → 3
"""

# ╔═╡ 2c86f61a-b328-4eac-a4ff-1e713007ef54
md"""
Finally, we must center and normalize the data...

**Note:** `dims=1` means that we are calculating both `mean` and `std` over _columns_.
"""

# ╔═╡ c08882ca-d617-4c0f-865f-898c0f639a82
center_M = (M .- mean(M, dims=1)) ./ std(M, dims=1)

# ╔═╡ a9afbe75-29a4-4a54-b6c2-2786d74f7a65
md"""
## PCA

**Principal component analysis (PCA)** is the process of computing the principal components and using them to perform a change of basis on the data, sometimes using only the first few principal components and ignoring the rest.

The principal components of a collection of points in a real coordinate space are a sequence of $p$ unit vectors, where the $i$-th vector is the direction of a line that best fits the data while being orthogonal to the first $i-1$ vectors. Here, a best-fitting line is defined as one that minimizes the average squared distance from the points to the line. These directions constitute an orthonormal basis in which different individual dimensions of the data are linearly uncorrelated.

**Note:** the details for the underlying mathematics can be found [here](https://en.wikipedia.org/wiki/Principal_component_analysis).

First, we will fit the model via PCA, using the [MultivariateStats.jl](https://github.com/JuliaStats/MultivariateStats.jl) package. `maxoutdim` is the output dimensions and we want it to be 2 in this case.

**Note:** since PCA expects each column to be an observation, we will use the transpose of the matrix.
"""

# ╔═╡ dffd6373-2e05-4746-bc44-ee2193ae2635
pca = fit(PCA, center_M', maxoutdim=2)

# ╔═╡ d13d825e-e807-4ef0-a9ab-45fc994500a2
md"""
Now, we can obtain the projection matrix by calling the function `projection`...

**Note:** once you multiply the projection matrix by your data, you get the dimensionality reduction of your data.
"""

# ╔═╡ cb2f6b28-d360-4bb8-a0c0-3d6d1b38e23d
P = projection(pca)

# ╔═╡ 8d89f524-68a0-425e-a4e9-2525ebdd914d
md"""
Given a PCA model, one can use it to transform observations into principal components, as:

```math
y = P^T(x - \mu)
```

where $P$ is the projection matrix, $x$ is the original feature vector and $\mu$  is the empirical mean vector by each original feature. We can perform this transformation manually or call `transform` function...
"""

# ╔═╡ 3946ec51-0336-4098-ba13-55abf68f9b41
# empirical mean vector
μ = mean(pca)

# ╔═╡ 8d219680-c870-4549-9421-5c7fff3a7607
# transforming the 1st instance manually
P' * (center_M[1,:] - μ)

# ╔═╡ a450f277-46cf-4f5b-b58a-b33b5aca5fb2
md"""
**Note:** the result above is the 1st feature vector represented by the first two principal components.
"""

# ╔═╡ 2e2d14e9-8fdb-4328-9218-f47307309f40
# transforming all data
Y = MultivariateStats.transform(pca, center_M')

# ╔═╡ 13b52e6a-dbe4-4dad-a7a6-260957fb0bdd
md"""
**Note:** `Y[:,1]` is the same as `P' * (center_data[1,:] - μ)`:
"""

# ╔═╡ 49f05ab7-b716-44c4-bb44-6c2e700a18ac
Y[:,1] ≈ (P' * (center_M[1,:] - μ))

# ╔═╡ c37484f0-3e35-4c5f-841a-e0e6061752b4
md"""
Given a PCA model, we also may use it to reconstruct (approximately) the observations from principal components, as:

```math
\tilde{x} = Py + \mu
```
where $P$ is the projection matrix, $x$ is the original feature vector and $\mu$  is the empirical mean vector by each original feature. We can perform this reconstruction using the `reconstruct` function...
"""

# ╔═╡ cf067613-8312-402c-88b2-fe507ce36ac9
X̅ = reconstruct(pca, Y)

# ╔═╡ 7594fb9c-443a-4ea8-a56a-f17c818ddbfe
md"""
However, since the reconstruction is approximate, the norm difference between $\tilde{x}$ and $x$ will not be zero...
"""

# ╔═╡ e0e09b17-da85-4879-bec8-adef9f125b31
norm(X̅ - center_M')

# ╔═╡ 0e73cd39-24d3-4de4-a30d-a53ad6f1d50c
md"""
Finally, we can generate a scatter plot of the cars...
"""

# ╔═╡ 5dfc1915-f832-4d72-b551-e3ddbfa693bc
Plots.scatter(Y[1,:], Y[2,:], color=:orange, legend=false,
			  xlabel="1st PC", ylabel="2nd PC")

# ╔═╡ 98c5d1d2-6c28-4c46-adeb-bb580d5dfe2a
md"""
Let's try to color the scatter plot by `Origin`...
"""

# ╔═╡ 8117e9fa-23d1-4539-aa28-c5bc4a3db849
begin
	# boolean lists
	usaᵢ = origin.=="USA"
	japᵢ = origin .== "Japan"
	eurᵢ = origin .== "Europe"
end;

# ╔═╡ 07dbb182-4565-48de-b41c-bd0488f67280
begin
	Plots.scatter(Y[1,usaᵢ], Y[2,usaᵢ], color=1, label="USA")
	Plots.scatter!(Y[1,japᵢ], Y[2,japᵢ], color=2, label="Japan")
	Plots.scatter!(Y[1,eurᵢ], Y[2,eurᵢ], color=3, label="Europe")

	Plots.xlabel!("1st PC")
	Plots.ylabel!("2nd PC")
end

# ╔═╡ 6bca80ac-8c1d-4f35-82e7-4c6c4206eb53
md"""
This is interesting! There seems to be three main clusters with cars from the US dominating two clusters.

Now, let's try to reduce the dimentionality to 3 principal components...
"""

# ╔═╡ 152fc839-661b-48aa-a047-9e16a35acff5
begin
	new_pca = fit(PCA, center_M', maxoutdim=3)
	new_Y = MultivariateStats.transform(new_pca, center_M')
end

# ╔═╡ 5f55ba88-c984-49d2-b4b9-e5cdd81e8c6c
scatter3d(new_Y[1,:], new_Y[2,:], new_Y[3,:], color=origin_codes,
		  xlabel="1st PC", ylabel="2nd PC", zlabel="3rd PC",
		  legend=false, camera=(50, 30))

# ╔═╡ bac92bf8-3059-4fcc-b1a9-42c91637fd17
md"""
## t-SNE

**t-Distributed Stochastic Neighbor Embedding (t-SNE)** is a statistical method for visualizing high-dimensional data by giving each datapoint a location in a two or three-dimensional map. It is based on Stochastic Neighbor Embedding originally developed by Sam Roweis and Geoffrey Hinton,[1] where Laurens van der Maaten proposed the t-distributed variant.

**Note:** the details for the underlying mathematics can be found [here](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding).

The next method we will use for dimensionality reduction is t-SNE. There are multiple ways you can call t-SNE from Julia. But we will take this opportunity to try out something new... We will call a function from the scikitlearn Python package. This makes use of the package [ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl).

**Note:** check this [t-SNE notebook](https://github.com/nassarhuda/JuliaTutorials/blob/master/TSNE/TSNE.ipynb).
"""

# ╔═╡ 6c163758-55b5-474b-8269-203884d70757
# importing sklearn t-SNE
@sk_import manifold: TSNE

# ╔═╡ 4676988f-5687-4aad-a054-48ae5defd46e
# instantiate a t-SNE object with 2 components
tsne = TSNE(n_components=2, perplexity=20.0, early_exaggeration=50)

# ╔═╡ 026668de-5323-4e40-b75c-c2eefbf46ef5
# fit-transforming data
Y₂ = tsne.fit_transform(center_M)

# ╔═╡ 7fb412b7-573e-4c5a-bd39-d9d7e637a9d4
md"""
**Note:** here, we do not have to work with the transpose data, since ScikitLearn already deals with observations as rows...

Let's plot the components...
"""

# ╔═╡ 8faa2556-bcfc-4289-9d17-d75ad10f2245
begin
	Plots.scatter(Y₂[usaᵢ,1], Y₂[usaᵢ,2], color=1, label="USA")
	Plots.scatter!(Y₂[japᵢ,1], Y₂[japᵢ,2], color=2, label="Japan")
	Plots.scatter!(Y₂[eurᵢ,1], Y₂[eurᵢ,2], color=3, label="Europe")

	Plots.xlabel!("1st Component")
	Plots.ylabel!("2nd Component")
end

# ╔═╡ 3114d2e5-e6eb-4198-98f3-4abfbb45bdeb
md"""
This is interesting! The same patterns saw before appears to hold here too.
"""

# ╔═╡ 7675ae48-a6ae-4189-990f-c1469ee11929
md"""
## UMAP

**Uniform Manifold Approximation and Projection (UMAP)** is a dimension reduction technique that can be used for visualisation similarly to t-SNE, but also for general non-linear dimension reduction. The algorithm is founded on three assumptions about the data:

1. The data is uniformly distributed on Riemannian manifold;
2. The Riemannian metric is locally constant (or can be approximated as such);
3. The manifold is locally connected.

From these assumptions it is possible to model the manifold with a fuzzy topological structure. The embedding is found by searching for a low dimensional projection of the data that has the closest possible equivalent fuzzy topological structure.

**Note:** the details for the underlying mathematics can be found [here](https://arxiv.org/abs/1802.03426).

To perform UMAP, we will use the [UMAP.jl](https://github.com/dillondaudert/UMAP.jl) package. First, let's calculate the correlation matrix between cars...

**Note:** `dims=2` means that correlation is calculated over rows/instances.
"""

# ╔═╡ bdaff924-889a-4dfc-93cc-4375f6016d82
L = cor(center_M, center_M, dims=2)

# ╔═╡ 3b892bcb-4c67-4309-be2c-5d562e4b98b5
md"""
Now, we can create an UMAP instance with two components...
"""

# ╔═╡ 5f56f6f7-f905-4271-a111-ba4092e3d12e
embed₁ = umap(L, 2)

# ╔═╡ 0cdb33ae-e168-42a3-974a-2a8cf0ccea8c
md"""
**Note:** the output components are being treated as rows (not columns).
"""

# ╔═╡ d70792c1-c3e4-48cb-8168-34cfa89ca0f4
begin
	Plots.scatter(embed₁[1,usaᵢ], embed₁[2,usaᵢ], color=1, label="USA")
	Plots.scatter!(embed₁[1,japᵢ], embed₁[2,japᵢ], color=2, label="Japan")
	Plots.scatter!(embed₁[1,eurᵢ], embed₁[2,eurᵢ], color=3, label="Europe")

	Plots.xlabel!("1st Component")
	Plots.ylabel!("2nd Component")
end

# ╔═╡ 6f34e24a-ef1f-4b33-a639-aec7a8a12aef
md"""
Instead of calculating the correlation matrix, we can calculate the pairwise Euclidean distance between instances using [Distances.jl](https://github.com/JuliaStats/Distances.jl) package...
"""

# ╔═╡ 08a89d70-7f5d-4a5e-8e4a-35abd03bb5b7
D = pairwise(Euclidean(), center_M, center_M, dims=1)

# ╔═╡ 0734fd78-fef5-4536-a99b-57eef9cd5f0f
md"""
Now, we can create again an UMAP instance with two components...
"""

# ╔═╡ d0ed58f2-93c1-4d4e-8f81-5f376c14ec41
embed₂ = umap(D, 2)

# ╔═╡ cef603ec-8e05-43cc-962e-11c763fe09af
begin
	Plots.scatter(embed₂[1,usaᵢ], embed₂[2,usaᵢ], color=1, label="USA")
	Plots.scatter!(embed₂[1,japᵢ], embed₂[2,japᵢ], color=2, label="Japan")
	Plots.scatter!(embed₂[1,eurᵢ], embed₂[2,eurᵢ], color=3, label="Europe")

	Plots.xlabel!("1st Component")
	Plots.ylabel!("2nd Component")
end

# ╔═╡ Cell order:
# ╟─efff10ac-824b-422b-b2f9-4074ea3437fb
# ╟─85e2d832-542c-49b0-a26d-b57a7a3d4404
# ╟─6e8c97d2-4d9a-4f00-99e1-69054903635c
# ╟─133df7ea-d3ec-41fa-bcb8-5eae4c853a55
# ╟─5271ffe0-f73a-430e-8ccc-5c6bd0937ce0
# ╠═6a53752f-3101-4b62-a661-c09c9df27f0f
# ╟─9a491a1d-32a0-496c-887a-cdfd49f01836
# ╠═dcc6939b-ba8e-4f4a-abcf-aa90efa60682
# ╟─56529677-8b5c-4912-b25a-728847b4f374
# ╟─59d179aa-a19c-497d-ae56-6e0b0b0007dc
# ╠═2f0ba288-19b9-49c8-845c-bdc2317328fc
# ╟─182b795f-c069-4ecd-aab6-89c66c93ee6e
# ╠═56e3357f-a692-4b29-8ca2-390ba62b9679
# ╟─53bae393-698c-4610-b18f-073a67d9bdc8
# ╠═193d7109-5499-40c3-a25f-b7b8560db5b9
# ╠═c3d8e866-c754-4ee6-8c1c-fab7f8048181
# ╠═231955b0-5ccd-46ad-b78d-96f861f1938a
# ╟─841ef716-8dbf-4aec-947e-6c0409a7fd64
# ╟─2c86f61a-b328-4eac-a4ff-1e713007ef54
# ╠═c08882ca-d617-4c0f-865f-898c0f639a82
# ╟─a9afbe75-29a4-4a54-b6c2-2786d74f7a65
# ╠═dffd6373-2e05-4746-bc44-ee2193ae2635
# ╟─d13d825e-e807-4ef0-a9ab-45fc994500a2
# ╠═cb2f6b28-d360-4bb8-a0c0-3d6d1b38e23d
# ╟─8d89f524-68a0-425e-a4e9-2525ebdd914d
# ╠═3946ec51-0336-4098-ba13-55abf68f9b41
# ╠═8d219680-c870-4549-9421-5c7fff3a7607
# ╟─a450f277-46cf-4f5b-b58a-b33b5aca5fb2
# ╠═2e2d14e9-8fdb-4328-9218-f47307309f40
# ╟─13b52e6a-dbe4-4dad-a7a6-260957fb0bdd
# ╠═49f05ab7-b716-44c4-bb44-6c2e700a18ac
# ╟─c37484f0-3e35-4c5f-841a-e0e6061752b4
# ╠═cf067613-8312-402c-88b2-fe507ce36ac9
# ╟─7594fb9c-443a-4ea8-a56a-f17c818ddbfe
# ╠═e0e09b17-da85-4879-bec8-adef9f125b31
# ╟─0e73cd39-24d3-4de4-a30d-a53ad6f1d50c
# ╠═5dfc1915-f832-4d72-b551-e3ddbfa693bc
# ╟─98c5d1d2-6c28-4c46-adeb-bb580d5dfe2a
# ╠═8117e9fa-23d1-4539-aa28-c5bc4a3db849
# ╠═07dbb182-4565-48de-b41c-bd0488f67280
# ╟─6bca80ac-8c1d-4f35-82e7-4c6c4206eb53
# ╠═152fc839-661b-48aa-a047-9e16a35acff5
# ╠═5f55ba88-c984-49d2-b4b9-e5cdd81e8c6c
# ╟─bac92bf8-3059-4fcc-b1a9-42c91637fd17
# ╠═6c163758-55b5-474b-8269-203884d70757
# ╠═4676988f-5687-4aad-a054-48ae5defd46e
# ╠═026668de-5323-4e40-b75c-c2eefbf46ef5
# ╟─7fb412b7-573e-4c5a-bd39-d9d7e637a9d4
# ╠═8faa2556-bcfc-4289-9d17-d75ad10f2245
# ╟─3114d2e5-e6eb-4198-98f3-4abfbb45bdeb
# ╟─7675ae48-a6ae-4189-990f-c1469ee11929
# ╠═bdaff924-889a-4dfc-93cc-4375f6016d82
# ╟─3b892bcb-4c67-4309-be2c-5d562e4b98b5
# ╠═5f56f6f7-f905-4271-a111-ba4092e3d12e
# ╟─0cdb33ae-e168-42a3-974a-2a8cf0ccea8c
# ╠═d70792c1-c3e4-48cb-8168-34cfa89ca0f4
# ╟─6f34e24a-ef1f-4b33-a639-aec7a8a12aef
# ╠═08a89d70-7f5d-4a5e-8e4a-35abd03bb5b7
# ╟─0734fd78-fef5-4536-a99b-57eef9cd5f0f
# ╠═d0ed58f2-93c1-4d4e-8f81-5f376c14ec41
# ╠═cef603ec-8e05-43cc-962e-11c763fe09af
