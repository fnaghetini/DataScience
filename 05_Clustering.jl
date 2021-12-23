### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ f1fb47b0-63ff-11ec-350c-7d4755d75c7b
begin
	using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
	Pkg.precompile()

	using PlutoUI
	using Clustering
	using VegaLite
	using VegaDatasets
	using DataFrames
	using Statistics
	using JSON
	using CSV
	using Distances
end;

# ╔═╡ cd0d79a3-2e64-4991-ade0-dac7422e323c
PlutoUI.TableOfContents(aside=true, indent=true, depth=3)

# ╔═╡ 55d15235-24d0-41c8-85e6-67f807a3b759
html"""
<p align="center">
	<img src="https://github.com/JuliaAcademy/DataScience/blob/main/datascience.png?raw=true" alt="Course-Logo" width="450px">
</p>
"""

# ╔═╡ 0ea01882-d01b-477f-bbb1-fc4f3042cc50
md"""
# 🗃 Clustering

**Cluster analysis** or **clustering** is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense) to each other than to those in other groups (clusters). It is a main task of exploratory data analysis, and a common technique for statistical data analysis, used in many fields.

Put simply, the task of clustering is to place observations that seem similar (in _feature space_) within the same cluster. Clustering is commonly used in two dimensional data where the goal is to create clusters based on coordinates. Here, we will use something similar. We will cluster houses based on their latitude-longitude locations using several different clustering methods.

[YouTube Link](https://www.youtube.com/watch?v=cwurgt7cn5s)
"""

# ╔═╡ fbfaeaf4-8e40-4c4e-b618-01a0bb16f9b9
md"""
## Load data

We will start off by getting some data. We will use data of 20,000+ California houses dataset. We will then learn whether house prices directly correlate with map location.
"""

# ╔═╡ 678ccf3f-aee0-45e9-b410-1efe156e2ef5
begin
	download("https://raw.githubusercontent.com/alexhegit/handson-ml2/master/datasets/housing/housing.csv", "newhouses.csv")
	houses = CSV.read("newhouses.csv", DataFrame)
end

# ╔═╡ d6fe53b6-a8bc-4bb7-aa2f-8e53bead237a
md"""
Let's check the available variables...
"""

# ╔═╡ 298b9aad-ed34-4cba-a74c-d103ffe1e347
names(houses)

# ╔═╡ 17d0f736-1faf-40e6-aa6a-17ef2e74bdb4
md"""
## Plot maps

We will use the [VegaLite.jl](https://github.com/queryverse/VegaLite.jl) package here for plotting. This package makes it very easy to plot information on a map. All you need is a JSON file of the map you intend to draw. Here, we will use the California counties JSON file and plot each house on the map and color code it via a heatmap of the price. This is done by the following line:

```julia
color="median_house_value:q"
```
"""

# ╔═╡ 6a3627e2-9995-4846-b11b-7631b5a2501e
begin
	# importing California counties shape
	cali_shape = JSON.parsefile("data/california-counties.json")
	VV = VegaDatasets.VegaJSONDataset(cali_shape, "california-counties.json")
end

# ╔═╡ 2f1c19e5-3719-4bc7-ab7a-80f1f6c53fff
begin
	# figure layout
	@vlplot(width=500, height=300) +
	# california counties shape
	@vlplot(mark={:geoshape,
				 fill=:black,
				 stroke=:white},
		   data={values=VV,
				 format={type=:topojson,
				 		 feature=:cb_2015_california_county_20m}},
		   projection={type=:albersUsa},) +
	# houses colored by values
	@vlplot(:circle,
			data=houses,
			projection={type=:albersUsa},
			longitude="longitude:q",
			latitude="latitude:q",
			size={value=12},
			color="median_house_value:q")
end

# ╔═╡ df1ab399-d07b-462f-ae51-598c2c9f7e05
md"""
One thing we will try and explore in this notebook is if clustering the houses has any direct relationship with their prices, so we will bucket the houses into intervals of $50,000 and re-perform the color codes based on each bucket.
"""

# ╔═╡ b6de785c-58c4-4c17-be08-f269791c4316
begin
	# insert bucket price column
	bucketprice = Int.(div.(houses[!,:median_house_value], 50000))
	insertcols!(houses, 3, :cprice => bucketprice)
end

# ╔═╡ 513e85d6-3480-4502-a6a8-cd7c9ed0077f
begin
	# figure layout
	@vlplot(width=500, height=300) +
	# california counties shape
	@vlplot(mark={:geoshape,
				 fill=:black,
				 stroke=:white},
		   data={values=VV,
				 format={type=:topojson,
				 		 feature=:cb_2015_california_county_20m}},
		   projection={type=:albersUsa},) +
	# houses colored by bucket price
	@vlplot(:circle,
			data=houses,
			projection={type=:albersUsa},
			longitude="longitude:q",
			latitude="latitude:q",
			size={value=12},
			color="cprice:n")
end

# ╔═╡ d2477d5e-ccc1-4e91-aec7-d6446961db1b
md"""
## k-Means Clustering

**k-Means Clustering** is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster. This results in a partitioning of the data space into Voronoi cells.

k-Means Clustering minimizes within-cluster variances (squared Euclidean distances), but not regular Euclidean distances, which would be the more difficult Weber problem: the mean optimizes squared errors, whereas only the geometric median minimizes Euclidean distances. For instance, better Euclidean solutions can be found using k-medians and k-medoids.

**Note:** for more math details about k-Means, check this [link](https://en.wikipedia.org/wiki/K-means_clustering).

First, we will apply k-Means to cluster coordinates into 10 cluesters and then we will insert a new column `cluster_means`.

**Note:** k-Means requires that observations must be represented as columns (not rows).
"""

# ╔═╡ c8e16e6c-b99e-405d-b0e4-0148e09cc818
begin
	# DataFrame of coordinates
	X = houses[!, [:latitude, :longitude]]
	# k-means model
	KM = kmeans(Matrix(X)', 10)
end

# ╔═╡ ece5be38-b8ba-4615-8755-6c05c9e6e77f
insertcols!(houses, 3, :cluster_means => KM.assignments)

# ╔═╡ d8171788-16dd-4f0b-95da-86b7a35fab92
md"""
Now we can visualize the data colored by the new column `cluster_means`...
"""

# ╔═╡ 0073e46e-0350-4f9c-975c-31355d9d3040
begin
	# figure layout
	@vlplot(width=500, height=300) +
	# california counties shape
	@vlplot(mark={:geoshape,
				 fill=:black,
				 stroke=:white},
		   data={values=VV,
				 format={type=:topojson,
				 		 feature=:cb_2015_california_county_20m}},
		   projection={type=:albersUsa},) +
	# houses colored by cluster_means column
	@vlplot(:circle,
			data=houses,
			projection={type=:albersUsa},
			longitude="longitude:q",
			latitude="latitude:q",
			size={value=12},
			color="cluster_means:n")
end

# ╔═╡ 65ece652-2caa-4c9f-a607-b19a7087a065
md"""
As we can see, location indeed affects price of house. we may want to know if this relationship remains true for location as in proximity to water, downtown or bus stops.
"""

# ╔═╡ f9ab7250-a2db-4824-9fda-b68f3fcb582e
md"""
## k-Medoids Clustering

The **k-Medoids** problem is a clustering problem similar to k-Means. The name was coined by Leonard Kaufman and Peter J. Rousseeuw with their PAM algorithm. Both the k-Means and k-Medoids algorithms are partitional (breaking the dataset up into groups) and attempt to minimize the distance between points labeled to be in a cluster and a point designated as the center of that cluster.

In contrast to the k-Means algorithm, k-Medoids chooses actual data points as centers (medoids or exemplars), and thereby allows for greater interpretability of the cluster centers than in k-Means, where the center of a cluster is not necessarily one of the input data points (it is the average between the points in the cluster). 

Furthermore, k-Medoids can be used with arbitrary dissimilarity measures, whereas k-Means generally requires Euclidean distance for efficient solutions. Because k-Medoids minimizes a sum of pairwise dissimilarities instead of a sum of squared Euclidean distances, it is more robust to noise and outliers than k-Means.

**Note:** for more math details about k-Medoids, check this [link](https://en.wikipedia.org/wiki/K-medoids).

For k-Medoids, we need to build a distance matrix as we did in the last notebook (UMAP for dimensionality reduction). We will use the [Distances.jl](https://github.com/JuliaStats/Distances.jl) package for this purpose and compute the pairwise Euclidean distances.
"""

# ╔═╡ 2d933b98-83c3-4f9e-9858-cd832cb6b457
begin
	# transpose matrix
	Xᵗ = Matrix(X)'

	# calculating Euclidean distance matrix
	D = pairwise(Euclidean(), Xᵗ, Xᵗ, dims=2)
end

# ╔═╡ 6a42ec90-b400-4638-91aa-4b982c2da069
# k-Medoids model
KMD = kmedoids(D, 10)

# ╔═╡ 8f55e8e6-37b2-4d1d-bb0c-db2c83f854be
# inserting k-Medoids column
insertcols!(houses, 3, :cluster_medoids => KMD.assignments)

# ╔═╡ 1dd518f3-3c63-4610-9736-7369f980392a
md"""
Now we can visualize the data colored by the new column `cluster_medoids`...
"""

# ╔═╡ 0ea39069-dbbf-4bd9-becb-560cdbeb9e8e
begin
	# figure layout
	@vlplot(width=500, height=300) +
	# california counties shape
	@vlplot(mark={:geoshape,
				 fill=:black,
				 stroke=:white},
		   data={values=VV,
				 format={type=:topojson,
				 		 feature=:cb_2015_california_county_20m}},
		   projection={type=:albersUsa},) +
	# houses colored by cluster_medoids column
	@vlplot(:circle,
			data=houses,
			projection={type=:albersUsa},
			longitude="longitude:q",
			latitude="latitude:q",
			size={value=12},
			color="cluster_medoids:n")
end

# ╔═╡ 33eed20d-9961-4e96-9997-e2866c20e097
md"""
## Hierarchical Clustering

**Hierarchical Clustering (HCA)** is a method of cluster analysis which seeks to build a hierarchy of clusters. Strategies for hierarchical clustering generally fall into two types:

- **Agglomerative**: This is a "bottom-up" approach: each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.

- **Divisive**: This is a "top-down" approach: all observations start in one cluster, and splits are performed recursively as one moves down the hierarchy.

In general, the merges and splits are determined in a greedy manner. The results of Hierarchical Clustering are usually presented in a dendrogram.

**Note:** for more math details about Hierarchical Clustering, check this [link](https://en.wikipedia.org/wiki/Hierarchical_clustering).

First, we will input the distance matrix `D` to `hclust` function and then generate 10 clusters using the `cutree` function. Finally, we will add the new `cluster_hca` column to our data.
"""

# ╔═╡ 177f94ef-6b45-4324-9b8a-6b07dc853e54
# HCA
HCA = hclust(D)

# ╔═╡ a5a2a363-9ae7-44d8-89cc-3aaaf7aa3d9e
# generating clusters
hca_clusters = cutree(HCA; k=10)

# ╔═╡ 56f2d129-6a8a-4750-b292-1682f7387165
# inserting k-Medoids column
insertcols!(houses, 3, :cluster_hca => hca_clusters)

# ╔═╡ bdc23b6c-b686-4971-984d-63c2154e661e
md"""
Now we can visualize the data colored by the new column `cluster_hca`...
"""

# ╔═╡ f4fc1f80-601f-4075-8c6d-16cb6e9f0223
begin
	# figure layout
	@vlplot(width=500, height=300) +
	# california counties shape
	@vlplot(mark={:geoshape,
				 fill=:black,
				 stroke=:white},
		   data={values=VV,
				 format={type=:topojson,
				 		 feature=:cb_2015_california_county_20m}},
		   projection={type=:albersUsa},) +
	# houses colored by cluster_hca column
	@vlplot(:circle,
			data=houses,
			projection={type=:albersUsa},
			longitude="longitude:q",
			latitude="latitude:q",
			size={value=12},
			color="cluster_hca:n")
end

# ╔═╡ b6091ca6-873c-4cc2-8ee2-30d884db514f
md"""
## DBSCAN

**Density-based Spatial Clustering of Applications with Noise (DBSCAN)** is a data clustering algorithm proposed by Martin Ester, Hans-Peter Kriegel, Jörg Sander and Xiaowei Xu in 1996. It is a density-based clustering non-parametric algorithm: given a set of points in some space, it groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away).

DBSCAN is one of the most common clustering algorithms and also most cited in scientific literature.

**Note:** for more math details about DBSCAN, check this [link](https://en.wikipedia.org/wiki/DBSCAN).

We just need to pass the distance matrix `D` to perform DBSCAN. We may also set both `eps` (the radius of a point neighborhood) and `minpts` (the minimum number of neighboring points to qualify a point as a density point) arguments.

**Note:** we cannot pass the number of clusters when using DBSCAN.
"""

# ╔═╡ 35702907-ecec-4441-a6c6-3497b02bfd51
# creating a new distance matrix (Square of Euclidean distance)
D² = pairwise(SqEuclidean(), Xᵗ, dims=2)

# ╔═╡ 431648b8-406b-4064-b42f-1bb7c0721026
# DBSCAN
DBS = dbscan(D², 0.05, 10)

# ╔═╡ 5f4aba99-22c4-43b9-be6e-ccc45f349158
# inserting DBSCAN column
insertcols!(houses, 3, :cluster_dbscan => DBS.assignments)

# ╔═╡ 7db3222d-eab6-4ddd-8d5e-7990f437a85e
md"""
Now we can visualize the data colored by the new column `cluster_dbscan`...
"""

# ╔═╡ b632e115-6278-4382-9d1b-1d708d40ac9a
begin
	# figure layout
	@vlplot(width=500, height=300) +
	# california counties shape
	@vlplot(mark={:geoshape,
				 fill=:black,
				 stroke=:white},
		   data={values=VV,
				 format={type=:topojson,
				 		 feature=:cb_2015_california_county_20m}},
		   projection={type=:albersUsa},) +
	# houses colored by cluster_dbscan column
	@vlplot(:circle,
			data=houses,
			projection={type=:albersUsa},
			longitude="longitude:q",
			latitude="latitude:q",
			size={value=12},
			color="cluster_dbscan:n")
end

# ╔═╡ Cell order:
# ╟─f1fb47b0-63ff-11ec-350c-7d4755d75c7b
# ╟─cd0d79a3-2e64-4991-ade0-dac7422e323c
# ╟─55d15235-24d0-41c8-85e6-67f807a3b759
# ╟─0ea01882-d01b-477f-bbb1-fc4f3042cc50
# ╟─fbfaeaf4-8e40-4c4e-b618-01a0bb16f9b9
# ╠═678ccf3f-aee0-45e9-b410-1efe156e2ef5
# ╟─d6fe53b6-a8bc-4bb7-aa2f-8e53bead237a
# ╠═298b9aad-ed34-4cba-a74c-d103ffe1e347
# ╟─17d0f736-1faf-40e6-aa6a-17ef2e74bdb4
# ╠═6a3627e2-9995-4846-b11b-7631b5a2501e
# ╠═2f1c19e5-3719-4bc7-ab7a-80f1f6c53fff
# ╟─df1ab399-d07b-462f-ae51-598c2c9f7e05
# ╠═b6de785c-58c4-4c17-be08-f269791c4316
# ╠═513e85d6-3480-4502-a6a8-cd7c9ed0077f
# ╟─d2477d5e-ccc1-4e91-aec7-d6446961db1b
# ╠═c8e16e6c-b99e-405d-b0e4-0148e09cc818
# ╠═ece5be38-b8ba-4615-8755-6c05c9e6e77f
# ╟─d8171788-16dd-4f0b-95da-86b7a35fab92
# ╠═0073e46e-0350-4f9c-975c-31355d9d3040
# ╟─65ece652-2caa-4c9f-a607-b19a7087a065
# ╟─f9ab7250-a2db-4824-9fda-b68f3fcb582e
# ╠═2d933b98-83c3-4f9e-9858-cd832cb6b457
# ╠═6a42ec90-b400-4638-91aa-4b982c2da069
# ╠═8f55e8e6-37b2-4d1d-bb0c-db2c83f854be
# ╟─1dd518f3-3c63-4610-9736-7369f980392a
# ╠═0ea39069-dbbf-4bd9-becb-560cdbeb9e8e
# ╟─33eed20d-9961-4e96-9997-e2866c20e097
# ╠═177f94ef-6b45-4324-9b8a-6b07dc853e54
# ╠═a5a2a363-9ae7-44d8-89cc-3aaaf7aa3d9e
# ╠═56f2d129-6a8a-4750-b292-1682f7387165
# ╟─bdc23b6c-b686-4971-984d-63c2154e661e
# ╠═f4fc1f80-601f-4075-8c6d-16cb6e9f0223
# ╟─b6091ca6-873c-4cc2-8ee2-30d884db514f
# ╠═35702907-ecec-4441-a6c6-3497b02bfd51
# ╠═431648b8-406b-4064-b42f-1bb7c0721026
# ╠═5f4aba99-22c4-43b9-be6e-ccc45f349158
# ╟─7db3222d-eab6-4ddd-8d5e-7990f437a85e
# ╠═b632e115-6278-4382-9d1b-1d708d40ac9a
