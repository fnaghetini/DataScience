### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ 1c630e40-628c-11ec-1793-05688dabf6c9
begin
	using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
	Pkg.precompile()

	using PlutoUI
	using Statistics
	using StatsBase
	using RDatasets
	using Plots
	using StatsPlots
	using KernelDensity
	using Distributions
	using LinearAlgebra
	using HypothesisTests
	using PyCall
	using MLBase
end;

# ╔═╡ 7242a33b-ec37-48d6-afca-8c8387a45a83
PlutoUI.TableOfContents(aside=true, indent=true, depth=3)

# ╔═╡ dd2aa8b3-9da3-418d-bf31-74574483360d
html"""
<p align="center">
	<img src="https://github.com/JuliaAcademy/DataScience/blob/main/datascience.png?raw=true" alt="Course-Logo" width="450px">
</p>
"""

# ╔═╡ abc828f1-fd14-4329-8b47-f4135300f937
md"""
# 📊 Statistics

Having a solid understanding of statistics in data science allows us to understand our data better, and allows us to create a quantifiable evaluation of any future conclusions.

[YouTube Link](https://www.youtube.com/watch?v=AAGxWEJ_eWk)
"""

# ╔═╡ a8aae3d4-198c-482f-a972-05be31be39c1
md"""
## Load dataset

In this notebook, we will use eruption data on the faithful geyser (Yellowstone, USA). The data will contain wait times between every consecutive times the geyser goes off (in minutes) and the length of the eruptions (also in minutes).

Let's get the data first using the [RDatasets.jl](https://github.com/JuliaStats/RDatasets.jl) package...

**Note:** the dataset will be loaded as a `DataFrame`.
"""

# ╔═╡ 63420679-9b77-4412-ba1c-fc471da6b1e9
data = dataset("datasets", "faithful")

# ╔═╡ 9d56b2cc-c4c6-4ce2-8d9c-33174fe5bc33
md"""
We can use the `describe` function to get some stats from our data...
"""

# ╔═╡ 67ac4a1d-eb15-473f-a548-07425fd627d8
describe(data)

# ╔═╡ 5ee886cc-2730-40cf-8497-507dcdc08c5f
md"""
Let's call each column (feature) separately...
"""

# ╔═╡ 110fda28-1b77-44d6-8485-a4ec6e012011
begin
	eruption = data[!, :Eruptions]
	# eruption = data.Eruptions

	waittime = data[!, :Waiting]
	# waittime = data.Waiting
end;

# ╔═╡ 6e093fbb-1e3a-4131-b3c2-b5891e734fa7
md"""
Now, we can plot both variables in the same scatter plot...
"""

# ╔═╡ ed9da9c2-53e4-45d6-9ec1-02c8bf54c1a5
begin
	scatter(eruption, label="Eruptions")
	scatter!(waittime, label="Wait Time", xlabel="Index", ylabel="Time (minutes)")
end

# ╔═╡ 13137e30-cfde-459a-adb4-1ddc07e626e8
md"""
## Statistics plots

As you can see, this doesn't tell us much about the data... Let's try some statistical plots.
"""

# ╔═╡ ff9d63e7-808a-4465-a68d-659a222afbae
md"""
### Boxplot
"""

# ╔═╡ c3672771-248a-47f9-a019-2de03cdd79f4
boxplot(["Eruption Length"], eruption, legend=false,
		size=(200,400), whisker_width=1, ylabel="Time (minutes)")

# ╔═╡ 46b55947-428a-483f-a00f-a232e5329fbd
md"""
Statistical plots such as a boxplot (and a violin plot), can provide a much better understanding of the data. Here, we immediately see that the median time of each eruption is about 4 minutes.
"""

# ╔═╡ e0933f5c-50b5-47cf-bae8-7a45a219fefb
md"""
### Histogram

The next plot we will see is a histogram plot. Check its arguments below...

- `x`: AbstractVector of values to be binned.
- `bins`: Integer, NTuple{2,Integer}, AbstractVector or Symbol. Default is :auto (the Freedman-Diaconis rule). For histogram-types, defines the approximate number of bins to aim for, or the auto-binning algorithm to use (`:sturges`, `:sqrt`, `:rice`, `:scott` or `:fd`). For fine-grained control pass a Vector of break values, e.g. `range(minimum(x), stop = maximum(x), length = 25)`.
- `weights`: Vector of weights for the values in x, for weighted bin counts.
- `normalize`: Bool or Symbol. Histogram normalization mode. Possible values are: `false`/`:none` (no normalization, default), `true`/`:pdf` (normalize to a discrete Probability Density Function, where the total area of the bins is 1), `:probability` (bin heights sum to 1) and `:density` (the area of each bin, rather than the height, is equal to the counts - useful for uneven bin sizes).
- `bar_position`: Symbol. Choose from `:overlay` (default), `:stack`. (warning: May not be implemented fully).
- `bar_width`: nothing or Number. Width of bars in data coordinates. When nothing, chooses based on x (or y when orientation = `:h`).
- `bar_edges`: Bool. Align bars to edges (true), or centers (the default).
- `orientation`: Symbol. Horizontal or vertical orientation for bar types. Values `:h`, `:hor`, `:horizontal` correspond to horizontal (sideways, anchored to y-axis), and `:v`, `:vert`, and `:vertical` correspond to vertical (the default).

Let's plot eruption length data...
"""

# ╔═╡ 943c8862-b909-4750-96da-7bda62c5fc8c
histogram(eruption, bins=:sqrt, label="Eruption", color=:orange,
		  xlabel="Eruption Length (minutes)", ylabel="Frequency")

# ╔═╡ ed480ae6-3302-4d0e-b83e-53a9c395d5ce
md"""
### Kernel density estimates

Next, we will see how we can fit a kernel density estimation function to our data. We will make use of the [KernelDensity.jl](https://github.com/JuliaStats/KernelDensity.jl) package.

With Kernel density estimates, we can see how things are changing...
"""

# ╔═╡ ed58e41f-3f73-40c1-92ec-5f2845c62b97
ρ = kde(eruption)

# ╔═╡ 952506dc-6cd8-4ecd-ace8-b9d914560289
md"""
If we want the histogram and the kernel density graph to be aligned we need to remember that the "density contribution" of every point added to one of these histograms is `density * nb of elements * bin width`. We do this, since KDE are densities, while the y-axis is showing the frequency.

**Note:** Read more about kernel density estimates [here](https://en.wikipedia.org/wiki/Kernel_density_estimation).
"""

# ╔═╡ 3ea44987-7ba7-4df6-a82a-c11a98ea784d
# the range of density data (x-axis)
ρ.x

# ╔═╡ 31274ae3-20da-4f52-9153-6f6b4d4472cb
# density data (y-axis)
ρ.density

# ╔═╡ 53f3e80a-2c88-405f-9131-21d07b132389
# density * nb of elements * bin width
scaled_density = ρ.density .* length(eruption) .* 0.2

# ╔═╡ ccfe5310-cc85-457c-bfbc-9671062b9088
begin
	histogram(eruption, bins=:sqrt, label="Eruption", color=:lightblue,
		  	  xlabel="Eruption Length (minutes)", ylabel="Frequency")

	plot!(ρ.x, scaled_density, linewidth=1.5, color=:black, label="KDE fit")
end

# ╔═╡ cbf471e5-523c-4550-835b-5c45b9648d8c
md"""
## Probability distributions

First, we will take a look at one probablity distribution, namely the Normal Distribution and verify that it generates a bell curve.
"""

# ╔═╡ 914412a4-69d4-4d9f-b575-03cfa03a7873
# generating random normal data
normaldata = randn(100_000)

# ╔═╡ 86c9cf10-2ade-45de-8649-a8dd5b2ae586
begin
	# generating the KDE
	ρₙ = kde(normaldata)

	# scaling density estimates
	scaled_ρₙ = ρₙ.density .* length(normaldata) * 0.1
end

# ╔═╡ e137810d-9e3c-43c8-905c-4efc2213a130
begin
	# plotting histogram
	histogram(normaldata, color=:lightblue, label="Z ~ N(0,1)",
			  xlabel="Z", ylabel="Frequency")

	# plotting KDE
	plot!(ρₙ.x, scaled_ρₙ, linewidth=3, color=:black, label="KDE fit")
end

# ╔═╡ f6430d83-a92b-4c03-a07e-8a6146cc9e1c
md"""
Another way to generate the same plot is via using the [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) package and choosing the probability distribution you want, and then drawing random numbers from it.
"""

# ╔═╡ 4ffd9a7a-c62c-40a4-8ec1-4e2351a2a66a
md"""
### Normal distribution

As an example, we will use `N = Normal()` below to draw random numbers from a [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution). The default $\mu$ and $\sigma$ values are 0 and 1 (standard Normal distribution).
"""

# ╔═╡ 978e2f4c-b411-459c-8024-4f6205484f63
begin
	# drawing numbers ~N(0,1)
	N = Normal()
	gaussdata = rand(N, 100_000)

	# calculating KDE
	rhoN = kde(gaussdata)
	scaled_rhoN = rhoN.density .* length(gaussdata) .* 0.1
end

# ╔═╡ d1483a5e-0b7e-4317-adb0-8f024fad21eb
begin
	# plotting histogram
	histogram(gaussdata, color=:lightblue, label="Z ~ N(0,1)",
			  xlabel="Z", ylabel="Frequency")

	# plotting KDE
	plot!(rhoN.x, scaled_rhoN, linewidth=3, color=:black, label="KDE fit")
end

# ╔═╡ 62470113-95dc-4b92-9e1d-3118efa0bd08
md"""
### Binomial distribution

Now, let's draw random numbers from a [Binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution).
"""

# ╔═╡ 7749b6b5-cf01-4812-b1ce-6fe3204b0bcf
begin
	# drawing numbers ~B with n=40 trials
	B = Binomial(40)
	binomdata = rand(B, 100_000)

	# calculating KDE
	rhoB = kde(binomdata)
	scaled_rhoB = rhoB.density .* length(binomdata) .* 0.5
end

# ╔═╡ ff68a550-9cc6-416e-96b8-07de164e5fc4
begin
	# plotting histogram
	histogram(binomdata, color=:lightblue, label="Z ~ B(40,0.5)",
			  xlabel="Z", ylabel="Frequency")

	# plotting KDE
	plot!(rhoB.x, scaled_rhoB, linewidth=3, color=:black, label="KDE fit")
end

# ╔═╡ 693de4f8-88b1-476f-9319-64edafb049af
md"""
### Fit data to a distribution

Next, we will try to fit a given set of numbers to a parametric distribution. Thus, we are going to use the StatsBase.jl `fit` function.
"""

# ╔═╡ 7f9949d9-22e2-4d4e-92ec-6e52e01d10aa
begin
	# creating 1000 random values
	randomdata = rand(1000)
	# fitting random values to a Standard Normal distribution
	fitted_distrib = fit(Normal, randomdata)
	# drawing 1000 values from the new fitted distribution
	new_data = rand(fitted_distrib, 1000)
end

# ╔═╡ 98301aae-61ca-46ab-936a-fbb310854778
# note that `fitted_distrib` is a `Distribution`
fitted_distrib

# ╔═╡ f270c785-3a7a-4e43-9f21-b9a6e199383b
begin
	# plotting new data (Normal distribution)
	histogram(new_data, nbins=20, fillalpha=0.3, label="Normal Data")

	# plotting original data
	histogram!(randomdata, nbins=20, linecolor=:red, fillalpha=0.3,
			   label="Original Data")
end

# ╔═╡ 2d5e5740-a3c6-4448-b79f-f231a0ac9d96
md"""
We can try to fit eruptions data to a Normal distribution, although eruptions distribution does not seem to be Normal.
"""

# ╔═╡ ef651442-e846-4a5e-9f3b-0da63afeca16
begin
	erup_distrib = fit(Normal, eruption)
	normal_erup = rand(erup_distrib, 1000)
end

# ╔═╡ a5a54536-4b60-4a9c-b423-2804cce37382
erup_distrib

# ╔═╡ 25b15bda-9bb4-4ed9-90e4-c1f937051314
begin
	# plotting fitted data (Normal distribution)
	histogram(normal_erup, nbins=20, fillalpha=0.3, label="Normal Data")

	# plotting original data
	histogram!(eruption, nbins=20, linecolor=:red, fillalpha=0.3,
			   label="Original Data")
end

# ╔═╡ b82db811-d97d-4efd-81bc-a48780b23dcb
md"""
## Hypothesis testing

Next, we will perform hypothesis testing using the [HypothesisTests.jl](https://github.com/JuliaStats/HypothesisTests.jl) package.

Let's try to use the `OneSampleTTest` function to perform a one sample t-test of random sampled values.

The one-sample t-test is a statistical hypothesis test used to determine whether an unknown population mean is different from a specific value. In this case, the default value under $H_0$ is 0. 
"""

# ╔═╡ af54608d-209f-4d16-9573-b4b8634d2acf
# generating 1000 random normal values
hypdata = randn(1000)

# ╔═╡ 867dab59-1c8d-44d9-9add-9f3999f1b86c
# performing one sample t-test
OneSampleTTest(hypdata)

# ╔═╡ 47c7f6df-0696-4b1e-b6c2-7eb8e782eb82
md"""
**Note:** as we can see above, we cannot reject the null hypothesis $H_0$. This was expected, since we drew sample values from a Standard Normal distribution, with $\mu=0$.

Now we can perform a one sample t-test of eruption data. We will probably be able to reject $H_0$, since the eruption length mean is close to 3.5 minutes...
"""

# ╔═╡ 6555c2a2-227f-419b-9b8b-d563767f3ea5
OneSampleTTest(eruption)

# ╔═╡ 78585d9d-5696-41dd-80f0-abab1585b008
md"""
## Correlation statistics

First of all, let's see visually if there is a correlation between `eruption` and `waittime` variables using a scatter plot...
"""

# ╔═╡ 650a1af5-b413-4794-bf51-77fa34cfb820
scatter(eruption, waittime, legend=false, color=:orange,
		xlabel="Eruption Length (minutes)", markersize=3,
		ylabel = "Time between Eruptions (minutes)")

# ╔═╡ 80db913d-faa8-43d5-9794-e17b02823cc5
md"""
These two variables may have a strong positive correlation. We can calculate both `Pearson` and `Spearman` correlation coefficients to check that.
"""

# ╔═╡ fcb84ac7-3bca-4284-8939-a55bd78a02fa
begin
	spearman_r = corspearman(eruption, waittime)
	"r(Spearman) = $spearman_r"
end

# ╔═╡ 960b8739-10d1-490d-ac49-e51527386161
begin
	pearson_r = cor(eruption, waittime)
	"r(Pearson) = $pearson_r"
end

# ╔═╡ 5bd50a69-4edb-435a-9263-e0b5d47851aa
md"""
Interesting! This means that the next time you visit Yellowstone National part ot see the faithful geysser and you have to wait for too long for it to go off, you will likely get a longer eruption!

Currently we are using the p-value of Spearman and Pearson correlation coefficients from Python. But you can follow the formula [here](https://stackoverflow.com/questions/53345724/how-to-use-julia-to-compute-the-pearson-correlation-coefficient-with-p-value) to implement your own.

Let's import the `scipy.stats` Python module.

**Note:** sometimes there are some issues getting Python and Julia to communicate as desired. We can explicity add `scipy.stats` using the [Conda.jl](https://github.com/JuliaPy/Conda.jl) package:

```julia
julia> using Conda
julia> Conda.add("scipy")
```
"""

# ╔═╡ d1ef6cd2-fda1-47f6-af12-8492ea70a539
scipy_stats = pyimport("scipy.stats");

# ╔═╡ c5f77715-02cb-437a-a45f-3f3a4dfba67b
begin
	rₛ, pvalueₛ = scipy_stats.spearmanr(eruption, waittime)
	"r(Spearman) = $(rₛ) || p-value = $(pvalueₛ)"
end

# ╔═╡ 615e7c40-bafe-475f-959d-29c5f5e845e9
begin
	rₚ, pvalueₚ = scipy_stats.pearsonr(eruption, waittime)
	"r(Pearson) = $(rₚ) || p-value = $(pvalueₚ)"
end

# ╔═╡ 32bc5a24-638e-43bc-a5c1-bd0a85db7849
md"""
## Confusion matrix & AUC

Finally, we will cover basic tools you will need such as confusion matrix and AUC scores. We use the [MLBase.jl](https://github.com/JuliaStats/MLBase.jl) package for that.

First, let's calculate a confusion matrix, given a target vector `y₁` and a prediction vector `ŷ₁`...
"""

# ╔═╡ 838a8a32-95f6-48c0-b87d-28a66ef431c5
begin
	# generating target and prediction vectors
	y₁ = [1, 1, 1, 1, 1, 1, 1, 2]
	yhat₁ = [1, 1, 2, 2, 1, 1, 1, 1]
end;

# ╔═╡ 68d7cd6c-bd76-406b-a3d7-4528ee9671fd
# compute default confusion matrix
C = confusmat(2, y₁, yhat₁)

# ╔═╡ 2bcf8647-368d-4b74-8e1a-4a4b1a577fdc
md"""
We may normalize cell values per class...
"""

# ╔═╡ 86ad3850-f631-42b2-8d19-bf20cba89c6b
C ./ sum(C, dims=2)

# ╔═╡ 38a2bb94-1d8f-4e1a-a07f-3bc1332301fd
md"""
Also, we may compute the accuracy (correct rate) from confusion matrix...

```math
accuracy = \frac{TP}{TP+TN+FP+FN}
```
"""

# ╔═╡ ea2c50c6-9404-4ba4-8fe7-4d1e1b3246ff
acc = sum(diag(C)) / length(y₁)

# ╔═╡ 70cbc1f9-171e-4664-87d1-6e73f0bc841c
md"""
Instead of calculating accuracy manually, we could use the `correctrate` function...
"""

# ╔═╡ 188c0219-8a54-4f28-95fd-9f2f525c12be
acc == correctrate(y₁, yhat₁)

# ╔═╡ 063d30ad-0ac4-4d50-b66b-9b06bf155fd8
md"""
Now, we can use the `roc` function to calculate ROC scores...
"""

# ╔═╡ f3cca2ef-17ec-44e0-996d-8a0cb2f2fcb1
begin
	# generating target and prediction vectors
	y₂ = [1, 1, 1, 1, 1, 1, 1, 0]
	yhat₂ = [1, 1, 0, 0, 1, 1, 1, 1]
end;

# ╔═╡ 9af05d0f-5daf-48f4-9949-79db70ae4bbc
ROC = MLBase.roc(y₂, yhat₂)

# ╔═╡ 92f98fe2-9154-4029-8282-45dee3f11d31
md"""
We may also use both `recall` and `precision` functions to compute these metrics...
"""

# ╔═╡ 0cfa5643-b65b-4374-a227-8f036ce32bbf
begin
	rec = recall(ROC)
	"Recall = $rec"
end

# ╔═╡ 93ea2a29-bac3-4629-bac5-5d03e8e45d80
begin
	prec = precision(ROC)
	"Precision = $prec"
end

# ╔═╡ Cell order:
# ╟─1c630e40-628c-11ec-1793-05688dabf6c9
# ╟─7242a33b-ec37-48d6-afca-8c8387a45a83
# ╟─dd2aa8b3-9da3-418d-bf31-74574483360d
# ╟─abc828f1-fd14-4329-8b47-f4135300f937
# ╟─a8aae3d4-198c-482f-a972-05be31be39c1
# ╠═63420679-9b77-4412-ba1c-fc471da6b1e9
# ╟─9d56b2cc-c4c6-4ce2-8d9c-33174fe5bc33
# ╠═67ac4a1d-eb15-473f-a548-07425fd627d8
# ╟─5ee886cc-2730-40cf-8497-507dcdc08c5f
# ╠═110fda28-1b77-44d6-8485-a4ec6e012011
# ╟─6e093fbb-1e3a-4131-b3c2-b5891e734fa7
# ╠═ed9da9c2-53e4-45d6-9ec1-02c8bf54c1a5
# ╟─13137e30-cfde-459a-adb4-1ddc07e626e8
# ╟─ff9d63e7-808a-4465-a68d-659a222afbae
# ╠═c3672771-248a-47f9-a019-2de03cdd79f4
# ╟─46b55947-428a-483f-a00f-a232e5329fbd
# ╟─e0933f5c-50b5-47cf-bae8-7a45a219fefb
# ╠═943c8862-b909-4750-96da-7bda62c5fc8c
# ╟─ed480ae6-3302-4d0e-b83e-53a9c395d5ce
# ╠═ed58e41f-3f73-40c1-92ec-5f2845c62b97
# ╟─952506dc-6cd8-4ecd-ace8-b9d914560289
# ╠═3ea44987-7ba7-4df6-a82a-c11a98ea784d
# ╠═31274ae3-20da-4f52-9153-6f6b4d4472cb
# ╠═53f3e80a-2c88-405f-9131-21d07b132389
# ╠═ccfe5310-cc85-457c-bfbc-9671062b9088
# ╟─cbf471e5-523c-4550-835b-5c45b9648d8c
# ╠═914412a4-69d4-4d9f-b575-03cfa03a7873
# ╠═86c9cf10-2ade-45de-8649-a8dd5b2ae586
# ╠═e137810d-9e3c-43c8-905c-4efc2213a130
# ╟─f6430d83-a92b-4c03-a07e-8a6146cc9e1c
# ╟─4ffd9a7a-c62c-40a4-8ec1-4e2351a2a66a
# ╠═978e2f4c-b411-459c-8024-4f6205484f63
# ╠═d1483a5e-0b7e-4317-adb0-8f024fad21eb
# ╟─62470113-95dc-4b92-9e1d-3118efa0bd08
# ╠═7749b6b5-cf01-4812-b1ce-6fe3204b0bcf
# ╠═ff68a550-9cc6-416e-96b8-07de164e5fc4
# ╟─693de4f8-88b1-476f-9319-64edafb049af
# ╠═7f9949d9-22e2-4d4e-92ec-6e52e01d10aa
# ╠═98301aae-61ca-46ab-936a-fbb310854778
# ╠═f270c785-3a7a-4e43-9f21-b9a6e199383b
# ╟─2d5e5740-a3c6-4448-b79f-f231a0ac9d96
# ╠═ef651442-e846-4a5e-9f3b-0da63afeca16
# ╠═a5a54536-4b60-4a9c-b423-2804cce37382
# ╠═25b15bda-9bb4-4ed9-90e4-c1f937051314
# ╟─b82db811-d97d-4efd-81bc-a48780b23dcb
# ╠═af54608d-209f-4d16-9573-b4b8634d2acf
# ╠═867dab59-1c8d-44d9-9add-9f3999f1b86c
# ╟─47c7f6df-0696-4b1e-b6c2-7eb8e782eb82
# ╠═6555c2a2-227f-419b-9b8b-d563767f3ea5
# ╟─78585d9d-5696-41dd-80f0-abab1585b008
# ╠═650a1af5-b413-4794-bf51-77fa34cfb820
# ╟─80db913d-faa8-43d5-9794-e17b02823cc5
# ╠═fcb84ac7-3bca-4284-8939-a55bd78a02fa
# ╠═960b8739-10d1-490d-ac49-e51527386161
# ╟─5bd50a69-4edb-435a-9263-e0b5d47851aa
# ╠═d1ef6cd2-fda1-47f6-af12-8492ea70a539
# ╠═c5f77715-02cb-437a-a45f-3f3a4dfba67b
# ╠═615e7c40-bafe-475f-959d-29c5f5e845e9
# ╟─32bc5a24-638e-43bc-a5c1-bd0a85db7849
# ╠═838a8a32-95f6-48c0-b87d-28a66ef431c5
# ╠═68d7cd6c-bd76-406b-a3d7-4528ee9671fd
# ╟─2bcf8647-368d-4b74-8e1a-4a4b1a577fdc
# ╠═86ad3850-f631-42b2-8d19-bf20cba89c6b
# ╟─38a2bb94-1d8f-4e1a-a07f-3bc1332301fd
# ╠═ea2c50c6-9404-4ba4-8fe7-4d1e1b3246ff
# ╟─70cbc1f9-171e-4664-87d1-6e73f0bc841c
# ╠═188c0219-8a54-4f28-95fd-9f2f525c12be
# ╟─063d30ad-0ac4-4d50-b66b-9b06bf155fd8
# ╠═f3cca2ef-17ec-44e0-996d-8a0cb2f2fcb1
# ╠═9af05d0f-5daf-48f4-9949-79db70ae4bbc
# ╟─92f98fe2-9154-4029-8282-45dee3f11d31
# ╠═0cfa5643-b65b-4374-a227-8f036ce32bbf
# ╠═93ea2a29-bac3-4629-bac5-5d03e8e45d80
