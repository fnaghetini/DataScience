### A Pluto.jl notebook ###
# v0.17.4

using Markdown
using InteractiveUtils

# ╔═╡ d16725a0-6cd3-11ec-0998-fd4af26f10dd
begin
	using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
	Pkg.precompile()

	using PlutoUI
	using Plots
	using Statistics
	using StatsBase
	using PyCall
	using DataFrames
	using GLM
	using Tables
	using XLSX
	using DataStructures
	using MLBase
	using RDatasets
	using LsqFit
end;

# ╔═╡ 91471950-703f-4927-8bce-f4478f4a53ba
PlutoUI.TableOfContents(aside=true, indent=true, depth=3)

# ╔═╡ a53d44c5-75c4-4508-aa19-33e838c9e0bb
md"""
# 📈 Regression

Regression Analysis is a set of statistical processes for estimating the relationships between a _dependent variable_ (often called the _outcome_ or _response_ variable) and one or more _independent variables_ (often called _predictors_, _covariates_, _explanatory variables_ or _features_). The most common form of regression analysis is linear regression, in which one finds the line (or a more complex linear combination) that most closely fits the data according to a specific mathematical criterion.

[YouTube Link](https://www.youtube.com/watch?v=5TCbIK_cpZE)
"""

# ╔═╡ e9458f99-4ec0-4167-b771-3090dfec6320
md"""
## Simple linear regression
"""

# ╔═╡ 784c5b0e-59dc-4b2d-bf40-112261ac8e85
md"""
### Building from scratch

Firstly, let's generate some random data and then plot it...
"""

# ╔═╡ 65e22edc-ef5c-412f-82f4-1c84da209d6d
begin
	# x values
	xvals = repeat(1:0.5:10, inner=2)
	# y values (with some noise)
	yvals = 3 .+ xvals .+ 2 .* rand(length(xvals)) .-1

	# plotting
	scatter(xvals, yvals, color=:black, leg=false,
			xlabel="X values", ylabel="Y values")
end

# ╔═╡ db2c3459-2127-4b46-8e73-546850767dc0
md"""
Assuming an ordinary least squares regression, the slope `a` of the fitted line is equal to the correlation between `y` and `x` corrected by the ratio of standard deviations of these variables. The intercept `b` of the fitted line is such that the line passes through the center of mass (`x`, `y`) of the data points.

Thus, we can easily build the `find_best_fit` function, which returns `a` and `b` coefficients.
"""

# ╔═╡ 3bb8c382-f97d-48dc-8cbe-99bf77de0784
function find_best_fit(x, y)
	X̅, Y̅ = mean(x), mean(y)
	Sˣ, Sʸ = std(x), std(y)

	a = cor(x,y) * (Sˣ/Sʸ)
	b = Y̅ - a*X̅

	return a, b
end

# ╔═╡ 75059ff4-6b9d-4e01-9aed-33ea8b0e9a71
md"""
We can now use the function above to calculate `a` and `b` and then compute the estimated values `ŷvals`...
"""

# ╔═╡ e83331a1-8655-4de0-a36f-b74b5e43674b
begin
	a, b = find_best_fit(xvals, yvals)
	ŷvals₁ = a .* xvals .+ b
end

# ╔═╡ d81e448b-b4e4-4d6f-8ca7-42c41569977d
md"""
### numpy

We can also import the `numpy` python package to perform a simple linear regression. We are able to use any `numpy` function by adding the prefix `np`, just like in Python.
"""

# ╔═╡ a047abfb-178c-40ac-8f8c-c3a8a73c1849
# importing numpy
np = pyimport("numpy");

# ╔═╡ 208af138-5131-4034-885b-c41f6ccf36f5
# numpy model
np_model = np.polyfit(xvals, yvals, 1)

# ╔═╡ 39884fc4-c41c-46e5-a92a-44d766243715
# estimated ŷ values
ŷvals₂ = np_model[1] .* collect(xvals) .+ np_model[2]

# ╔═╡ 13719feb-60ae-433b-bf1e-df6ac045215d
md"""
### GLM

Finally, we can use the [GLM.jl](https://github.com/JuliaStats/GLM.jl) package to perform simple linear regression using the `lm` function.
"""

# ╔═╡ 10c072d4-b793-458c-b947-d5dc3aae5e25
# converting data into a DataFrame
data = DataFrame(X=xvals, Y=yvals);

# ╔═╡ 68a0574f-5b17-48ae-b8c4-a03c7e385561
# GLM model
glm_model = lm(@formula(Y ~ X), data)

# ╔═╡ ecb1cb29-2b57-4197-9327-e762fc876281
# estimated ŷ values
ŷvals₃ = predict(glm_model)

# ╔═╡ 056b28f3-8da4-4a94-a33c-72b30c8aa0aa
md"""
### Comparing methods

Now we can plot all the three models. We can see below that both `numpy` and `GLM` models are exactly the same. The model that we build from scratch is slightly different.
"""

# ╔═╡ 9a5dcad1-9365-423c-b27a-a6c206850848
begin
	# plotting data
	scatter(xvals, yvals, color=:black, label="Data",
			xlabel="X values", ylabel="Y values", legend=:topleft)

	# plotting "from scratch" fitted line
	plot!(xvals, ŷvals₁, color=:red, label="From Scratch")

	# plotting numpy model
	plot!(xvals, ŷvals₂, color=:green, label="Numpy")

	# plotting GLM model
	plot!(xvals, ŷvals₃, color=:blue, label="GLM")
end

# ╔═╡ a89ec73a-5e35-49ad-8829-dc2ecc7e8aac
md"""
## Load real data

Now let's get some real data. We will use housing information from zillow, check out the file `zillow_data_download_april2020.xlsx` for a quick look of what the data looks like. Our goal will be to build a linear regression model between the number of houses listed vs the number of houses sold in a few states. Fitting these models can serve as a key real estate indicator.
"""

# ╔═╡ aa78c706-8860-46f1-9982-ea10bdc4fccb
# loading data
S = XLSX.readxlsx("data/zillow_data_download_april2020.xlsx")

# ╔═╡ 618ab9ee-ee9c-4dbb-b48f-ebeeb15549c6
md"""
**Note:** there are three different sheets in our data.

## Data wrangling

Let's create a `DataFrame` from the `Sale_counts_city` sheet...
"""

# ╔═╡ b530e842-bfa4-4c41-9cc9-4aa629e12f99
# selecting sheet
sale_counts = S["Sale_counts_city"][:]

# ╔═╡ 041e87c7-deee-46c1-85be-ea47df16217c
# converting Matrix into DataFrame
df_sale_counts = DataFrame(sale_counts[2:end,:], Symbol.(sale_counts[1,:]))

# ╔═╡ 19a44596-2367-4964-b21e-7d84ba147915
md"""
Now, we can do the same with the `MonthlyListings_City` sheet...
"""

# ╔═╡ ab8cf8d2-9ac6-4361-b168-243f781e7dfa
begin
	# selecting sheet
	month_list = S["MonthlyListings_City"][:]
	# converting Matrix into DataFrame
	df_month_list = DataFrame(month_list[2:end,:], Symbol.(month_list[1,:]))
end

# ╔═╡ 215483a1-6059-437d-a4b6-ca3cd897f13b
md"""
We may filter just the data from February 2020...
"""

# ╔═╡ b075d090-91aa-4d1f-bb9f-a93dd65413e7
begin
	# monthly listing data (Feb 2020)
	monthlist_2020_2 = df_month_list[!,[1,2,3,4,5,end]]
	rename!(monthlist_2020_2, Symbol("2020-02") => :listing)
end

# ╔═╡ 1b8f7b0d-a843-4af7-90b7-4a64d74edab0
begin
	# sale counts data (Feb 2020)
	salecounts_2020_2 = df_sale_counts[!,[1,end]]
	rename!(salecounts_2020_2, Symbol("2020-02") => :sales)
end

# ╔═╡ 39456bea-24dc-482b-9d8b-33a9f2ed715a
md"""
Next, we can (outer) join our two DataFrames and, then, remove missing values...
"""

# ╔═╡ f195db71-f2a4-4f5c-bfa4-2d9ec1f23a32
begin
	feb2020data = outerjoin(monthlist_2020_2, salecounts_2020_2, on=:RegionID)
	dropmissing!(feb2020data)
end

# ╔═╡ 2da58c4e-8db4-4d80-aa29-f3ec10952f73
md"""
Let's define variables for all the three columns we are interested in...
"""

# ╔═╡ 93179d9a-5bf4-488b-b710-28a5f56b2e26
begin
	sales = feb2020data[!,:sales]
	counts = feb2020data[!,:listing]
	states = feb2020data[!,:StateName]
end;

# ╔═╡ 560a134f-1175-4d56-83af-1ad6bbe1a99f
md"""
Now, we can select just the top 10 states (most cited states). We will use the `counter` function from the [DataStructures.jl](https://github.com/JuliaCollections/DataStructures.jl) package.
"""

# ╔═╡ 0c4a00f6-69f2-4fc3-a7f8-ca6e38404ac9
# counter object (Accumulator)
C = counter(states)

# ╔═╡ a02a9fdf-46bc-4ea1-bb29-b1731a66c057
# converting into a Dictionary
Cdict = C.map

# ╔═╡ 63b596be-9392-4efb-8d31-7f1d6e9c7254
begin
	# getting counts as an Array
	countvals = values(Cdict) |> collect
	
	# getting state names as an Array
	state_labels = keys(Cdict) |> collect
end;

# ╔═╡ feced955-2e69-44ae-bc2a-82d4f2136b38
# getting indexes from 10 top states
topstateindexes = sortperm(countvals, rev=true)[1:10]

# ╔═╡ 831484f5-6cdc-4641-8adf-117a33c518c2
md"""
**Note:** the built-in `sortperm` function returns a permutation vector `I` that puts `v[I]` in sorted order. The order is specified using the same keywords as `sort!`.
"""

# ╔═╡ b94019d8-a7d8-4781-819b-28eeac823812
# getting top 10 states names
topstates = state_labels[topstateindexes]

# ╔═╡ 4735fc87-68cb-4617-aa7c-6139ba7b7c0a
md"""
## Linear regression models

Firstly, we will plot the model for each state using the formula `Y ~ X`.
"""

# ╔═╡ ec7ae9e6-b191-4578-8a47-9d2245280316
begin
	# creating a blank Array of Plot
	all_plots₁ = Array{Plots.Plot,1}(undef,10)

	for (i,s) in enumerate(topstates)
		# finding indexes of all occurences of a given state
		ids = findall(feb2020data[!,:StateName] .== s)
		# getting data for a given state
		data = DataFrame(X=Float64.(counts[ids]), Y=Float64.(sales[ids]))
		# training the model
		ols = GLM.lm(@formula(Y ~ X), data)

		# plotting data by each state
		all_plots₁[i] = scatter(data.X, data.Y, markersize=2,
								xlims=(0,500), ylims=(0,500),
								color=i, aspect_ratio=:equal,
								legend=false, title=s)

		# plotting model by each state
		plot!(data.X, predict(ols), color=:black)
	end

	# display plots
	plot(all_plots₁..., layout=(2,5), size=(850,300))
end

# ╔═╡ 216dc17e-54c1-4520-890d-8663165b10dd
md"""
We can do the same thing, but now we will plot all the elements within the same figure.

**Note:** when we use the `Y ~ X + 0` formula, it forces the intercept to be zero.
"""

# ╔═╡ dddfb8fb-7047-4417-8dd2-569a93277f0c
begin
	plot()
	
	for (i,s) in enumerate(topstates)
	    ids = findall(feb2020data[!,:StateName] .== s)
	    data = DataFrame(X=Float64.(counts[ids]), Y=Float64.(sales[ids]))
	    ols = GLM.lm(@formula(Y ~ 0 + X), data)
		
	    scatter!(counts[ids], sales[ids], markersize=2, xlim=(0,500),
				 ylim=(0,500), color=i, aspect_ratio=:equal,
	        	 legend=false, marker=(3,3,stroke(0)), alpha=0.2)
		
	    if s ∈ ["NC","CA","FL"]
	        annotate!([(500-20,10+coef(ols)[1]*500, text(s,10))])
	    end

	    plot!(counts[ids], predict(ols), color=i, linewidth=2)
	end

	xlabel!("Listings")
	ylabel!("Sales")
end

# ╔═╡ 5116c117-75e4-4989-b06f-2883bba48330
md"""
## Logistic regression

So far, we have shown several ways to solve the linear regression problem in Julia. Here, we will first start with a motivating example of when you would want to use logistic regression. Let's assume that our predictor vector is binary (`0` or `1`), let's fit a linear regression model.
"""

# ╔═╡ 3a86069b-3f24-4eec-a7ae-f77189a4fd77
# generating random data
𝒮 = DataFrame(X=collect(1:7), Y=[1,0,1,1,1,1,1])

# ╔═╡ 453385dd-8bda-42b8-9b4a-6a4963b21346
# linear regression model
lin_reg = lm(@formula(Y ~ X), 𝒮)

# ╔═╡ bd31693d-dee9-4534-b4d4-3904997e8020
begin
	# plotting data
	scatter(𝒮.X, 𝒮.Y, legend=false, xlabel="X", ylabel="Y")
	#plotting model
	plot!(1:7, predict(lin_reg))
end

# ╔═╡ 0fb142cd-ac45-42e8-b5d1-24c6142392bb
md"""
What this plot quickly shows is that linear regression may end up predicting values outside the `[0,1]` interval. For an example like this, we will use logistic regression. Interestingly, a generalized linear model (https://en.wikipedia.org/wiki/Generalized_linear_model) unifies concepts like linear regression and logistic regression, and the [GLM.jl](https://github.com/JuliaStats/GLM.jl) package allows you to apply either of these regressions easily by specifying the `distribution family` and the `link` function.

To apply logistic regression via the `GLM.jl` package, you can readily use the `Binomial` family and the `LogitLink` link function.

### Load data

Firstly, let's load some data from the [RDatasets.jl](https://github.com/JuliaStats/RDatasets.jl) package...
"""

# ╔═╡ 3caa5331-2507-4299-b2e4-50e8a5cbbe46
cats = dataset("MASS", "cats")

# ╔═╡ 22c201af-9437-4dde-8653-508ebf514008
md"""
### Data wrangling

We will map the sex of each cat to a binary `0/1` value.
"""

# ╔═╡ 16367b7a-65ea-4fa0-abad-a1cc25c60ee6
# getting the map
sex_map = labelmap(cats.Sex)

# ╔═╡ 1d7ef9db-dd15-4ed4-841a-b4baf02302a1
# encoding Sex column
sex_enconde = labelencode(sex_map, cats.Sex)

# ╔═╡ f42bdc3a-943b-4157-853d-860af52e059c
# plotting data by Sex
scatter(cats.BWt, cats.HWt, color=sex_enconde, legend=false)

# ╔═╡ 310f2282-d6dd-405d-8399-e75fdc75c27c
md"""
Females (blue) seem to be more present in the lower left corner and Males (orange) seem to be present in the top right corner.

We will use the `HWt` variable as the only feature and the `encoded_sex` as the target. Let's convert labels from `1/2` into `0/1`...
"""

# ╔═╡ cc911a53-9139-4c90-b02b-15f30e6772f0
cats_df = DataFrame(X=cats[!,:HWt], Y=sex_enconde .- 1)

# ╔═╡ 79bbe213-c15a-4ecb-aef7-107846234f1d
md"""
### Modeling data

Let's run a logistic regression model on this data.
"""

# ╔═╡ 11b1f5d1-7ae2-43dd-bd35-5b0ba6fde6f4
# logistic model
probit = glm(@formula(Y ~ X), cats_df, Binomial(), LogitLink())

# ╔═╡ e3e9d570-af57-4065-84ec-b61400fa449d
begin
	# plotting ground truth data
	scatter(cats_df.X, cats_df.Y, label="Ground Truth", color=6)
	
	# plotting predictions
	scatter!(cats_df.X, predict(probit), label="Predictions", color=7,
			 xlabel="Heart Rate", ylabel="Pr(Sex=M)")
end

# ╔═╡ dfc76a4c-4e2b-4588-92e1-2209507fd670
md"""
**Note:** as you can see, contrary to the linear regression case, the predicted values do not go beyond `1`.
"""

# ╔═╡ 31308555-dfc9-4e28-9108-8a83d65b7770
md"""
## Non-linear regression

Finally, sometimes you may have a set of points and the goal is to fit a non-linear function (maybe a quadratic function, a cubic function, an exponential function...). The way we would solve such a problem is by minimizing the least square error between the fitted function and the observations we have. We will use the package [LsqFit.jl](https://github.com/JuliaNLSolvers/LsqFit.jl) for this task.

**Note:** this problem is usually modeled as a numerical optimizaiton problem.
"""

# ╔═╡ 4a7d2c25-04b1-4ccb-85cd-7276902a9679
md"""
### Generate data

We will first set up our data and plot it...
"""

# ╔═╡ 7203ba1b-ce21-4c53-9dce-358df13df21e
begin
	x = 0:0.05:10
	y = 1 * exp.(-x * 2) + 2 * sin.(0.8 * π * x) + 0.15 * randn(length(x));
	scatter(x, y, legend=false)
end

# ╔═╡ 775e6fbe-8c10-4f94-b53e-618e4dd551fc
md"""
### Modeling data

Then, we set up the model with `model(x,p)`. The vector `p` is what to be estimated given a set of values `x`.
"""

# ╔═╡ 5d75ebce-1181-4490-ae10-8bcc08456fdf
begin
	@. model(v, p) = p[1] * exp(-v * p[2]) + p[3] * sin(0.8 * π *v)
	p0 = [0.5, 0.5, 0.5]
	myfit = curve_fit(model, x, y, p0)
end

# ╔═╡ a94163fe-e1ac-483a-bc2d-5d7795cfe188
md"""
⚠️ A note about `curve_fit`: this function can take multiple other inputs, for instance the Jacobian of what you are trying to fit. We don't dive into these details here, but be sure to check out the LsqFit.jl package to see what other things you can can pass to create a better fit.

Also note that julia has multiple packages that allow you to create Jacobians so you don't have to write them yourself. Two such packages are [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl) or [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl).

↪️ Back to our example. Let's define the paramenter variable and then calculate the predictions `ŷ`...
"""

# ╔═╡ 6a965e6d-4724-47c8-a3bb-bfec7d6706c9
begin
	p = myfit.param
	ŷ = p[1] * exp.(-x * p[2]) + p[3] * sin.(0.8 * π * x)
end

# ╔═╡ c7d07a47-3f3f-482e-9c06-78877b91e380
md"""
We are ready now to plot the curve we have generated...
"""

# ╔═╡ 6ca3290d-5980-4749-b309-fd84d73b7bb5
begin
	scatter(x, y, legend=false, xlabel="X", ylabel="Y")
	plot!(x, ŷ)
end

# ╔═╡ Cell order:
# ╟─d16725a0-6cd3-11ec-0998-fd4af26f10dd
# ╟─91471950-703f-4927-8bce-f4478f4a53ba
# ╟─a53d44c5-75c4-4508-aa19-33e838c9e0bb
# ╟─e9458f99-4ec0-4167-b771-3090dfec6320
# ╟─784c5b0e-59dc-4b2d-bf40-112261ac8e85
# ╠═65e22edc-ef5c-412f-82f4-1c84da209d6d
# ╟─db2c3459-2127-4b46-8e73-546850767dc0
# ╠═3bb8c382-f97d-48dc-8cbe-99bf77de0784
# ╟─75059ff4-6b9d-4e01-9aed-33ea8b0e9a71
# ╠═e83331a1-8655-4de0-a36f-b74b5e43674b
# ╟─d81e448b-b4e4-4d6f-8ca7-42c41569977d
# ╠═a047abfb-178c-40ac-8f8c-c3a8a73c1849
# ╠═208af138-5131-4034-885b-c41f6ccf36f5
# ╠═39884fc4-c41c-46e5-a92a-44d766243715
# ╟─13719feb-60ae-433b-bf1e-df6ac045215d
# ╠═10c072d4-b793-458c-b947-d5dc3aae5e25
# ╠═68a0574f-5b17-48ae-b8c4-a03c7e385561
# ╠═ecb1cb29-2b57-4197-9327-e762fc876281
# ╟─056b28f3-8da4-4a94-a33c-72b30c8aa0aa
# ╟─9a5dcad1-9365-423c-b27a-a6c206850848
# ╟─a89ec73a-5e35-49ad-8829-dc2ecc7e8aac
# ╠═aa78c706-8860-46f1-9982-ea10bdc4fccb
# ╟─618ab9ee-ee9c-4dbb-b48f-ebeeb15549c6
# ╠═b530e842-bfa4-4c41-9cc9-4aa629e12f99
# ╠═041e87c7-deee-46c1-85be-ea47df16217c
# ╟─19a44596-2367-4964-b21e-7d84ba147915
# ╠═ab8cf8d2-9ac6-4361-b168-243f781e7dfa
# ╟─215483a1-6059-437d-a4b6-ca3cd897f13b
# ╠═b075d090-91aa-4d1f-bb9f-a93dd65413e7
# ╠═1b8f7b0d-a843-4af7-90b7-4a64d74edab0
# ╟─39456bea-24dc-482b-9d8b-33a9f2ed715a
# ╠═f195db71-f2a4-4f5c-bfa4-2d9ec1f23a32
# ╟─2da58c4e-8db4-4d80-aa29-f3ec10952f73
# ╠═93179d9a-5bf4-488b-b710-28a5f56b2e26
# ╟─560a134f-1175-4d56-83af-1ad6bbe1a99f
# ╠═0c4a00f6-69f2-4fc3-a7f8-ca6e38404ac9
# ╠═a02a9fdf-46bc-4ea1-bb29-b1731a66c057
# ╠═63b596be-9392-4efb-8d31-7f1d6e9c7254
# ╠═feced955-2e69-44ae-bc2a-82d4f2136b38
# ╟─831484f5-6cdc-4641-8adf-117a33c518c2
# ╠═b94019d8-a7d8-4781-819b-28eeac823812
# ╟─4735fc87-68cb-4617-aa7c-6139ba7b7c0a
# ╠═ec7ae9e6-b191-4578-8a47-9d2245280316
# ╟─216dc17e-54c1-4520-890d-8663165b10dd
# ╠═dddfb8fb-7047-4417-8dd2-569a93277f0c
# ╟─5116c117-75e4-4989-b06f-2883bba48330
# ╠═3a86069b-3f24-4eec-a7ae-f77189a4fd77
# ╠═453385dd-8bda-42b8-9b4a-6a4963b21346
# ╠═bd31693d-dee9-4534-b4d4-3904997e8020
# ╟─0fb142cd-ac45-42e8-b5d1-24c6142392bb
# ╠═3caa5331-2507-4299-b2e4-50e8a5cbbe46
# ╟─22c201af-9437-4dde-8653-508ebf514008
# ╠═16367b7a-65ea-4fa0-abad-a1cc25c60ee6
# ╠═1d7ef9db-dd15-4ed4-841a-b4baf02302a1
# ╠═f42bdc3a-943b-4157-853d-860af52e059c
# ╟─310f2282-d6dd-405d-8399-e75fdc75c27c
# ╠═cc911a53-9139-4c90-b02b-15f30e6772f0
# ╟─79bbe213-c15a-4ecb-aef7-107846234f1d
# ╠═11b1f5d1-7ae2-43dd-bd35-5b0ba6fde6f4
# ╠═e3e9d570-af57-4065-84ec-b61400fa449d
# ╟─dfc76a4c-4e2b-4588-92e1-2209507fd670
# ╟─31308555-dfc9-4e28-9108-8a83d65b7770
# ╟─4a7d2c25-04b1-4ccb-85cd-7276902a9679
# ╠═7203ba1b-ce21-4c53-9dce-358df13df21e
# ╟─775e6fbe-8c10-4f94-b53e-618e4dd551fc
# ╠═5d75ebce-1181-4490-ae10-8bcc08456fdf
# ╟─a94163fe-e1ac-483a-bc2d-5d7795cfe188
# ╠═6a965e6d-4724-47c8-a3bb-bfec7d6706c9
# ╟─c7d07a47-3f3f-482e-9c06-78877b91e380
# ╠═6ca3290d-5980-4749-b309-fd84d73b7bb5
