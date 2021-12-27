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

[YouTube Link](https://www.youtube.com/watch?v=OQRPeIQasdo)

As we will see later, we will use different classifiers and at the end of this notebook, we will compare them. We will define our accuracy function right now to get it out of the way. We will use a simple **accuracy** function that returns the ratio of the number of correctly classified observations (true positives) to the total number of predictions.
"""

# ╔═╡ 68c8634d-a052-4bd0-a101-5111dd9d4f68
findaccuracy(ŷ, y) = sum(ŷ .== y) / length(y)

# ╔═╡ 091524fe-ad86-45a3-ad25-370fe25a7a4e
md"""
## Load data

We will use Iris dataset from [RDatasets.jl](https://github.com/JuliaStats/RDatasets.jl) package. It is already stored as a `DataFrame`.
"""

# ╔═╡ 7a1a6481-eb71-4eb8-b47e-cf65c2c4cafa
iris = dataset("datasets", "iris")

# ╔═╡ 0bca5475-72f2-4189-b3d8-dd06cc951fc8
md"""
## Data wrangling

First, let's define our feature matrix `X`...
"""

# ╔═╡ e7f4e859-5e3c-439d-b069-46f17f2acf02
X = Matrix(iris[:, 1:4])

# ╔═╡ 0c1f6c24-ab74-494a-89d8-ba3fbda7fa24
md"""
And our label vector `labels`...
"""

# ╔═╡ b086ca45-6a60-4d4e-8387-de7e247812ed
labels = iris[:, 5]

# ╔═╡ c495a8f7-17d0-42f7-9adf-440d83d310c1
md"""
Since our dependent variable is categorical, let's encode it using the [MLBase.jl](https://github.com/JuliaStats/MLBase.jl) package...

**Note:** some algorithms require numeric labels rather than string labels.
"""

# ╔═╡ 35fad54f-e861-46e7-ab43-04a205c1991f
# mapping labels
irismap = labelmap(labels)

# ╔═╡ df121f29-a69d-42c9-b6a7-f129c4e94506
# enconding labels vector
y = labelencode(irismap, labels)

# ╔═╡ 1f39a7ed-7320-4f72-b4a0-fac44a71bb75
md"""
## Train/test split

In classification, we often want to use some of the data to fit a model (i.e. _training data_), and the rest of the data to validate (i.e. _testing data_). We will get this data ready now, so that we can easily use it in the rest of this notebook.

We will create the `train_test_split` function to perform train/test split. It will return the `y_train` indexes...

**Note:** the following `train_test_split` function performs a stratified random sampling by class.
"""

# ╔═╡ ecb46a59-1acf-48b9-b3c7-4c319388bc33
function train_test_split(y::Array{Int64,1}, train_ratio::Float64)
	classes = unique(y)
	y_train_idx = []
	for class in classes
		idxs = findall(y .== class)
		selected_idxs = randsubseq(idxs, train_ratio)
		push!(y_train_idx, selected_idxs...)
	end

	return y_train_idx
end

# ╔═╡ 8ca957e5-1bec-4498-aaca-86dccca1a49b
md"""
**Note:** `randsubseq(A, p)` function returns a vector consisting of a random subsequence of the given array `A`, where each element of `A` is included (in order) with independent probability `p`. Technically, this process is known as _Bernoulli sampling_ of `A`.
"""

# ╔═╡ 8868df3e-5933-4670-bbaf-eb8be100382b
# y train indexes
train_idxs = train_test_split(y, 0.7)

# ╔═╡ 27f97d52-c5c2-41a3-9faf-a8271a404b23
# y test indexes
test_idxs = setdiff(1:length(y), train_idxs)

# ╔═╡ 95a6bc1e-7986-4d86-9843-ceac5b668e4c
md"""
We will need one more function, and that is the function that will assign classes based on the predicted values when the predicted values are continuous.

**Example:** suppose that we have $\hat{y}_i = 0.95$. This is a continuous prediction and we may convert it into $1$ (i.e. _setosa_).
"""

# ╔═╡ 137c8d77-d6fd-44eb-937e-5978d201670d
assign_class(ŷᵢ) = argmin(abs.(ŷᵢ .- [1,2,3]))

# ╔═╡ 6317039e-1926-4143-a4d6-61f61b5e83e0
md"""
## Data modeling

We are going to build 7 different classification models:
1. Lasso
2. Ridge
3. Elastic Net
4. Decision Trees
5. Random Forests
6. Nearest Neighbour
7. Support Vector Machine

Then, we will compare model accuracies to figure out which models have the best performance.
"""

# ╔═╡ 1bd96a25-9542-432b-8e54-d6896a9cdade
md"""
### Lasso

**Least Absolute Shrinkage and Selection Operator (Lasso)** is a regression analysis method that performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the resulting statistical model. It was originally introduced in Geophysics, and later by Robert Tibshirani who coined the term.

Lasso was originally formulated for linear regression models. This simple case reveals a substantial amount about the estimator. These include its relationship to Ridge and best subset selection and the connections between Lasso coefficient estimates and so-called _soft thresholding_. It also reveals that (like standard linear regression) the coefficient estimates do not need to be unique if covariates (independent variables) are collinear.

**Note:** check [here](https://en.wikipedia.org/wiki/Lasso_(statistics)) for more math details.

For Lasso, Ridge and Elastic Net methods, we are going to use the [GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl) package.

We use the `glmnet` function, we will get several different solutions and $\lambda$ values (weighting parameter for the norm factor). Thus, we may use the `glmnetcv` function to get the _best_ $\lambda$ value (i.e. the $\lambda$ value with the lowest mean loss).

**Note:** to perform Lasso, we must set `alpha=1`, which is the default `alpha` value for the `glmnet` function.
"""

# ╔═╡ dd6889f6-18da-4218-bc48-b6589699498a
# Lasso solutions
lasso_sol = glmnet(X[train_idxs, :], y[train_idxs])

# ╔═╡ d03a03d1-fc97-4cb3-a852-5a3bc359a141
# performing cross validation
lasso_cv = glmnetcv(X[train_idxs, :], y[train_idxs])

# ╔═╡ d6008ed9-2a46-446a-b684-1d961c3ae20a
# best λ for Lasso
lasso_λ = lasso_sol.lambda[argmin(lasso_cv.meanloss)]

# ╔═╡ 039b7860-02fb-4f97-a1c2-9d20af989aac
# Lasso final model
lasso_f̂ = glmnet(X[train_idxs, :], y[train_idxs],lambda=[lasso_λ])

# ╔═╡ d0192106-f984-41e8-aa50-897481729dca
md"""
Now, we can predict labels for testing data...
"""

# ╔═╡ c106c3da-6444-4e64-9530-b6102766cd22
lasso_pred = GLMNet.predict(lasso_f̂, X[test_idxs,:])

# ╔═╡ 0da3fe69-f22d-4a29-9c88-c6b8e2e13d08
md"""
Note that predictions above are continuous rather than discrete. Thus, we must use the `assign_class` function to convert continuous predictions into discrete predictions...
"""

# ╔═╡ 1da8374e-aba5-4da3-bfe4-8f450408a20b
lasso_ŷ = assign_class.(lasso_pred)

# ╔═╡ 26ff22e5-e54e-4b85-b6d2-0db838cc1aca
md"""
Finally, we can compute Lasso accuracy using the `findaccuracy` function...
"""

# ╔═╡ 5d31a589-459e-4c11-8ea8-9840eb2cf8c0
lasso_acc = findaccuracy(lasso_ŷ, y[test_idxs])

# ╔═╡ 0040a4e5-13d3-4277-a865-e3f1bd328388
md"""
### Ridge

**Ridge** is a method of estimating the coefficients of multiple-regression models in scenarios where independent variables are highly correlated.

Ridge regression was developed as a possible solution to the imprecision of least square estimators when linear regression models have some multicollinear (highly correlated) independent variables—by creating a ridge regression estimator (RR). This provides a more precise ridge parameters estimate, as its variance and mean square estimator are often smaller than the least square estimators previously derived.

The main difference between Lasso and Ridge is that the former optimizes for L1-norm and Ridge optimizes for L2-norm.

**Note:** check [here](https://en.wikipedia.org/wiki/Ridge_regression) for more math details.

To perform Ridge, we are going to use the same functions. However, we must set `alpha = 0`.
"""

# ╔═╡ face4238-2511-4de5-8df3-2d25a668386e
# Ridge solutions
ridge_sol = glmnet(X[train_idxs, :], y[train_idxs], alpha=0)

# ╔═╡ faf82a4f-5acb-4d45-bf33-ded1dd82733c
# performing cross validation
ridge_cv = glmnetcv(X[train_idxs, :], y[train_idxs], alpha=0)

# ╔═╡ 103b74be-e907-41c5-ba08-3bd3ee4242e1
# best λ for Ridge
ridge_λ = ridge_sol.lambda[argmin(ridge_cv.meanloss)]

# ╔═╡ b16bfae9-5020-48fd-a796-f77d67a53f2a
# Ridge final model
ridge_f̂ = glmnet(X[train_idxs, :], y[train_idxs],lambda=[ridge_λ], alpha=0)

# ╔═╡ 4d175b94-43d1-497c-8022-42f544789e6d
md"""
Now, we can predict labels for testing data...
"""

# ╔═╡ e5e6130f-62de-4966-b164-9323b0b3bfb5
ridge_pred = GLMNet.predict(ridge_f̂, X[test_idxs,:])

# ╔═╡ a5a16631-0cd9-44cf-ae12-3f66b822f74b
md"""
Note that predictions above are also continuous rather than discrete. Thus, we must use the `assign_class` function to convert continuous predictions into discrete predictions...
"""

# ╔═╡ 887520af-9c11-4185-9f97-7da958332f76
ridge_ŷ = assign_class.(ridge_pred)

# ╔═╡ 6e37ae05-527f-4e5d-9207-383c0290f8d9
md"""
Finally, we can compute Ridge accuracy using the `findaccuracy` function...
"""

# ╔═╡ cf702060-5b49-451a-9075-3fb6f544e666
ridge_acc = findaccuracy(ridge_ŷ, y[test_idxs])

# ╔═╡ 94547be0-0467-4323-b5cd-f1efa9b98ac8
md"""
### Elastic Net

In the fitting of linear or logistic regression models, the **Elastic Net** is a regularized regression method that linearly combines the L1 and L2 penalties of the lasso and ridge methods.

**Note:** check [here](https://en.wikipedia.org/wiki/Elastic_net_regularization) for more math details.

To perform Elastic Net, we are going to use the same functions. However, we must set `alpha = 0.5`.
"""

# ╔═╡ ca590a9a-5909-4f1e-b02e-8af5da63439f
# Elastic Net solutions
enet_sol = glmnet(X[train_idxs, :], y[train_idxs], alpha=0.5)

# ╔═╡ d2e707e7-81ef-4b3d-8b8b-5ef65c8fc754
# performing cross validation
enet_cv = glmnetcv(X[train_idxs, :], y[train_idxs], alpha=0.5)

# ╔═╡ 9da0251a-ec4e-4b15-9527-9212af497e61
# best λ for Elastic Net
enet_λ = enet_sol.lambda[argmin(enet_cv.meanloss)]

# ╔═╡ eb72d5c2-15c4-41a9-81a2-c7c9f5f047ec
# Elastic Net final model
enet_f̂ = glmnet(X[train_idxs, :], y[train_idxs], alpha=0.5, lambda=[enet_λ])

# ╔═╡ 7a73633e-87d2-449b-8115-db6bf7e00266
md"""
Now, we can predict labels for testing data...
"""

# ╔═╡ 87906726-319e-454e-9299-77f6647081b8
enet_pred = GLMNet.predict(enet_f̂, X[test_idxs, :])

# ╔═╡ 5cdc7e8c-42e4-4f75-ab10-0a2b65d55809
md"""
Note that predictions above are also continuous rather than discrete. Thus, we must use the `assign_class` function to convert continuous predictions into discrete predictions...
"""

# ╔═╡ 14e764bb-94d9-4b1a-b2ce-1a39b930334a
enet_ŷ = assign_class.(enet_pred)

# ╔═╡ 46e9152e-7ee0-4420-8de6-331cef06b6dd
md"""
Finally, we can compute Elastic Net accuracy using the `findaccuracy` function...
"""

# ╔═╡ de00cd9a-009b-4991-aace-17ea06e41a1a
enet_acc = findaccuracy(enet_ŷ, y[test_idxs])

# ╔═╡ d999d681-d250-401e-87fe-963e625bfa56
md"""
### Decision Trees

**Decision Trees Learning** uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). Tree models where the target variable can take a discrete set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels.

**Note:** check [here](https://en.wikipedia.org/wiki/Decision_tree) for more math details.

We are going to use the [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) package.
"""

# ╔═╡ 7936c964-03a4-4498-83ad-25a6cc7825a5
# defining Decision Tree model
dt_f̂ = DecisionTreeClassifier(max_depth=2)

# ╔═╡ 00258a2e-3c99-4810-879c-9b15489e52ae
# training algorithm
DecisionTree.fit!(dt_f̂, X[train_idxs,:], y[train_idxs])

# ╔═╡ 4c0e88ae-a7ca-45ca-9a23-fbd4cd752cbe
md"""
Now, we can predict labels for testing data...
"""

# ╔═╡ 02ef2267-6694-4006-b9c2-21241e7ff13a
dt_ŷ = DecisionTree.predict(dt_f̂, X[test_idxs, :])

# ╔═╡ da6959b5-29a0-42c6-8a55-3711b5f41074
md"""
Finally, we can compute Decision Tree accuracy using the `findaccuracy` function...
"""

# ╔═╡ bc55879d-d6b2-4562-a710-988968aa3ae4
dt_acc = findaccuracy(dt_ŷ, y[test_idxs])

# ╔═╡ 50a8aa34-fa23-4dbc-b69f-b897a54ad157
md"""
### Random Forests

**Random Forests** are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. Random decision forests correct for decision trees habit of overfitting to their training set.

**Note:** check [here](https://en.wikipedia.org/wiki/Random_forest) for more math details.

The `RandomForestClassifier` function is available through the [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) package as well.
"""

# ╔═╡ 71672bfa-558e-4ceb-88fb-aa61f615603e
# defining Random Forests model
rf_f̂ = RandomForestClassifier(n_trees=20)

# ╔═╡ 7d3e38ab-009a-445b-9e63-acc8e3ac4d8c
# training algorithm
DecisionTree.fit!(rf_f̂, X[train_idxs,:], y[train_idxs])

# ╔═╡ 18564590-adaa-4f6d-8180-0c4ece6a4398
md"""
Now, we can predict labels for testing data...
"""

# ╔═╡ b19ddb74-a299-48f7-9bec-20d078edafb4
rf_ŷ = DecisionTree.predict(rf_f̂, X[test_idxs,:])

# ╔═╡ a36aad87-8772-4ff6-9e63-34313346f37e
md"""
Finally, we can compute Decision Tree accuracy using the `findaccuracy` function...
"""

# ╔═╡ 04d476ca-07cb-4138-bd1b-679c223d7faf
rf_acc = findaccuracy(rf_ŷ, y[test_idxs])

# ╔═╡ 626d7979-0362-4a54-8f7d-1df787a68195
md"""
### Nearest Neighbour

The **Nearest Neighbour Search** algorithm aims to find the point in the tree that is nearest to a given input point. This search can be done efficiently by using the tree properties to quickly eliminate large portions of the search space.

**Note:** check [here](https://en.wikipedia.org/wiki/K-d_tree) for more math details.

We are going to use the [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl) package. This is not straight forward, since we need to do some preprocessing first. Thus, let's build the KD tree. 

**Note:** we need to transpose `X` matrix in order to use NearestNeighbors.jl package.
"""

# ╔═╡ f81dfbd1-b018-4098-a484-2619215abace
# building KD tree
kdtree = KDTree(X[train_idxs,:]')

# ╔═╡ 14c7f4ba-1e06-4fe5-9fb5-b0e9bbf4793d
md"""
Now, we will use the `knn` function, which performs a lookup of the `k` nearest neigbours to the `points` from the data in the `tree`. If `sortres = true`, the result is sorted such that the results are in the order of increasing distance to the point. It returns both `indices` and `distances` of the training data.
"""

# ╔═╡ 0dd179a1-d48d-42d2-85c6-62ba48af0ccb
# getting indices and distances
knn_idxs, knn_dists = knn(kdtree, X[test_idxs,:]', 5, true);

# ╔═╡ 588afb06-4f7b-458f-8037-6ef53b96b60b
# indices
knn_idxs

# ╔═╡ ff2fcea0-3588-4a79-9906-f3effa148e86
md"""
**Note:** this is a nested `Array`, where each element is an 1D `Array` with the **indexes** of the 5 nearest points. Each element represents a test observation.
"""

# ╔═╡ 75d5de77-08db-4bdd-b418-fe9fa9e6353d
# distances
knn_dists

# ╔═╡ 4644848e-c9ae-4742-9ce3-4747a85b1fd2
md"""
**Note:** this is also a nested `Array`, where each element is an 1D `Array` with the **distances** of the 5 nearest points. Each element represents a test observation. Each element is in ascending distance order.
"""

# ╔═╡ fa554c22-2156-43a6-8014-cbd98914f27d
# 5 nearest train observation labels to each test observation
nn_y_train = y[train_idxs][hcat(knn_idxs...)]

# ╔═╡ be1405be-750e-4a74-895a-ed56beb0cd36
# counting the number of labels per test observation
nn_possible_labels = map(i -> counter(nn_y_train[:, i]), 1:size(nn_y_train, 2))

# ╔═╡ 45625714-3ff0-4b46-8087-14ecbe672d08
md"""
Now, we can predict labels for testing data. Thus, we need to select the most frequent label per each test observation.
"""

# ╔═╡ ed93cb77-9eee-491a-835c-2952bd612946
# selecting the most frequent label per each test observation
nn_ŷ = map(i -> parse(Int,
					  string(string(argmax(nn_possible_labels[i])))),
		   1:size(nn_y_train, 2))

# ╔═╡ ec8854ea-542a-4e15-8533-ec6225cafa72
md"""
Finally, we can compute Nearest Neighbour accuracy using the `findaccuracy` function...
"""

# ╔═╡ 1ee8b7ec-3542-40ae-945c-3bf489266985
nn_acc = findaccuracy(nn_ŷ, y[test_idxs])

# ╔═╡ 112a337f-11ab-4074-88bd-17a887562d43
md"""
### Support Vector Machine

**Support Vector Machines (SVM)** are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis.

Given a set of training examples, each marked as belonging to one of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier. SVM maps training examples to points in space so as to maximise the width of the gap between the two categories. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.

**Note:** check [here](https://en.wikipedia.org/wiki/Support-vector_machine) for more math details.

Here, we will use the [LIBSVM.jl](https://github.com/JuliaML/LIBSVM.jl) package. We need to transpose `X` matrix in order to use this package.
"""

# ╔═╡ 428463aa-0cc7-4008-95c7-8d1b5a28f2de
# training SVM model
svm_f̂ = svmtrain(X[train_idxs,:]', y[train_idxs])

# ╔═╡ fb93f2ce-36d2-48f0-bafd-5ca18afadfd7
md"""
Now, we can predict labels for testing data. At the same time, we will also get an `Array` with decision values...
"""

# ╔═╡ f5bb8271-4239-4c4e-956d-9714a38d2c25
svm_ŷ, decision_values = svmpredict(svm_f̂, X[test_idxs,:]');

# ╔═╡ 4a3d2073-f465-4fe2-9c4d-4ceb810a2285
# SVM predictions
svm_ŷ

# ╔═╡ ead6066e-6257-4659-84a6-98ea79993ea7
md"""
Finally, we can compute Support Vector Machine accuracy using the `findaccuracy` function...
"""

# ╔═╡ 92c46bcc-b3fa-4704-85cf-74782b9e6bef
svm_acc = findaccuracy(svm_ŷ, y[test_idxs])

# ╔═╡ 0f44eabe-c806-4154-b54e-65f8c678ef47
md"""
## Comparing models

Now, we can compare the accuracies of each model...
"""

# ╔═╡ e1fdde22-08ea-4ae4-82a6-b7adbb6e250a
begin
	models = ["Lasso", "Ridge", "Elastic Net", "DT", "RF", "NN", "SVM"]

	accs = [lasso_acc, ridge_acc, enet_acc, dt_acc, rf_acc, nn_acc, svm_acc]

	hcat(models, accs)
end

# ╔═╡ Cell order:
# ╟─3247c742-65df-11ec-0eed-7d76c7786941
# ╟─ff42960a-f27f-41b8-8eee-14fb5b0454ba
# ╟─b4c7b20b-ef19-4684-933b-7a00bfb69dce
# ╟─73a1eed0-af5b-42b3-a4b8-a6f191f49daa
# ╠═68c8634d-a052-4bd0-a101-5111dd9d4f68
# ╟─091524fe-ad86-45a3-ad25-370fe25a7a4e
# ╠═7a1a6481-eb71-4eb8-b47e-cf65c2c4cafa
# ╟─0bca5475-72f2-4189-b3d8-dd06cc951fc8
# ╠═e7f4e859-5e3c-439d-b069-46f17f2acf02
# ╟─0c1f6c24-ab74-494a-89d8-ba3fbda7fa24
# ╠═b086ca45-6a60-4d4e-8387-de7e247812ed
# ╟─c495a8f7-17d0-42f7-9adf-440d83d310c1
# ╠═35fad54f-e861-46e7-ab43-04a205c1991f
# ╠═df121f29-a69d-42c9-b6a7-f129c4e94506
# ╟─1f39a7ed-7320-4f72-b4a0-fac44a71bb75
# ╠═ecb46a59-1acf-48b9-b3c7-4c319388bc33
# ╟─8ca957e5-1bec-4498-aaca-86dccca1a49b
# ╠═8868df3e-5933-4670-bbaf-eb8be100382b
# ╠═27f97d52-c5c2-41a3-9faf-a8271a404b23
# ╟─95a6bc1e-7986-4d86-9843-ceac5b668e4c
# ╠═137c8d77-d6fd-44eb-937e-5978d201670d
# ╟─6317039e-1926-4143-a4d6-61f61b5e83e0
# ╟─1bd96a25-9542-432b-8e54-d6896a9cdade
# ╠═dd6889f6-18da-4218-bc48-b6589699498a
# ╠═d03a03d1-fc97-4cb3-a852-5a3bc359a141
# ╠═d6008ed9-2a46-446a-b684-1d961c3ae20a
# ╠═039b7860-02fb-4f97-a1c2-9d20af989aac
# ╟─d0192106-f984-41e8-aa50-897481729dca
# ╠═c106c3da-6444-4e64-9530-b6102766cd22
# ╟─0da3fe69-f22d-4a29-9c88-c6b8e2e13d08
# ╠═1da8374e-aba5-4da3-bfe4-8f450408a20b
# ╟─26ff22e5-e54e-4b85-b6d2-0db838cc1aca
# ╠═5d31a589-459e-4c11-8ea8-9840eb2cf8c0
# ╟─0040a4e5-13d3-4277-a865-e3f1bd328388
# ╠═face4238-2511-4de5-8df3-2d25a668386e
# ╠═faf82a4f-5acb-4d45-bf33-ded1dd82733c
# ╠═103b74be-e907-41c5-ba08-3bd3ee4242e1
# ╠═b16bfae9-5020-48fd-a796-f77d67a53f2a
# ╟─4d175b94-43d1-497c-8022-42f544789e6d
# ╠═e5e6130f-62de-4966-b164-9323b0b3bfb5
# ╟─a5a16631-0cd9-44cf-ae12-3f66b822f74b
# ╠═887520af-9c11-4185-9f97-7da958332f76
# ╟─6e37ae05-527f-4e5d-9207-383c0290f8d9
# ╠═cf702060-5b49-451a-9075-3fb6f544e666
# ╟─94547be0-0467-4323-b5cd-f1efa9b98ac8
# ╠═ca590a9a-5909-4f1e-b02e-8af5da63439f
# ╠═d2e707e7-81ef-4b3d-8b8b-5ef65c8fc754
# ╠═9da0251a-ec4e-4b15-9527-9212af497e61
# ╠═eb72d5c2-15c4-41a9-81a2-c7c9f5f047ec
# ╟─7a73633e-87d2-449b-8115-db6bf7e00266
# ╠═87906726-319e-454e-9299-77f6647081b8
# ╟─5cdc7e8c-42e4-4f75-ab10-0a2b65d55809
# ╠═14e764bb-94d9-4b1a-b2ce-1a39b930334a
# ╟─46e9152e-7ee0-4420-8de6-331cef06b6dd
# ╠═de00cd9a-009b-4991-aace-17ea06e41a1a
# ╟─d999d681-d250-401e-87fe-963e625bfa56
# ╠═7936c964-03a4-4498-83ad-25a6cc7825a5
# ╠═00258a2e-3c99-4810-879c-9b15489e52ae
# ╟─4c0e88ae-a7ca-45ca-9a23-fbd4cd752cbe
# ╠═02ef2267-6694-4006-b9c2-21241e7ff13a
# ╟─da6959b5-29a0-42c6-8a55-3711b5f41074
# ╠═bc55879d-d6b2-4562-a710-988968aa3ae4
# ╟─50a8aa34-fa23-4dbc-b69f-b897a54ad157
# ╠═71672bfa-558e-4ceb-88fb-aa61f615603e
# ╠═7d3e38ab-009a-445b-9e63-acc8e3ac4d8c
# ╟─18564590-adaa-4f6d-8180-0c4ece6a4398
# ╠═b19ddb74-a299-48f7-9bec-20d078edafb4
# ╟─a36aad87-8772-4ff6-9e63-34313346f37e
# ╠═04d476ca-07cb-4138-bd1b-679c223d7faf
# ╟─626d7979-0362-4a54-8f7d-1df787a68195
# ╠═f81dfbd1-b018-4098-a484-2619215abace
# ╟─14c7f4ba-1e06-4fe5-9fb5-b0e9bbf4793d
# ╠═0dd179a1-d48d-42d2-85c6-62ba48af0ccb
# ╠═588afb06-4f7b-458f-8037-6ef53b96b60b
# ╟─ff2fcea0-3588-4a79-9906-f3effa148e86
# ╠═75d5de77-08db-4bdd-b418-fe9fa9e6353d
# ╟─4644848e-c9ae-4742-9ce3-4747a85b1fd2
# ╠═fa554c22-2156-43a6-8014-cbd98914f27d
# ╠═be1405be-750e-4a74-895a-ed56beb0cd36
# ╟─45625714-3ff0-4b46-8087-14ecbe672d08
# ╠═ed93cb77-9eee-491a-835c-2952bd612946
# ╟─ec8854ea-542a-4e15-8533-ec6225cafa72
# ╠═1ee8b7ec-3542-40ae-945c-3bf489266985
# ╟─112a337f-11ab-4074-88bd-17a887562d43
# ╠═428463aa-0cc7-4008-95c7-8d1b5a28f2de
# ╟─fb93f2ce-36d2-48f0-bafd-5ca18afadfd7
# ╠═f5bb8271-4239-4c4e-956d-9714a38d2c25
# ╠═4a3d2073-f465-4fe2-9c4d-4ceb810a2285
# ╟─ead6066e-6257-4659-84a6-98ea79993ea7
# ╠═92c46bcc-b3fa-4704-85cf-74782b9e6bef
# ╟─0f44eabe-c806-4154-b54e-65f8c678ef47
# ╠═e1fdde22-08ea-4ae4-82a6-b7adbb6e250a
