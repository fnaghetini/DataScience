### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 6c3082f0-6196-11ec-1d58-03f55d178603
begin
	using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
	Pkg.precompile()

	using PlutoUI
	using LinearAlgebra
	using SparseArrays
	using Images
	using MAT
end;

# ╔═╡ 6ff4cfec-ba3e-4937-bc3a-bf2d22f1a8f6
PlutoUI.TableOfContents(aside=true, indent=true, depth=3)

# ╔═╡ 63f2a85b-a426-4c37-a1a0-049d00cfed23
html"""
<p align="center">
	<img src="https://github.com/JuliaAcademy/DataScience/blob/main/datascience.png?raw=true" alt="Course-Logo" width="450px">
</p>
"""

# ╔═╡ a3ead5b3-0251-4054-b6f7-63bf51b5ff71
md"""
# 📐 Linear Algebra

A lot of the Data Science methods we will see in this tutorial require some understanding of Linear Algebra, and, in this notebook, we will focus on how Julia handles matrices, the types that exist, and how to call basic Linear Algebra tasks.

[YouTube link](https://www.youtube.com/watch?v=bndXPsRHPg0)

**Note:** matrices in Julia are stored column-wise. It's not actually a matrix (2D structure). In fact, it's an 1D structure.

```math
\begin{bmatrix}
	1   & 2   & 3   \\
	11  & 12  & 13  \\
	110 & 120 & 130 \\
\end{bmatrix}
```

is stored in memory as:

```math
\begin{bmatrix}
	1   \\
	11  \\
    110 \\
    2   \\
	12  \\
    120 \\
    3   \\
	13  \\
    130 \\
\end{bmatrix}
```

"""

# ╔═╡ 1fe44825-0933-4fb8-9b36-511f0e7ffeea
md"""
## Getting started

We will get started with creating a random (2D) matrix...
"""

# ╔═╡ cd961076-6baf-4626-9089-d33b9dbc94ce
A = rand(10, 10)

# ╔═╡ eaaffef2-903c-49cf-80b4-bb5adf7d8e92
md"""
We can transpose the matrix with the `'` operator...
"""

# ╔═╡ 92c1a395-bd96-477a-95dc-294a2ac99784
Aᵗ = A'

# ╔═╡ 5c852935-3055-4178-8ac3-3a6f1d6e8de9
md"""
The `transpose` function creates a matrix of type `Adjoint`. In Julia, it is a lazy adjoint. Often, we can easily perform Linear Algebra operations such as `A * A'` without actually transposing the matrix. This type is intended for Linear Algebra usage.
"""

# ╔═╡ de94c2dc-b4b0-42a2-aff0-e54543df82e1
typeof(Aᵗ)

# ╔═╡ 4770ca47-71ea-4fe1-a5ad-52c4e31f3bdd
"Adjoint = $(sizeof(Aᵗ)) bytes"

# ╔═╡ eccced3e-5898-4d93-8950-58aa77fc4c0b
"Matrix = $(sizeof(copy(Aᵗ))) bytes"

# ╔═╡ 02cbd31b-bd62-48f4-be5e-242279e215d9
md"""
**Note:** we can convert an `Adjoint` object into a `Matrix` by copying it with the `copy` function.

We can check if $A$ is positive definite with the `isposdef` function.
"""

# ╔═╡ 89fb6b4c-911b-4a39-bdef-b70dbe17f14f
isposdef(A)

# ╔═╡ 2082cb17-c373-459d-abc1-abe41cfa023a
md"""
We can do matrix multiplication with the `*` operator.
"""

# ╔═╡ efc88903-189b-48f4-8cde-17bb2c407010
A * Aᵗ

# ╔═╡ e6c8e66b-f9a3-4b93-b5a4-97c030d381d4
md"""
We basically can access matrix elements in two different ways. We can think about a matrix as an 1D column-wise structure and use just one index...
"""

# ╔═╡ bb678a13-b45d-410b-b8d5-6346eb839c56
A[11]

# ╔═╡ 41e22cf0-2448-4643-89ea-7315fb8b0b32
md"""
Or we can intuitively access matrix elements with a 2D indexing syntax...
"""

# ╔═╡ e759d9a1-29cb-4e15-b629-a913ce721b6f
A[1,2]

# ╔═╡ f662e8dd-d998-4e8c-8924-aee9fe633129
md"""
Both methods are equivalent, but the first one is faster...
"""

# ╔═╡ 39797209-2206-495f-b517-2ed219ff47ca
A[13] == A[3,2]

# ╔═╡ f9e97b10-4908-4811-a255-c76c0eaed01c
md"""
We can use the sliders below to access matrix `A` elements using the 2nd method...

Row $i$ $(@bind i Slider(1:10, show_value=true))

Column $j$ $(@bind j Slider(1:10, show_value=true))
"""

# ╔═╡ 157971c4-04d9-41de-b378-4a9e998725e9
A[i,j]

# ╔═╡ d82db838-5ba6-456d-bdeb-cb75a20e081c
md"""
Now, we can create a (1D) vector of random values...
"""

# ╔═╡ 5dc61a15-48e9-46a2-b893-9fb97ecc8b55
b = rand(10)

# ╔═╡ 0ec44c81-40d5-4b3f-bdec-d3e0cab7de74
md"""
Suppose that we have a linear system $Ax = b$ and we want the solution $x$. We can easily solve this system, using the `\` operator.

The `\` operator gives the multiplication of `b` by the inverse of `A` on the left. It gives floating-point results for integer arguments. It allows you to solve a system of linear equations, and often uses a suitable matrix factorization to solve the problem. We will cover factorizations next.

**Note:** `\` is always the recommended way to solve a linear system. You almost never want to call the `inv` function.
"""

# ╔═╡ 10474148-5a43-4cc1-8e9f-de57bd3019bc
x = A \ b

# ╔═╡ 859cb7ee-d9d8-4bd7-b7c9-5777867297b7
md"""
where `x` is the solution vector.

When there isn't any solution `x` for the system $Ax = b$, the `\` operator will return a vector whose the norm of $Ax-b$ is the minimum least square solution.

**Note:** `A` is a `Matrix` type, and `b` is a `Vector` type. A `Matrix` is a 2D `Array`, and a `Vector` is an 1D `Array`.
"""

# ╔═╡ 9f56f781-5abe-407c-b449-0d917dfa2767
Matrix{Float64} == Array{Float64, 2}

# ╔═╡ 32b59524-9b72-40a2-828f-96f20b0dd924
Vector{Float64} == Array{Float64, 1}

# ╔═╡ 4732c64c-c957-4bdb-9332-7ccb8f40f655
md"""
We can also calculate the norm of a structure with the `norm` function. Let's calculate the norm of $Ax-b$, which should be a number close to zero, since $Ax=b$...
"""

# ╔═╡ b4b29ad0-8573-40ea-a2d8-c25183825149
norm(A*x - b)

# ╔═╡ 8c19fd6c-2f71-480d-8fd0-2241b6f3debe
md"""
## Factorization

A common tool used in Linear Algebra is matrix factorizations. These factorizations are often used to solve linear systems like $Ax=b$, and as we will see later in this tutorial... $Ax=b$ comes up in a lot of Data Science problems.
"""

# ╔═╡ 11bc3a82-19d2-453f-b40b-1eb488ebb323
md"""
### LU factorization

The **Lower–Upper (LU) factorization** (decomposition) factors a matrix as the product of a _lower triangular matrix_ $L$ and an _upper triangular matrix_ $U$. The product sometimes includes a _permutation matrix_ $P$ as well.

In LU factorization (with partial pivoting), we know that $LU=PA$, where $L$ and $U$ are lower and upper matrices and $P$ and $A$ are permutation and original matrices.

In Julia, we can use the `lu` function to perform LU decomposition.
"""

# ╔═╡ 983aeaea-a556-49ba-aedf-898c1b71407d
# LU factorization
LUₐ = lu(A)

# ╔═╡ c8d51d21-ec99-4451-a87e-b51d8f118e24
md"""
We can access lower, upper and permutation matrices...
"""

# ╔═╡ b2b6c7d0-8abb-415e-b668-1fe089b42b04
# lower triangular matrix
Lₐ = LUₐ.L

# ╔═╡ 790d0305-216f-4335-ab37-acff2de0f18f
# upper triangular matrix
Uₐ = LUₐ.U

# ╔═╡ 8bb9722c-9cb6-4316-831b-26cb64f93619
# permutation matrix
Pₐ = LUₐ.P

# ╔═╡ e3571e13-1fa6-44d6-8a31-f810c11cd1f4
# check if LU = PA
Lₐ*Uₐ ≈ Pₐ*A

# ╔═╡ 311b85d4-6509-4945-a451-41020cac465e
# norm should be near zero
norm(Lₐ*Uₐ - Pₐ*A)

# ╔═╡ ea01ba90-0b30-4998-b1a5-e7fa17e53679
md"""
### QR factorization

The **QR factorization**, also known as a QR factorization or QU factorization, is a decomposition of a matrix $A$ into a product $A = QR$ of an _orthogonal matrix_ $Q$ and an _upper triangular matrix_ $R$.

QR decomposition is often used to solve the linear least squares problem and is the basis for a particular eigenvalue algorithm, the QR algorithm.

In Julia, we can use the `qr` function to perform QR decomposition.
"""

# ╔═╡ ada5e3a2-b801-4d2b-8ed8-0b9d8e0cf132
# QR factorization
QRₐ = qr(A)

# ╔═╡ 930c8f5d-a05b-4623-84c3-9b58fe59a721
md"""
We can access both orthogonal and upper triangular matrices...
"""

# ╔═╡ c03c19b1-ec14-4215-9b54-f6cc6b5a2e5c
# orthogonal matrix
Qₐ = QRₐ.Q

# ╔═╡ 7371a76e-ae89-4138-a614-fa41bfb2dfb0
# upper triangular matrix
Rₐ = QRₐ.R

# ╔═╡ da750f88-e883-4bec-8e2e-79a91da23f9b
# check if A = QR
A ≈ Qₐ*Rₐ

# ╔═╡ 2669897d-cd5a-45c6-afe0-7a91153e3686
# norm should be near zero
norm(A - Qₐ*Rₐ)

# ╔═╡ 737762ba-b142-4fd9-9862-af69f7415030
md"""
### Cholesky factorization

The **Cholesky factorization**,  or Cholesky decomposition, is a decomposition of a Hermitian, positive-definite matrix into the product of a _lower triangular matrix_ $L$ and its _conjugate transpose_ $L^t$. In other words, $A = LL^t$ which is useful for efficient numerical solutions.

Cholesky factorization is a type of LU factorization, where $L$ and $U$ matrices are the transposed matrices of each other. That's why we call them $L$ and $L^t$.

**Note:** The input matrix needs to be _symmetric positive definite_.
"""

# ╔═╡ b1ebbd9d-01d5-4eed-85c3-732491322582
# make a positive definite matrix
Aᵖ = A * Aᵗ

# ╔═╡ 6cea7788-57a1-426b-8ede-0a424970e2e3
# check if Aˢ is positive definite
isposdef(Aᵖ)

# ╔═╡ f7a4fffa-a633-45c9-98a5-b25a047a8a3f
# Cholesky factorization
choₐ = cholesky(Aᵖ)

# ╔═╡ c64cf987-9329-4139-b7e3-0fc21e6f0d92
md"""
We can access both lower triangular matrix and its conjugate transpose... 
"""

# ╔═╡ aa195042-2f7e-4881-9d01-834085dccc11
# lower triangular matrix
L = choₐ.L

# ╔═╡ 503192fc-8384-4116-b389-38f04015953c
# lower triangular conjugate transpose
Lᵗ = choₐ.U

# ╔═╡ 61ea5d3c-63aa-480d-8252-0762f586232f
# check A = LLᵗ
Aᵖ ≈ L*Lᵗ

# ╔═╡ 49803952-b650-4e5b-9b65-a0d84b21134b
# norm should be near zero
norm(Aᵖ - L*Lᵗ)

# ╔═╡ 3f0a1c9f-8b8a-4886-97bd-3810dfa0263b
md"""
### The `factorize` function

The `factorize` function compute a _convenient factorization of_ $A$, based upon the type of the input matrix. factorize checks $A$ to see if it is symmetric/triangular/etc. If $A$ is passed as a generic matrix, `factorize` checks every element of $A$ to verify/rule out each property.

|     Properties of `A`    |	    Type of factorization       |
|:------------------------:|:----------------------------------:|
Positive-definite          | Cholesky (see `cholesky`)          |
Dense Symmetric/Hermitian  | Bunch-Kaufman (see `bunchkaufman`) |
Sparse Symmetric/Hermitian | LDLt (see `ldlt`)                  |
Triangular                 | Triangular                         |
Diagonal                   | Diagonal                           |
Bidiagonal                 | Bidiagonal                         |
Tridiagonal	               | LU (see `lu`)                      |
Symmetric real tridiagonal | LDLt (see `ldlt`)                  |
General square	           | LU (see `lu`)                      |
General non-square         | QR (see `qr`)                      |

**Example:** if `factorize` is called on a Hermitian positive-definite matrix, for instance, then `factorize` will return a Cholesky factorization.
"""

# ╔═╡ 433c016e-334a-43fa-b315-608b4bbdef51
factorize(A)

# ╔═╡ 1d9e27be-c21b-4b45-be25-dd1f5b954cb2
md"""
In this case, `factorize` performs the LU factorization, since `A` is a square matrix (no need to perform QR factorization), although it is not a positive definite matrix (cannot perform Cholesky factorization).
"""

# ╔═╡ d806f387-587a-438c-bb22-f6f14b4d4509
md"""
## Some LinearAlgebra.jl functions

With `LinearAlgebra.jl`, we can specify what kind of matrix we want to build. For example, we can use the `diagm` function to create a diagonal matrix whose main diagonal is a vector...
"""

# ╔═╡ 1755c27b-b170-4b5f-acf9-87e722fd6fad
D = diagm([1,2,3,4,5])

# ╔═╡ 10f4304f-06da-4f0a-bfa9-a148169b531f
typeof(D)

# ╔═╡ 926cb115-c47f-44e3-bb40-53533d0a1354
md"""
Although it is a diagonal matrix, it's a `Matrix`. To convert it into a `Diagonal`, we can do...
"""

# ╔═╡ d86cf3cc-0490-4144-b732-41e40f380813
Diagonal(D)

# ╔═╡ 7a7002b3-2523-4b33-8dde-511728dfaf0e
md"""
We can also build a identity matrix with the `I` function by just passing the number of elements...

**Note:** the matrix below is `Diagonal` type.
"""

# ╔═╡ 9980569d-ca11-432e-afb5-33001dea3746
I(3)

# ╔═╡ a31e0b0d-159f-43f3-aed8-a3b7a33bb73b
typeof(I(3))

# ╔═╡ 05bd36f7-b0db-4a58-a591-4cde55139ade
md"""
We can perform $A + I$ and Julia will figure it out the convenient size of $I$.
"""

# ╔═╡ fee37f40-b39d-4cfa-8f25-882a723d5874
A + I

# ╔═╡ 83ba1696-9345-49d8-993a-78517d4ffecb
A + 2I

# ╔═╡ 1d8970d9-7fae-4fef-b4e0-a92aa3e911a7
md"""
## Sparse Linear Algebra

Sparse matrices are stored in Compressed Sparse Column (CSC) form. This form is useful when we are dealing with matrices in which there are very few non-zero elements.

Here, we will use the `SparseArrays.jl` package.

We can create a sparse random matrix with the `sprand` function. The arguments are: number os rows, number of columns and the probability of non-zero values.
"""

# ╔═╡ 512da58a-f8d5-4e0c-8c12-de12cc2a4ba5
S = sprand(5, 8, 0.2)

# ╔═╡ 0293d061-05bc-484f-abcb-76dd4b54fd30
md"""
We can convert it into a regular `Matrix`...
"""

# ╔═╡ f051a1cc-cd3d-465d-9ec0-cb711ad691d7
M = Matrix(S)

# ╔═╡ ab47cafa-3d00-4e7a-8af9-103241fc68c1
md"""
Let's compare the memory allocation of both `SparseMatrixCSC` and `Matrix` types...
"""

# ╔═╡ 8ebe9e1d-981b-4f96-9598-91c8893b8818
"Regular matrix = $(sizeof(M)) bytes."

# ╔═╡ 223cca30-9c72-4ae3-aa51-85c54a362cf4
"Sparse matrix = $(sizeof(S)) bytes."

# ╔═╡ 640da366-e78b-4a62-aedc-2668ae69ae95
md"""
We can access the non-zero elements of the matrix, using the `nzval` attribute. This attribute returns a `Vector` with all non-zero elements.
"""

# ╔═╡ 3a5c505a-0620-4cf4-a546-0a01c191b512
S.nzval

# ╔═╡ 6b140d76-a5c3-4b86-9cbd-116a724851af
md"""
We can access the number of rows with the `m` attribute...
"""

# ╔═╡ 12bdbf11-c352-4243-b080-b28c34527e3b
"There are $(S.m) rows in the matrix."

# ╔═╡ 6d53f3e5-66d1-433f-ad4e-0800b0ecb70f
md"""
And we can also access the number of columns with the `n` attribute...
"""

# ╔═╡ 17764c04-a202-4563-98a4-eddb9f432e6d
"There are $(S.n) columns in the matrix."

# ╔═╡ a441d0b6-9a18-46e4-899d-f3ea2642c7b7
md"""
## Images as matrices

Let's get to the more _"data science-y"_ side. We will do so by working with images (which can be viewed as matrices), and we will use the `SVD` decomposition.

First let's load an image. I chose this image as it has a lot of details.
"""

# ╔═╡ 0f82de90-1a24-49a6-9673-3abf01283817
X₁ = load("data/khiam-small.jpg")

# ╔═╡ 4eb22f0f-b287-4be2-b013-824371a44c26
md"""
An image may be treated as a `Matrix` of `RGB` pixels as we can see below...
"""

# ╔═╡ 3dc31e0a-194c-4ed2-ae6f-c258170aba1a
typeof(X₁)

# ╔═╡ 83c6b744-750c-4c74-8b79-a3d78e5b2c12
md"""
Thus, we can access pixels by slicing an image...
"""

# ╔═╡ b61a2c57-6443-4e8a-b48a-28e2ad1fca4f
# the first pixel
X₁[1,1]

# ╔═╡ 7f003ee7-6958-48e1-a3d8-11876cfa1dbd
# a random chosen pixel
X₁[100,20]

# ╔═╡ c2d97352-fb67-4d66-a999-a8362e41ee84
# first row of pixels
X₁[1,:]

# ╔═╡ 45f5c9ac-d450-49d6-af9d-28704161f360
# first column of pixels
X₁[:,1]

# ╔═╡ 946bf2e5-51e4-4b8b-a316-27e29f15c2e3
md"""
We can easily convert the image to gray scale. This is useful when working with images, since we can deal with a single value instead of a RGB tuple.
"""

# ╔═╡ 69c75582-f4a1-443e-bda2-385681712821
Xgray = Gray.(X₁)

# ╔═╡ 42a1ee17-86ab-41d8-8b89-5064700e2e64
md"""
We can easily extract the RGB layers from the image. We will make use of the `reshape` function below to reshape a vector to a matrix...
"""

# ╔═╡ ff77b413-a905-44d6-973d-f7d1193dbb25
begin
	# red layer
	r_arr = map(i -> X₁[i].r, 1:length(X₁))
	r_mtx = Float64.(reshape(r_arr, size(X₁)))

	# red.(X₁)
end

# ╔═╡ f98fdf05-e6e8-4d0e-b6a6-d3398c2d1ab5
begin
	# green layer
	g_arr = map(i -> X₁[i].g, 1:length(X₁))
	g_mtx = Float64.(reshape(g_arr, size(X₁)))

	# green.(X₁)
end

# ╔═╡ 34b5d940-7d9a-40fb-a38f-16b267130026
begin
	b_arr = map(i -> X₁[i].b, 1:length(X₁))
	b_mtx = Float64.(reshape(b_arr, size(X₁)))

	# blue.(X₁)
end

# ╔═╡ 3b2724d7-f429-447d-948c-e11a8fc20572
md"""
We can now create a matrix of all zeros of equal size as the image and then plot each layer separately...
"""

# ╔═╡ c071dc2a-e53e-402b-b800-6601025b62c7
Z_mtx = zeros(size(r_mtx)...)

# ╔═╡ 5e54ac09-2276-4760-adc9-a05082a5c17f
# red layer
RGB.(r_mtx, Z_mtx, Z_mtx)

# ╔═╡ 33b3de91-010a-4bd2-863d-199bfbfb8ed8
# green layer
RGB.(Z_mtx, g_mtx, Z_mtx)

# ╔═╡ ac394a4c-f983-487c-9d90-1cada3b9e398
# blue layer
RGB.(Z_mtx, Z_mtx, b_mtx)

# ╔═╡ 59daefac-6e47-4ecc-9f4a-033faa55fc59
md"""
We can easily obtain the `Float64` values of the grayscale image.
"""

# ╔═╡ 9d1fa923-e20a-44cc-b0b6-0656f85178fc
Xgrayvalues = Float64.(Xgray)

# ╔═╡ a551a1cb-0d7d-4c82-8f7b-9d92274516f3
md"""
### SVD factorization

The **Singular Value Decomposition (SVD)** is a factorization of a real or complex matrix. It generalizes the eigendecomposition of a square normal matrix with an orthonormal eigenbasis to any $n \times m$ matrix.

The SVD of an $n \times m$ complex matrix $A$ is a factorization of the form $A = U \Sigma V^t$, where $U$ is an $n \times m$ _complex unitary matrix_, $\Sigma$ is an $n \times m$ _rectangular diagonal matrix with singular values_ (non-negative real numbers), and $V$ is an $n \times n$ _complex unitary matrix_.

The SVD is an indicator of how much redundant information you have within your data. For example, if we have just a few non-zero values (i.e. low rank matrix), we may assume that we could store this data in much less memory.

**Note:** _low rank_ means that the matrix has very few linearly independent rows or columns.

We will downsample the grayscale values using the SVD. First, let's obtain the factorization using the function `svd`.
"""

# ╔═╡ d66ae839-2d03-40f6-a82e-1e8a063e27f0
SVD = svd(Xgrayvalues)

# ╔═╡ 54caef76-2f0d-4e9f-a3a1-725ab3d64648
md"""
We may access $U$, $\Sigma$ and $V$...
"""

# ╔═╡ a28364a5-5c03-4b42-b354-2cdb1880ca1d
# rectangular unitary matrix
U = SVD.U

# ╔═╡ 0a30e77d-8770-4b75-9c7d-e45f6d301022
# diagonal matrix with singular values
Σ = diagm(SVD.S)

# ╔═╡ ac8e9a63-708c-408f-bc91-a57356dae648
# transposed square unitary matrix
Vᵗ = SVD.V'

# ╔═╡ 96766fac-6950-4554-b89a-32143f120f04
# check if A = UΣVᵗ
Xgrayvalues ≈ U*Σ*Vᵗ

# ╔═╡ 1de26f9d-89eb-4397-adba-ed566cd054d6
# norm should be near zero
norm(Xgrayvalues - U*Σ*Vᵗ)

# ╔═╡ 90922832-4ed1-4fa6-bb61-1242c17e1cd8
md"""
According to the Eckart–Young theorem, we can get the closest rank estimation to the matrix $A$ by extracting the top $k$ singular values.

Let's try to extract the top 4 singular vectors/values to form a new matrix/image.
"""

# ╔═╡ 8fc6ee4b-5abc-45e5-a642-6570af32a492
begin
	# range of singular values
	r₄ = 1:4
	# first 4 columns of U 
	u₄ = U[:,r₄]
	# first 4 columns of V
	v₄ = SVD.V[:,r₄]
	# Σ with the top 4 singular values
	σ₄ = spdiagm(Σ[r₄])

	# downsampled image
	img₄ = u₄*σ₄*v₄'

	# grayscale final image
	Gray.(img₄)
end

# ╔═╡ ddf9e602-ae0a-43c3-a4b9-c052c513ffc5
md"""
As you can see, it's still far away from the original image. Let's try using 50 singular vectors/values.
"""

# ╔═╡ 91575c63-16b5-4b30-8f50-392c058218c9
begin
	# range of singular values
	r = 1:50
	# first 50 columns of U 
	u = U[:,r]
	# first 50 columns of V
	v = SVD.V[:,r]
	# Σ with the top 50 singular values
	σ = spdiagm(0 => SVD.S[r])

	# downsampled image
	img = u*σ*v'

	# grayscale final image
	Gray.(img)
end

# ╔═╡ 4113a8aa-af35-40fa-836a-f9e347431b5b
md"""
This looks better, even though it's not identical to the original image. We can see this from the norm difference below.
"""

# ╔═╡ eb0f7a49-c6b9-4b94-9fb1-f9cdcf6c8ee6
norm(Xgrayvalues - img)

# ╔═╡ bb25ecf5-b4ed-42a7-8ccd-b67d6d8b8bf0
md"""
### Face recognition

Our next problem will still be related to images, but this time we will solve a simple form of the face recognition problem. Let's get the data first.

**Note:** we can access the data (`Matrix` type) using the notation `data["V2"]`.
"""

# ╔═╡ 716fec98-ef10-465a-a216-beceb4057a91
data = matread("data/face_recog_qr.mat")

# ╔═╡ a9dd7d9e-c91e-4a35-ba77-b765161b58b5
md"""
Each one of the 490 vectors in `data["V2"]` is a fase image. Let's reshape the first one and take a look...
"""

# ╔═╡ 194a229c-0a2f-46a4-bdbe-678df9dae106
begin
	fig₁ = reshape(data["V2"][:,1], 192, 168)
	Gray.(fig₁)
end

# ╔═╡ 049cf246-8dbf-4af8-a60d-01e162097272
md"""
Now we will go back to the vectorized version of this image, and try to select the images that are most similar to it from the "dictionary" matrix.

Let's use `b = q[:]` to be the query image.

**Note:** the notation `[:]` vectorizes a matrix column wise.
"""

# ╔═╡ 70338fd5-570d-4c71-becd-b30a768554cc
b¹ = fig₁[:]

# ╔═╡ f1704b39-f48d-49ce-9ec2-77b1fa2af0f8
md"""
We will now remove the first image from the dictionary. The goal is to find the solution of the linear system $Ax=b$ where $A$ is the dictionary of all images.

In face recognition problem, we really want to minimize the norm differece $norm(Ax-b)$ but the `\` operator actually solves a least squares problem even when the matrix at hand is not invertible.
"""

# ╔═╡ dfbaf736-ccb1-4e7c-a2eb-b432e1498008
begin
	A¹ = data["V2"][:,2:end]
	x¹ = A¹\b¹

	# display image
	Gray.(reshape(A¹*x¹, 192, 168))
end

# ╔═╡ 383a0be3-ce45-43e6-9fe9-d925a40ad9f5
md"""
Now, let's check the norm difference...
"""

# ╔═╡ dd60c14e-d157-48a7-955f-845656636beb
norm(A¹*x¹-b¹)

# ╔═╡ 3f20fcb5-6e94-48e0-b2c3-9ff828269e23
md"""
This was an easy problem. Let's try to make the picture harder to recover. We will add some random error...
"""

# ╔═╡ 50c8a7f4-b620-441f-b941-ece127899ce0
begin
	fig_temp = fig₁ + rand(size(fig₁,1), size(fig₁,2)) * 0.5
	fig₂ = fig_temp ./ maximum(fig_temp)

	# display noisy image
	Gray.(fig₂)
end

# ╔═╡ 6aee282e-e19d-4e8f-bc45-bf20474d41b9
md"""
Let's define $b$ by vectorizing the matrix `fig₂` and then get $A$...
"""

# ╔═╡ 29e803bd-402e-484e-acb7-9053fc8f6a70
b² = fig₂[:]

# ╔═╡ 5511fbe1-901b-4c0f-b354-21795f3316dc
A² = A¹

# ╔═╡ bec4214d-6cbe-43d1-a02a-82d4245bdd27
md"""
Now, let's find the solution $x$ and check the norm difference...
"""

# ╔═╡ 798f5b17-48b4-4349-b850-e3e2e62418de
begin
	x² = A²\b²
	norm(A²*x² - b²)
end

# ╔═╡ ea16d6a3-34ac-4c36-b916-47439fbcd0fa
md"""
The error is so much bigger this time. Finally, let's check the resulting image...
"""

# ╔═╡ 64f6dd6a-0cbf-4f05-b780-c4b2c74ce839
Gray.(reshape(A²*x², 192, 168))

# ╔═╡ Cell order:
# ╟─6c3082f0-6196-11ec-1d58-03f55d178603
# ╟─6ff4cfec-ba3e-4937-bc3a-bf2d22f1a8f6
# ╟─63f2a85b-a426-4c37-a1a0-049d00cfed23
# ╟─a3ead5b3-0251-4054-b6f7-63bf51b5ff71
# ╟─1fe44825-0933-4fb8-9b36-511f0e7ffeea
# ╠═cd961076-6baf-4626-9089-d33b9dbc94ce
# ╟─eaaffef2-903c-49cf-80b4-bb5adf7d8e92
# ╠═92c1a395-bd96-477a-95dc-294a2ac99784
# ╟─5c852935-3055-4178-8ac3-3a6f1d6e8de9
# ╠═de94c2dc-b4b0-42a2-aff0-e54543df82e1
# ╠═4770ca47-71ea-4fe1-a5ad-52c4e31f3bdd
# ╠═eccced3e-5898-4d93-8950-58aa77fc4c0b
# ╟─02cbd31b-bd62-48f4-be5e-242279e215d9
# ╠═89fb6b4c-911b-4a39-bdef-b70dbe17f14f
# ╟─2082cb17-c373-459d-abc1-abe41cfa023a
# ╠═efc88903-189b-48f4-8cde-17bb2c407010
# ╟─e6c8e66b-f9a3-4b93-b5a4-97c030d381d4
# ╠═bb678a13-b45d-410b-b8d5-6346eb839c56
# ╟─41e22cf0-2448-4643-89ea-7315fb8b0b32
# ╠═e759d9a1-29cb-4e15-b629-a913ce721b6f
# ╟─f662e8dd-d998-4e8c-8924-aee9fe633129
# ╠═39797209-2206-495f-b517-2ed219ff47ca
# ╟─f9e97b10-4908-4811-a255-c76c0eaed01c
# ╠═157971c4-04d9-41de-b378-4a9e998725e9
# ╟─d82db838-5ba6-456d-bdeb-cb75a20e081c
# ╠═5dc61a15-48e9-46a2-b893-9fb97ecc8b55
# ╟─0ec44c81-40d5-4b3f-bdec-d3e0cab7de74
# ╠═10474148-5a43-4cc1-8e9f-de57bd3019bc
# ╟─859cb7ee-d9d8-4bd7-b7c9-5777867297b7
# ╠═9f56f781-5abe-407c-b449-0d917dfa2767
# ╠═32b59524-9b72-40a2-828f-96f20b0dd924
# ╟─4732c64c-c957-4bdb-9332-7ccb8f40f655
# ╠═b4b29ad0-8573-40ea-a2d8-c25183825149
# ╟─8c19fd6c-2f71-480d-8fd0-2241b6f3debe
# ╟─11bc3a82-19d2-453f-b40b-1eb488ebb323
# ╠═983aeaea-a556-49ba-aedf-898c1b71407d
# ╟─c8d51d21-ec99-4451-a87e-b51d8f118e24
# ╠═b2b6c7d0-8abb-415e-b668-1fe089b42b04
# ╠═790d0305-216f-4335-ab37-acff2de0f18f
# ╠═8bb9722c-9cb6-4316-831b-26cb64f93619
# ╠═e3571e13-1fa6-44d6-8a31-f810c11cd1f4
# ╠═311b85d4-6509-4945-a451-41020cac465e
# ╟─ea01ba90-0b30-4998-b1a5-e7fa17e53679
# ╠═ada5e3a2-b801-4d2b-8ed8-0b9d8e0cf132
# ╟─930c8f5d-a05b-4623-84c3-9b58fe59a721
# ╠═c03c19b1-ec14-4215-9b54-f6cc6b5a2e5c
# ╠═7371a76e-ae89-4138-a614-fa41bfb2dfb0
# ╠═da750f88-e883-4bec-8e2e-79a91da23f9b
# ╠═2669897d-cd5a-45c6-afe0-7a91153e3686
# ╟─737762ba-b142-4fd9-9862-af69f7415030
# ╠═b1ebbd9d-01d5-4eed-85c3-732491322582
# ╠═6cea7788-57a1-426b-8ede-0a424970e2e3
# ╠═f7a4fffa-a633-45c9-98a5-b25a047a8a3f
# ╟─c64cf987-9329-4139-b7e3-0fc21e6f0d92
# ╠═aa195042-2f7e-4881-9d01-834085dccc11
# ╠═503192fc-8384-4116-b389-38f04015953c
# ╠═61ea5d3c-63aa-480d-8252-0762f586232f
# ╠═49803952-b650-4e5b-9b65-a0d84b21134b
# ╟─3f0a1c9f-8b8a-4886-97bd-3810dfa0263b
# ╠═433c016e-334a-43fa-b315-608b4bbdef51
# ╟─1d9e27be-c21b-4b45-be25-dd1f5b954cb2
# ╟─d806f387-587a-438c-bb22-f6f14b4d4509
# ╠═1755c27b-b170-4b5f-acf9-87e722fd6fad
# ╠═10f4304f-06da-4f0a-bfa9-a148169b531f
# ╟─926cb115-c47f-44e3-bb40-53533d0a1354
# ╠═d86cf3cc-0490-4144-b732-41e40f380813
# ╟─7a7002b3-2523-4b33-8dde-511728dfaf0e
# ╠═9980569d-ca11-432e-afb5-33001dea3746
# ╠═a31e0b0d-159f-43f3-aed8-a3b7a33bb73b
# ╟─05bd36f7-b0db-4a58-a591-4cde55139ade
# ╠═fee37f40-b39d-4cfa-8f25-882a723d5874
# ╠═83ba1696-9345-49d8-993a-78517d4ffecb
# ╟─1d8970d9-7fae-4fef-b4e0-a92aa3e911a7
# ╠═512da58a-f8d5-4e0c-8c12-de12cc2a4ba5
# ╟─0293d061-05bc-484f-abcb-76dd4b54fd30
# ╠═f051a1cc-cd3d-465d-9ec0-cb711ad691d7
# ╟─ab47cafa-3d00-4e7a-8af9-103241fc68c1
# ╠═8ebe9e1d-981b-4f96-9598-91c8893b8818
# ╠═223cca30-9c72-4ae3-aa51-85c54a362cf4
# ╟─640da366-e78b-4a62-aedc-2668ae69ae95
# ╠═3a5c505a-0620-4cf4-a546-0a01c191b512
# ╟─6b140d76-a5c3-4b86-9cbd-116a724851af
# ╠═12bdbf11-c352-4243-b080-b28c34527e3b
# ╟─6d53f3e5-66d1-433f-ad4e-0800b0ecb70f
# ╠═17764c04-a202-4563-98a4-eddb9f432e6d
# ╟─a441d0b6-9a18-46e4-899d-f3ea2642c7b7
# ╠═0f82de90-1a24-49a6-9673-3abf01283817
# ╟─4eb22f0f-b287-4be2-b013-824371a44c26
# ╠═3dc31e0a-194c-4ed2-ae6f-c258170aba1a
# ╟─83c6b744-750c-4c74-8b79-a3d78e5b2c12
# ╠═b61a2c57-6443-4e8a-b48a-28e2ad1fca4f
# ╠═7f003ee7-6958-48e1-a3d8-11876cfa1dbd
# ╠═c2d97352-fb67-4d66-a999-a8362e41ee84
# ╠═45f5c9ac-d450-49d6-af9d-28704161f360
# ╟─946bf2e5-51e4-4b8b-a316-27e29f15c2e3
# ╠═69c75582-f4a1-443e-bda2-385681712821
# ╟─42a1ee17-86ab-41d8-8b89-5064700e2e64
# ╠═ff77b413-a905-44d6-973d-f7d1193dbb25
# ╠═f98fdf05-e6e8-4d0e-b6a6-d3398c2d1ab5
# ╠═34b5d940-7d9a-40fb-a38f-16b267130026
# ╟─3b2724d7-f429-447d-948c-e11a8fc20572
# ╠═c071dc2a-e53e-402b-b800-6601025b62c7
# ╠═5e54ac09-2276-4760-adc9-a05082a5c17f
# ╠═33b3de91-010a-4bd2-863d-199bfbfb8ed8
# ╠═ac394a4c-f983-487c-9d90-1cada3b9e398
# ╟─59daefac-6e47-4ecc-9f4a-033faa55fc59
# ╠═9d1fa923-e20a-44cc-b0b6-0656f85178fc
# ╟─a551a1cb-0d7d-4c82-8f7b-9d92274516f3
# ╠═d66ae839-2d03-40f6-a82e-1e8a063e27f0
# ╟─54caef76-2f0d-4e9f-a3a1-725ab3d64648
# ╠═a28364a5-5c03-4b42-b354-2cdb1880ca1d
# ╠═0a30e77d-8770-4b75-9c7d-e45f6d301022
# ╠═ac8e9a63-708c-408f-bc91-a57356dae648
# ╠═96766fac-6950-4554-b89a-32143f120f04
# ╠═1de26f9d-89eb-4397-adba-ed566cd054d6
# ╟─90922832-4ed1-4fa6-bb61-1242c17e1cd8
# ╠═8fc6ee4b-5abc-45e5-a642-6570af32a492
# ╟─ddf9e602-ae0a-43c3-a4b9-c052c513ffc5
# ╠═91575c63-16b5-4b30-8f50-392c058218c9
# ╟─4113a8aa-af35-40fa-836a-f9e347431b5b
# ╠═eb0f7a49-c6b9-4b94-9fb1-f9cdcf6c8ee6
# ╟─bb25ecf5-b4ed-42a7-8ccd-b67d6d8b8bf0
# ╠═716fec98-ef10-465a-a216-beceb4057a91
# ╟─a9dd7d9e-c91e-4a35-ba77-b765161b58b5
# ╠═194a229c-0a2f-46a4-bdbe-678df9dae106
# ╟─049cf246-8dbf-4af8-a60d-01e162097272
# ╠═70338fd5-570d-4c71-becd-b30a768554cc
# ╟─f1704b39-f48d-49ce-9ec2-77b1fa2af0f8
# ╠═dfbaf736-ccb1-4e7c-a2eb-b432e1498008
# ╟─383a0be3-ce45-43e6-9fe9-d925a40ad9f5
# ╠═dd60c14e-d157-48a7-955f-845656636beb
# ╟─3f20fcb5-6e94-48e0-b2c3-9ff828269e23
# ╠═50c8a7f4-b620-441f-b941-ece127899ce0
# ╟─6aee282e-e19d-4e8f-bc45-bf20474d41b9
# ╠═29e803bd-402e-484e-acb7-9053fc8f6a70
# ╠═5511fbe1-901b-4c0f-b354-21795f3316dc
# ╟─bec4214d-6cbe-43d1-a02a-82d4245bdd27
# ╠═798f5b17-48b4-4349-b850-e3e2e62418de
# ╟─ea16d6a3-34ac-4c36-b916-47439fbcd0fa
# ╠═64f6dd6a-0cbf-4f05-b780-c4b2c74ce839
