### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 74479a7e-8c34-11eb-0e09-73f47f8013bb
begin
	using Plots, LaTeXStrings, PrettyTables, DataFrames, LinearAlgebra,
          PlutoUI, BenchmarkTools, ForwardDiff, Printf, Random, FFTW
	include("tools.jl")
end

# ╔═╡ 8728418e-8c34-11eb-3313-c52ecadbf252
md"""

## §3 Least Squares

The topic of the next set of lectures is least squares regression. Fitting the parameters of a model to general observations about that model is a ubiquitious problem occuring throughout the sciences, engineering and technology. We will first motivate this by showing how we can "discover" the driving force for a simple ODE system by observing it. Then we will move to a simpler setting where we can explore more precisely/rigorously how we can connect least squares methods and approximation theory. Finally, we will see how least squares methods can be tweaked in an unexpected way to solve the difficult max-norm best approximation problem!

* Motivation: an inverse problem
* fitting a trigonometric polynomial to random data
* lsq fitting versus approxmation
* iteratively reweighted lsq for best approximation in the max-norm
"""

# ╔═╡ 09906c00-8e88-11eb-3251-c7f847633d9e
md"""

## §3.1 Motivation: discovering an ODE

cf. notebook `ata_03a_inv.jl`
"""

# ╔═╡ 9202b69e-9013-11eb-02a2-2f1c8e2fcc0c
md"""

## §3.2 Fitting to Point Values

We now consider the vastly simpler question of determining the coefficients (parameters) of a trigonometric polynomial ``t_N \in \mathcal{T}_N`` or ``t_N \in \mathcal{T}_N'`` by minimising the least squares functional
```math
	L(\boldsymbol{c}) := \frac12 \sum_{m = 1}^M \big| t_N(x_j) - f_j \big|^2,
```
where the tuples ``(x_j, f_j)`` are the "training data" and ``L`` is called the loss function.

Trigonometric interpolation can in fact be seen as a special case: if we take ``M = 2N`` and ``x_j`` the interpolation points, the loss can be minimised to achieve ``L(\boldsymbol{c}) = 0`` and the minimiser is precisely the solution of the linear system that defined the minimiser.

But we also want to explore the situation that we *cannot* choose the training data ``(x_j, f_j)`` but it is given to us. An interesting generic case that is in fact close to many real-world scenarios is that the training data is random. That is, we will take
```math
	x_j \sim U(-\pi, \pi), \qquad {\rm iid}
```
Moreover, we assume that the function values ``f_j`` are consistent, i.e. they arise from evaluation of a smooth function ``f(x)`` possibly subject to noise (e.g. due to measurement errors or model errors),
```math
	f_j = f(x_j) + \eta_j, \qquad \eta_j \sim N(0, \sigma), \quad {\rm iid}.
```
Assuming that the noise is normally distributed and iid is a particularly convenient scenario for analysis.
"""


# ╔═╡ 73983438-8c51-11eb-3142-03410d610022
md"""
#### Implementation

Before we can start experimenting with this scenario, we need to discuss how to implement least squares problems. We will only be concerned with the case when ``M \geq 2N`` i.e. there is sufficient data to determine the coefficients (at least in principle).

We begin by rewriting it in terms of the parameters, here for the case ``t_N \in \mathcal{T}_N``. Let ``A_{mk} = e^{i k x_j}`` be the value of the ``k``th basis function at the data point ``x_m``, then
```math
	\begin{aligned}
		L(\boldsymbol{c})
		&=
			\frac12 \sum_{m = 1}^M
			\bigg| \sum_{k = -N}^N c_k e^{i k x_m} - f_m \bigg|^2
		\\
		&=
			\frac12
			\frac12 \sum_{m = 1}^M
			\bigg| \sum_{k = -N}^N  A_{mk} c_k - f_m \bigg|^2
		\\
		&=
			\frac12 \sum_{m = 1}^M \Big| (A \boldsymbol{c})_m - f_m \Big|^2
		\\
		&=
			\frac12 \big\| A \boldsymbol{c} - \boldsymbol{f} \big\|^2,
	\end{aligned}
```
where ``\boldsymbol{f} = (f_m)_{m = 1}^M``.
This is a *linear least-squares system*. The matrix ``A`` is called the *design matrix*.

The first-order criticality condition, ``\nabla L(\boldsymbol{c}) = 0`` takes the form
```math
	A^* A \boldsymbol{c} = A^* \boldsymbol{f}
```
The equations making up this linear system are called the *normal equations*. 

**Lemma:** The least square problem has a unique minimizer if and only if the normal equations have a unique solution if and only if ``A`` has full rank.

We might be tempted to assemble the matrix ``A`` then form ``A^* A`` via matrix multiplication and then solve the system, e.g., using the Cholesky factorisation. This can go very badly since ``{\rm cond}(A^* A) = {\rm cond}(A)^2``, i.e. numerical round-off can become severe. Instead one should normally use the numerically very stable QR factorisation: there exist ``Q \in \mathbb{C}^{M \times 2N}`` and ``R \in \mathbb{C}^{2N \times 2N}`` such that 
```math 
		A = Q R 
```
With that in hand, we can manipulate ``A^* A = R^* Q^* Q R = R^* R`` and hence 
```math 
\begin{aligned} 
	& A^* A \boldsymbol{c} = A^* \boldsymbol{f} \\ 
	%
	\Leftrightarrow \qquad & 
	R^* R \boldsymbol{c} = R^* Q^* \boldsymbol{f} \\ 
	%
	\Leftrightarrow \qquad & 
	R \boldsymbol{c} = Q^* \boldsymbol{f}.
\end{aligned}
```
Moreover, since ``R`` is upper triangular the solution of this system can be performed in ``O(N^2)`` operations.
"""

# ╔═╡ d08fb928-8c53-11eb-3c35-574ef188de6b

# implementation of a basic least squares code
begin
	"""
	note that we now use k = -N,..., N; but we respect the ordering of the FFT	
	"""
	kgridproj(N) = [0:N; -N:-1]
	
	"""
	trigonometric basis consistent with `kgridproj`
	"""
	trigbasis(x, N) = [exp(im * x * k) for k = kgridproj(N)]

	function designmatrix(X, N)
		A = zeros(ComplexF64, length(X), 2*N+1)
		for (m, x) in enumerate(X)
			A[m, :] .= trigbasis(x, N)
		end
		return A
	end

	"""
	Fit a trigonometric polynomial to the data ``X = (x_m), F = (f_m)``.
	"""
	function lsqfit(X, F, N)
		A = designmatrix(X, N)
		return qr(A) \ F   # this performs the  R \ (Q' * F) for us
	end

	
	trigprojeval(x, c) = real(sum(c .* trigbasis(x, (length(c)-1) ÷ 2)))
end

# ╔═╡ fc8495a6-8c50-11eb-14ac-4dbad6baa3c3
md"""
We can explore this situation with a numerical experiment:

Data $M$: $(@bind _M1 Slider(10:10:500; show_value=true))

Degree $N$: $(@bind _N1 Slider(5:100; show_value=true))

Noise $\eta = 10^{p}$; choose $p$: $(@bind _p1 Slider(-5:0; show_value=true))
"""
# $(@bind _eta Slider([0.0001, 0.001, 0.01, 0.1]))

# ╔═╡ 1ba8aa58-8c51-11eb-2d66-775d0fd31747
let N = _N1, M = _M1, σ = 10.0^(_p1), f = x -> 1 / (1 + exp(10*sin(x)))
	Random.seed!(2) # make sure we always produce the same random points
	if M < 2*N+1
		M = 2*N+1
		msg = "M must be >= 2N+1"
	end
	X = 2*π * rand(M)
	F = f.(X) + σ * randn(length(X))
	c = lsqfit(X, F, N)
	xp = range(0, 2π, length = 200)
	plot(xp, f.(xp), lw=4, label = L"f", size = (400, 200),
			title = "N = $N, M = $M",
		 ylims = [-0.3, 1.3])
	P = plot!(xp, trigprojeval.(xp, Ref(c)), lw=2, label = "fit")
	plot!(P, X, F, lw=0, ms=2, m=:o, c=:black, label = "")
end

# ╔═╡ 48175a0c-8e87-11eb-0f42-e9ca0f676e87
md"""
### WARNING

A proper treatment of this subject requires an in-depth computational statistics course. We cannot go into all the subtleties that are required here, such as cross-validation, regularisation, model selection. ... But we are in the age of data science and I *highly* recommend taking some advanced courses on these topics!

Here, we will only explore some approximation-theoretic perspectives on balancing available data with choice of model, i.e. polynomial degree. There is obviously a non-trivial relationship between these. Secondly we will explore what we can say about optimal choice of data points in order to learn about how to choose sampling points if we could choose as we wish.
"""

# ╔═╡ e6cf2c86-9043-11eb-04a7-f1367ad64b6b
md"""
## §3.3 Equispaced Data

As a warm-up we first explore the case when we get to *choose* the training datapoints. We already discussed that without *a priori* knowledge of the function to be fitted we should choose equi-spaced data points. Specifically, let us choose 
```math
	x_m = m\pi/M, \qquad m = 0, \dots, 2M-1.
```
We then evaluate the target function to obtain the training data, ``f_m := f(x_m)`` and minimize the loss 
```math
L(\boldsymbol{c}) = \sum_{m = 1}^{2M-1} \bigg| \sum_{k = -N}^N c_k e^{i k x_m} - f_m \bigg|^2
``` 
to obtain the parameters. With ``M = N`` this is equivalent to trigonometric interpolation (easy exercise), where we even have a fast solver available (FFT). 
"""

# ╔═╡ 82d74ec2-92a1-11eb-0d58-4bb674a8640e
md"""
### Analysis of the lsq system 

Let us write out the loss function explicitly, but weighted, 
```math
	L_M(c) = \frac{1}{2M} \sum_{m = 0}^{2M-1} \big| t_N(x_m) - f(x_m) \big|^2
```
where ``t_N \in \mathcal{T}_N`` has coefficients ``\boldsymbol{c} = (c_k)_{k = -N}^N``. Note that this is the periodic trapezoidal rule approximation of
```math
	L_\infty(c) = \int_{0}^{2\pi} \big|t_N(x) - f(x) \big|^2 \,dx.
```
We know that minimizing ``L_\infty(c)`` gives the best possible ``L^2`` approximation, i.e., if ``c = \arg\min L_\infty`` then 
```math 
	t_N = \Pi_N f = \sum_{k = -N}^N \hat{f}_k e^{i k x}
``` 
and 
```math
	\|t_N - f \|_{L^2} \leq \| t_N' - f  \|_{L^2} \qquad \forall t_N' \in \mathcal{T}_N'.
```
"""

# ╔═╡ 879e9596-90a8-11eb-23d6-935e367eeb17
md"""
Because ``L_M`` is an approximation ``L_\infty`` we can intuit that ``t_N = \arg\min L_M`` will be "close" to ``\Pi_N f`` in some sense. The following result is makes this precise:

**Proposition:** Let ``f \in C_{\rm per}``, and ``t_{NM} = \arg\min_{\mathcal{T}_N} L_M`` with ``M > N``, then ``t_N = \Pi_N I_M f``. In particular, 
```math
	\|t_N - f \|_{L^2} 
	 \leq 
	\| \Pi_N f - f \|_{L^2} + \| \Pi_N (I_M f - f) \|_{L^2}.
```

**Proof:** 
```math 
\begin{aligned}
	L_M(t_N) 
	&= 
	\frac{1}{2M} \sum_{m = 0}^{2M-1} |t_N(x_m) - f(x_m)|^2
	\\ &= 
	\frac{1}{2M} \sum_{m = 0}^{2M-1} |t_N(x_m) - I_M f(x_m)|^2
	\\ &= 
	\| t_N - I_M f \|_{L^2}^2
\end{aligned}
```
by applying for the discrete and then the semi-discrete Plancherel theorem. 
This means that minimising ``L_M`` actually minimizes the ``L^2``-distance to the trigonometric interpolant ``I_M f``, but with ``M > N``, i.e., 
```math
	t_N = \Pi_N I_M f.
```
The stated result now follows easily. ``\square``
"""

# ╔═╡ 3abcfea6-92a2-11eb-061a-d9752403eff8
md"""
**Remark:** We can study the "error term" ``\| \Pi_N (f - I_M f) \|_{L^2}`` in more detail, but it should be intuitive that for ``M \gg N`` it will be much smaller than the best-approximation term ``\| f - \Pi_N f \|_{L^2}``. Importantly, this gives us an overarching strategy to consider when we perform least-squares fits: find an approxmiation error concept, ``\| \bullet - f\|_{L^2}`` that is independent of the data ``(x_m, f_m)`` and which is minimized up to a higher-order term.
"""

# ╔═╡ c4b66e46-90a7-11eb-199e-cded424c7020
md"""
### Fast solver

We now turn to the implementation of the least squares system that we studied in the previous section. While the naive implementation requires ``O(M N^2)`` cost of the QR factorisation, we can use the orthogonality of the trigonometri polynomials to replace this with a matrix multiplication of ``O(MN)`` cost. But the representation 
```math 
	t_{NM} = \Pi_N I_M f 
```
gives us a clue for an even faster O(M \log M) algorithm: 
* Compute ``I_M f`` via the FFT; ``O(M \log M)`` operations
* Obtain ``\Pi_N I_M f`` by only retaining the coefficients ``k = -N, \dots, N``; ``O(N)`` operations.
"""

# ╔═╡ 69fa7e00-90ae-11eb-0681-2d9295ae5368
begin
	function approxL2proj(f, N, M)
		# generate the sample points
		X = range(0, 2π - π/M, length = 2M)
		# the k-grid we obtain from the
		# degree-M trigonometric interpolant
		Km = [0:M; -M+1:-1]
		# contruc the trigonometric interpolant I_M f
		F̂m = fft(f.(X)) / (2M)
		# and find the subset defining Π_N I_M f
		F̂n = [ F̂m[1:N+1]; F̂m[end-N+1:end] ]
	end 
	
	L2err(f, F̂; xerr = range(0, 2π, length=31 * length(F̂))) = 
		sqrt( sum(abs2, f.(xerr) - trigprojeval.(xerr, Ref(F̂))) / length(xerr) )
				
end

# ╔═╡ e3315558-90b5-11eb-3510-e327a6c2d209
md"""
We are now ready to run some numerical tests to confirm our theory. We pick two examples from Lecture 1: 
```math
\begin{aligned}
f_4(x) &= |\sin(2x)|^3 \\
f_7(x) &= \frac{1}{1 + 10*\sin(x)^2} \\
\end{aligned}
```
In truth there is little to explore here, the lsq solutions perform extremely well, even for a low number of training points. Indeed, the trigonometric interpolant itself already comes surprisingly close to the ``L^2``-best approximation. But let us remember that this was just a warm-up case!
"""

# ╔═╡ 8bd7b6fe-91d1-11eb-2d14-134257fa2878
begin
	f4(x) = abs(sin(x))^3
	f7(x) = 1 / (1.0 + 10*sin(x)^2)
	flabels = [L"f_4", L"f_7"]
end;

# ╔═╡ 7f9b5750-91d2-11eb-32ee-6dc5b74d9c0e
let f = f7, NN = 5:5:60, MM = 2*NN
	err = [ L2err(f, approxL2proj(f, N, M)) for (N, M) in zip(NN, MM) ]
	plot(NN, err, lw=3, label = L"f_7", 
		 xlabel = L"N", ylabel = L"\Vert f - \Pi_{NM} f \Vert_{L^2}", 
		 yscale = :log10, size = (350, 230), title = L"f_7~~{\rm analytic}")
	plot!(NN[4:8], exp.(- 1/sqrt(10) * NN[4:8]), lw=2, c=:black, ls = :dash, label = "")
end 

# ╔═╡ 879bf380-91d5-11eb-0b46-85b15d8b2826
let f = f4, NN = (2).^(3:10), MM = 2 * NN
	err = [ L2err(f, approxL2proj(f, N, M)) for (N, M) in zip(NN, MM) ]
	plot(NN, err, lw=3, label = L"f_4", 
		 xlabel = L"N", ylabel = L"\Vert f - \Pi_{NM} f \Vert_{L^2}", 
		 xscale = :log10, yscale = :log10, size = (350, 200), 
		 title = L"f_4 \in C^{2,1}")
	plot!(NN[4:end], NN[4:end].^(-3.5), lw=2, c=:black, ls = :dash, label = "")
end 

# ╔═╡ 8b7e2280-8e87-11eb-0448-9f3a5acf6032
md"""

## §3.3 Random training points

We now return to the case we experimented with at the beginning of this lecture, choose ``x_m \sim U([0, 2\pi])``, iid. While this appears to be a natural choice of random samples, specific applications might lead to different choices. However, it is crucial here. The reason is the following: 

We are trying to approximate 
```math
	f(x) \approx \sum_k c_k B_k(x)
```
where ``B_k`` is a basis of function on ``[0, 2\pi]``. Suppose we sample ``x_m \sim \rho dx`` where ``\rho`` is a general probability density on ``[0, 2\pi]``. The following theory requires that ``\{B_k\}`` is an orthonormal basis with respect to that measure, i.e., 
```math
	\int_{0}^{2\pi} B_k(x) B_{l}(x) \rho(x) dx = \delta_{kl}
```
Since the trigonometric polynomial basis is orthonormal w.r.t. the standard ``L^2``-inner product, i.e., ``\rho(x) = 1/(2\pi)`` we will also sample with respect to that measure. 

If ``x_m`` are distributed according to a different distribution then we need to adjust our basis. We can explore this in the assignment. 
"""

# ╔═╡ d8a55fd0-91dc-11eb-033c-0b10126c5ac7
md"""
The following results are taken from 

	Albert Cohen, Mark A Davenport, and Dany Leviatan. On the stability and accuracy of least squares approximations. Found. Comut. Math., 13(5):819–834, October 2013.

Our first result states stability of the least squares system with high probability:

**Theorem [Stability]:** Let ``x_m \sim U(0, 2\pi)``, idd, and ``A_{mk} = e^{i k x_m}`` then 
```math
	\mathbb{P}\big[ \| A^* A - I \|_{\rm op} \geq 1/2 \big] 
	\leq 2 N \exp\big( - 0.1 M N^{-1} \big)
```
This result is readily interpreted: if ``M \gg N`` then the normal equations are well-conditioned with high probability. In particular this also means that the design matrix ``A`` has full rank and that its ``R``-factor is also well-conditioned.

The second result states a resulting near best approximation error estimate: 

**Theorem [Error]:** Let ``x_m \sim U(0, 2\pi)``, iid and let ``t_{NM}`` denote the resulting degree ``N`` least squares approximant. There exists a constant ``c`` such that, if 
```math
	N \leq \frac{c}{1+r} \frac{M}{\log M} 
```
then 
```math
	\mathbb{E}\big[ \| f - t_{NM} \|_{L^2}^2 \big] 
	\leq
	(1+o(M)) \|f - \Pi_N f \|_{L^2}^2 + 2 \| f \|_{L^\infty}^2 M^{-r}.
```

Similarly as in our introductory example, this result gives us a best-approximation error up to an additional term that depends on how many training points we are given. To properly appreciate it we can show that it implies the following result: 
"""

# ╔═╡ fb0081a6-91df-11eb-00a9-a9deb7581813
md"""
* If ``f`` is continuous(ly differentiable) but not analytic then we expect that ``\|f - \Pi_N f \|_{L^2} \approx N^{-q}`` for some ``q``. In this case, choosing ``M \geq c N \log N`` with any ``c > `` we obtain that ``M^{-r} \lesssim (N \log N)^{-r} \ll N^{-q}``, i.e., 
```math 
	\mathbb{E}\big[ \| f - t_{NM} \|_{L^2}^2 \big]  \lesssim N^{-q}.
```

* If ``f`` is analytic then this is a little trickier: the idea is to choose ``N = c (M / \log M)^a`` for some ``a > 0`` which leads to ``r = c' (M / \log M)^{1-a}`` and hence 
```math
	M^{-r} = \exp\Big( - r \log M \Big) = 
	\exp\Big( - c' M^{1-a} (\log M)^{a} \Big)
```
To ensure this scales the same as 
```math
	\rho^{-N} = e^{-\alpha N} = \exp\Big( - \alpha c (M/\log M)^a \Big)
``` 
we must choose ``1-a = a`` i.e. ``a = 1/2``. That is, we obtain that for a suitable choice of ``c``, and ``N = c (M / \log M)^{1/2}`` we recover the optimal rate 
```math
	\mathbb{E}\big[ \| f - t_{NM} \|_{L^2}^2 \big] \lesssim \rho^{-N}.
```
"""

# ╔═╡ 97a2f8fe-91e0-11eb-2721-9395f949cc48
md"""
Let us again test these predictions numerically.
"""

# ╔═╡ b44e273a-91e0-11eb-1b99-3b20b207513d
begin 
	function lsqfit_rand(f::Function, N::Integer, M::Integer) 
		X = 2*π*rand(M)
		return lsqfit(X, f.(X), N) 
	end 

	L2err_rand(f, N, M; xerr = range(0, 2π, length=31*M)) = 
		sqrt( sum(abs2, f.(xerr) - trigprojeval.(xerr, Ref(lsqfit_rand(f, N, M)))) / (2*M) )
end

# ╔═╡ c479ca34-91e1-11eb-191e-2b8d5f2e8211
let f = f4, NN = (2).^(3:9), MM1 = 2 * NN .+ 1, MM2 = 3 * NN, 
							 MM3 = 2 * ceil.(Int, NN .* log.(NN))
	
	err1 = L2err_rand.(f, NN, MM1)
	err2 = L2err_rand.(f, NN, MM2)
	err3 = L2err_rand.(f, NN, MM3)
	plot(NN, err1, lw=2, label = L"M = 2N + 1", 
		 xlabel = L"N", ylabel = L"\Vert f - \Pi_{NM} f \Vert_{L^2}", 
		 xscale = :log10, yscale = :log10, size = (450, 250), 
		 title = L"f_4 \in C^{2,1}", legend = :outertopright)
	plot!(NN, err2, lw=2, label = L"M = 3N")
	plot!(NN, err3, lw=2, label = L"M = 2N \log N")		
	plot!(NN[4:end], NN[4:end].^(-3.5), lw=2, c=:black, ls = :dash, label = "")
end 

# ╔═╡ 96329376-91e0-11eb-0c0e-5f80723255f8
let f = f7, NN = 5:5:40, MM1 = 2 * NN .+ 1, MM2 = 3*NN,  
							 MM3 = 2 * ceil.(Int, NN.^(1.5))
	err1 = L2err_rand.(f, NN, MM1)
	err2 = L2err_rand.(f, NN, MM2)
	err3 = L2err_rand.(f, NN, MM3)
	plot(NN, err1, lw=2, label = L"M = 2N + 1", 
		 xlabel = L"N", ylabel = L"\Vert f - \Pi_{NM} f \Vert_{L^2}", 
		 yscale = :log10, size = (450, 250), 
		 title = L"f_7 ~~{\rm analytic}", legend = :outertopright)
	plot!(NN, err2, lw=2, label = L"M = 3N")
	plot!(NN, err3, lw=2, label = L"M = 2 N^{3/2}")		
	plot!(NN[4:end], exp.(-1/sqrt(10) * NN[4:end]), lw=2, c=:black, ls = :dash, label = "")
end 

# ╔═╡ 20b6fcfe-93f2-11eb-1b8d-852aeb86d4f8
md"""
In this final example we see a clear gap between theory and practise. Is it just pre-asyptotics? Something about this specific example? Maybe the theory isn't sharp? Or maybe the specific function we are considering has additional properties?
"""

# ╔═╡ 76f189b8-93f2-11eb-259d-d52fea279464
md"""

## Outlook: Algebraic Polynomials

Consider a non-periodic version of our favourite example 
```math
	f(x) = \frac{1}{1 + x^2}, \qquad x \in [-1, 1].
```
Since we are no longer on a periodic domain, let us use algebraic instead of trigonometric polynomials to approximate it, i.e. we seek a polynomial 
```math
p(x) = \sum_{n = 0}^N c_n x^n
```
such that ``p \approx f`` in ``[-1,1]``. Analogously as for trigonometric polynomials, we could try to determine the coefficients via interpolation, 
```math
	p(x_n) = f(x_n),  \qquad x_n = -1 + 2n/N, \qquad n = 0, \dots, N. 
```
here with equispaced nodes. 

This is a bad idea: 
"""

# ╔═╡ e611030a-93f2-11eb-3581-1b3f4b41360a
# Implementation of Runge example 
let f = x -> 1/(1+25*x^2), NN1 = [5, 8, 10], NN2 =  5:5:30
	function poly_fit(N)
		# this is numerically unstable - do not do this!!! 
		# we will learn in the next lecture how to do stable numerical interpolation
		A = [   (-1 + 2*m/N)^n  for m = 0:N, n = 0:N ]
		F = [ f((-1 + 2*m/N)) for m = 0:N ]
		return A \ F 
	end
	# do not do this either, it is neither efficient nor stable!
	poly_eval(x, c) = sum( c[n] * x^(n-1) for n = 1:length(c) )
	
	# first plot 
	xp = range(-1, 1, length=300)
	P1 = plot(xp, f.(xp); lw=4, label = "exact",
			  size = (400, 400), xlabel = L"x")
	for (iN, N) in enumerate(NN1)
		xi = [(-1 + 2*m/N) for m = 0:N]
		c = poly_fit(N)
		plot!(P1, xp, poly_eval.(xp, Ref(c)), c = iN+1, lw=2,label = L"p_{%$(N)}")
		plot!(P1, xi, f.(xi), lw=0, c = iN+1, m = :o, ms=3, label = "")
	end 
	
	# second plot 
	xerr = range(-1, 1, length=3_000)
	err = [ norm( f.(xerr) - poly_eval.(xerr, Ref(poly_fit(N))), Inf )
			for N in NN2 ]
	P2 = plot(NN2, err, lw = 3, label = L"\Vert f - I_N f \Vert", 
			  yscale = :log10, xlabel = L"N", legend = :topleft)
	plot(P1, P2, size = (600, 300), title = "Witch of Agnesi")
end

# ╔═╡ 530b2eb6-93f5-11eb-2c54-6317369a6b21
md"""
Next lecture will cover a range of "random topics" that I won't cover in much (or any) depth but which are both fun and important to have seen once. The first of these will be to explain how all our ideas from trigonometric approximation do carry over to algebraic approximation as long as we take the right perspective. 
"""

# ╔═╡ Cell order:
# ╟─74479a7e-8c34-11eb-0e09-73f47f8013bb
# ╟─8728418e-8c34-11eb-3313-c52ecadbf252
# ╟─09906c00-8e88-11eb-3251-c7f847633d9e
# ╟─9202b69e-9013-11eb-02a2-2f1c8e2fcc0c
# ╟─73983438-8c51-11eb-3142-03410d610022
# ╠═d08fb928-8c53-11eb-3c35-574ef188de6b
# ╟─fc8495a6-8c50-11eb-14ac-4dbad6baa3c3
# ╟─1ba8aa58-8c51-11eb-2d66-775d0fd31747
# ╟─48175a0c-8e87-11eb-0f42-e9ca0f676e87
# ╟─e6cf2c86-9043-11eb-04a7-f1367ad64b6b
# ╟─82d74ec2-92a1-11eb-0d58-4bb674a8640e
# ╟─879e9596-90a8-11eb-23d6-935e367eeb17
# ╟─3abcfea6-92a2-11eb-061a-d9752403eff8
# ╟─c4b66e46-90a7-11eb-199e-cded424c7020
# ╠═69fa7e00-90ae-11eb-0681-2d9295ae5368
# ╟─e3315558-90b5-11eb-3510-e327a6c2d209
# ╠═8bd7b6fe-91d1-11eb-2d14-134257fa2878
# ╠═7f9b5750-91d2-11eb-32ee-6dc5b74d9c0e
# ╠═879bf380-91d5-11eb-0b46-85b15d8b2826
# ╟─8b7e2280-8e87-11eb-0448-9f3a5acf6032
# ╟─d8a55fd0-91dc-11eb-033c-0b10126c5ac7
# ╟─fb0081a6-91df-11eb-00a9-a9deb7581813
# ╟─97a2f8fe-91e0-11eb-2721-9395f949cc48
# ╠═b44e273a-91e0-11eb-1b99-3b20b207513d
# ╠═c479ca34-91e1-11eb-191e-2b8d5f2e8211
# ╠═96329376-91e0-11eb-0c0e-5f80723255f8
# ╟─20b6fcfe-93f2-11eb-1b8d-852aeb86d4f8
# ╟─76f189b8-93f2-11eb-259d-d52fea279464
# ╟─e611030a-93f2-11eb-3581-1b3f4b41360a
# ╟─530b2eb6-93f5-11eb-2c54-6317369a6b21
