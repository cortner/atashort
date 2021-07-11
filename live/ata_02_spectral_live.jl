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

# ╔═╡ 3f1cfd12-7b86-11eb-1371-c5795b87ef5b
begin
	using Plots, LaTeXStrings, PrettyTables, DataFrames, LinearAlgebra, 
		  PlutoUI, BenchmarkTools, ForwardDiff, Printf
	include("../tools.jl")
end;

# ╔═╡ 10164e58-86ab-11eb-24c7-630872bf513d
# to use the fft algorithms, import them as follows:
using FFTW

# ╔═╡ 7d7d43e2-8a71-11eb-2031-f76c30b64f5e
using SIAMFANLEquations  # we use this package to solve nonlinear systems

# ╔═╡ 76e9a7f6-86a6-11eb-2741-6b8759be971b
md"""
## §2 Spectral Methods 

The purpose of the second group of lectures is to study a class of numerical methods for solving differential equations, called *Fourier Spectral Methods*. For example we will learn how to solve periodic boundary value problems such as 
```math
\begin{aligned}
  - u'' + u &= f, \\ 
	u(-\pi) &= u(\pi), \\ 
    u'(-\pi) &= u'(\pi)
\end{aligned}
```
to very high accuracy. 

Before we begin we will develop a fast algorithm to evaluate the trigonometric interpolant, and more generally to convert between nodal values and fourier coefficients.
"""

# ╔═╡ 04b54946-86a7-11eb-1de0-2f61e7b8f790
md"""
## §2.1 The Fast Fourier Transform 

### The Discrete Fourier Transform

Recall from §1 that the trigonometric interpolant ``I_N f`` of a function ``f`` is given by
```math
	I_N f(x) = \sum_{k = -N+1}^{N-1} \hat{F}_k e^{i k x} + \hat{F}_N \cos(N x)
```
and the coefficients are determined by the linear system 
```math
	\sum_{k = -N+1}^N \hat{F}_k e^{i k x_j} = F_j, \qquad j = 0, \dots, 2N-1.
```
where ``F_j = f(x_j)`` and ``x_j = j \pi / N``. We have moreover shown numerically and proved this in A1 that the system matrix is orthogonal (up to rescaling), i.e., if 
```math
	A = \big( e^{i k x_j} \big)_{k,j}
```
then 
```math
	A A^H = 2N I
```
In particular ``A`` is invertible, i.e., the mapping ``F \mapsto \hat{F}, \mathbb{C}^{2N} \to \mathbb{C}^{2N}`` is invertible. 
This mapping is called the discrete fourier transform (DFT) and its inverse is called the inverse discrete fourier transform (IDFT, ``\hat{F} \mapsto F``). Both use a different scaling than we use here; specifically, the most commen definition is 
```math
\begin{aligned}
	{\rm DFT}[G]_k &= \sum_{j = 0}^{2N-1} e^{- i k j \pi / N} G_j, \\ 
	{\rm IDFT}[\hat{G}]_j &= \frac{1}{2N} \sum_{k = -N+1}^N e^{i k j \pi / N} \hat{G}_k.
\end{aligned}
```
This means the the mappings ``F \mapsto \hat{F}, \hat{F} \mapsto F`` can be written as 
```math 
	\hat{F} = (2N)^{-1} \cdot {\rm DFT}[F], \qquad F = 2N \cdot {\rm IDFT}[\hat{F}]
```
"""

# ╔═╡ d57bead8-86b3-11eb-3095-1fee9108c6b1
md"""
The cost of evaluating the DFT and IDFT naively is ``O(N^2)`` (matrix-vector multiplication) but the special structures in the DFT make it possible to evaluate them in ``O(N \log (N))`` operations. This was first observed by Gauss (1876), and much later rediscovered and popularized by [Cooley & Tukey (1965)](https://en.wikipedia.org/wiki/Cooley–Tukey_FFT_algorithm). It is generally considered one of the [most important algorithms of the 20th century](https://www.computer.org/csdl/magazine/cs/2000/01/c1022/13rRUxBJhBm). 

In Julia, the FFT is implemented in the [FFTW package](https://github.com/JuliaMath/FFTW.jl) (the Fastest Fourier Transform in the West). Before we study it, we can try it out:
"""

# ╔═╡ 7b0578ee-8744-11eb-0bf1-555f11fbb0fd
begin
	# let's also define some general utility functions
	
	# the strange (??) ordering of the k-grid is determined by 
	# the convention used for the FFT algorithms
	xgrid(N) = [ j * π / N  for j = 0:2N-1 ]
	kgrid(N) = [ 0:N; -N+1:-1 ]
	
	function dft(F)   # F -> F̂
		N = length(F) ÷ 2
		A = [ exp(im * k * x) for k in kgrid(N), x in xgrid(N) ]
		return (A' * F) / (2*N)
	end
end

# ╔═╡ 13e8fc40-86ab-11eb-1f63-9d2ed7538e7e
let N = 100
	# run a random tests to confirm FFT = DFT
	F = rand(ComplexF64, N)
	norm(dft(F) - fft(F) / N, Inf) 
end

# ╔═╡ c3e57120-86ab-11eb-2268-4f7338540556
let N = 100
	# run a random test to see how fft, ifft work
	F = rand(ComplexF64, N) 
	norm(F - ifft(fft(F)), Inf)
end

# ╔═╡ fc6671fa-8748-11eb-3d6b-e50f405b446f
md"Finally, let's compare the Timing of DFT vs FFT (times in seconds):"

# ╔═╡ 96d114ee-8748-11eb-05f8-a72869439a84
let NN = [5, 10, 20, 40, 80, 160]
	FF = [ rand(ComplexF64, 2*N) for N in NN ]   # random trial vectors 
	times_dft = [ @belapsed dft($F) for F in FF ]
	times_fft = [ @belapsed fft($F) for F in FF ]
	ata_table( (NN, "``N``", "%d"), 
		       (times_dft, "DFT", "%1.2e"), 
		       (times_fft, "FFT", "%1.2e"), 
	           (times_fft./times_dft, "FFT/DFT", "%1.1e"), 
			   )
end

# ╔═╡ 6da69574-86b4-11eb-3300-9b1d62ede475
md"""
What is the idea behind the FFT that gives it such a great performance? Note that the ``O(N \log(N))`` scaling is very close to the theoretically optimal complexity. There are many good references to study the FFT, and there is little point in reproducing this here. But we can at least discuss the main idea of the radix-2 FFT; see whiteboard lecture, and [LN, Sec. 3.6]. We will prove the following result: 

**Theorem:** If ``N = 2^n`` then the DFT (and the IDFT) can be evaluated with ``O(N \log(N))`` operations and ``O(N)`` storage.

With this in hand, we can now rewrite our trigonometric interpolation routines as follows. (Though sometimes we will simply use `fft` and `ifft` directly.)
"""


# ╔═╡ 3c81eca4-86b5-11eb-0e54-d53593b063bc
begin
	"""
	construct the coefficients of the trigonometric interpolant
	"""
	triginterp(f, N) = fft(f.(xgrid(N))) / (2*N)
	
	
	"""
	to evaluate a trigonometric polynomial just sum coefficients * basis
	we the take the real part because we assume the function we are 
	approximating is real.
	"""
	evaltrig(x, F̂) = sum( real(F̂ₖ * exp(im * x * k))
						  for (F̂ₖ, k) in zip(F̂, kgrid(length(F̂) ÷ 2)) )
end 

# ╔═╡ c93dbae2-86b5-11eb-1468-bd709534e1af
md"""
Approximating ``f(x) = \sin(2x) / (0.1 + \cos^2(x))``  

Choose a polynomial degree:  $(@bind _N1 Slider(5:20))
"""

# ╔═╡ a18b061c-86b5-11eb-3c44-0bc846854b1b
let f = x -> sin(2*x) / (0.1 + cos(x)^2)
	xp = range(0, 2*π, length=500)
	X = xgrid(_N1)
	F̂ = triginterp(f, _N1)
	plot(xp, f.(xp), lw=4, label = L"f", size = (500, 300))
	plot!(xp, evaltrig.(xp, Ref(F̂)), lw=2, label = L"I_N f")
	plot!(X, f.(X), lw=0, c=2, m=:o, ms=3, label = "", 
		  title = latexstring("N = $(_N1)"))
end

# ╔═╡ bc30cf3c-86b6-11eb-1f21-ff29b647a839
md"""
Approximating ``f(x) = e^{- |\sin(x)|}``

Choose a polynomial degree:  $(@bind _p2 Slider(2:10))
"""

# ╔═╡ e02f56bc-86b6-11eb-3a66-0d0b94677262
let f = x -> exp( - abs(sin(x)) )
	N2 = 2^_p2
	xp = range(0, 2*π, length=1000)
	X = xgrid(N2)
	F̂ = triginterp(f, N2)
	plot(xp, f.(xp), lw=4, label = L"f", size = (500, 300))
	plot!(xp, evaltrig.(xp, Ref(F̂)), lw=2, label = L"I_N f")
	plot!(X, f.(X), lw=0, c=2, m=:o, ms=3, label = "", 
		  title = latexstring("N = $(N2)"))
end

# ╔═╡ 240250ae-86b7-11eb-1046-7f29472897fd
md"""

## §2.2 Fourier transform of linear homogeneous differential operators

Let 
```math
	t_N(x) = \sum_k \hat{F}_k e^{i k x}
``` 
be a trigonometric polynomial, then 
```math
	t_N'(x) = \frac{d t_N(x)}{dx} = \sum_k \hat{F}_k (i k) e^{i k x}
```
We have two nice properties: 
* If ``t_N \in \mathcal{T}_N`` then ``t_N' = dt_N/dx \in \mathcal{T}_N`` as well.
* If ``t_N \in \mathcal{T}_N'`` then ``t_N'' \in \mathcal{T}_N'`` as well.
* the differentiation of ``t_N`` corresponds to multiplying the Fourier coefficients ``\hat{F}_k`` by ``i k``. 

In other words if we represent a function by its fourier coefficients then we can *exactly* represent differentiation operator ``d/dx`` by a diagonal matrix, 
```math
	\hat{F} \mapsto \hat{d} {\,.\!\!*\,} \hat{F} = \big( i k \hat{F}_k )_{k = -N+1}^N.
```
where ``{\,.\!\!*\,}`` denotes element-wise multiplication. This is an extremely convenient property when we try to discretise a differential equation and extends to general linear homogeneous differential operators: 
```math
	L := \sum_{p = 0}^P a_p \frac{d^p}{dx^p} \qquad \text{becomes} \qquad 
	\hat{L}(k) = \sum_{p = 0}^P a_p (i k)^p.
```
By which we mean that 
```math
	s_N = L f_N, \qquad \Rightarrow \qquad 
	\hat{S}_k =  \hat{L}_k \hat{F}_k.
```


There are other important operators that also become diagonal under the Fourier transform, the most prominent being the convolution operator.

"""

# ╔═╡ 5ebefefe-86b7-11eb-227f-3d5e02a142fd
md"""

## §2.3 Spectral methods for linear homogeneous problems

Let us return to the introductory example, 
```math
	- u'' + u = f, 
```
and imposing periodic boundary conditions. We now perform the following steps: 

* Approximate ``u`` by a trigonometric polynomial ``u_N \in \mathcal{T}_N'``. 
* Approximate ``f`` by a trigonometric polynomial ``f_N \in \mathcal{T}_N'``. 
In real space the equation becomes ``- u_N'' + u_N = f_N``, and expanded 
```math
	\sum_{k = -N+1}^N \hat{U}_k \big[ - (i k)^2 + 1 \big] e^{i kx}
	= \sum_{k = -N+1}^N \hat{F}_k e^{i kx}
```
* Equating coefficients and noting that ``-(ik)^2 = k^2`` we obtain 
```math
	(1 + k^2) \hat{U}_k = \hat{F}_k
```
or, equivalently, 
```math
	\hat{U}_k = (1+k^2)^{-1} \hat{F}_k.
```
This is readily implemented in a short script.
"""

# ╔═╡ 452c65b2-8806-11eb-2d7a-3f4312071cd1
md"""
Polynomial degree:  $(@bind _N2 Slider(5:20, show_value=true))

Right-hand side ``f(x) = ``: $(@bind _fstr2 TextField())
"""

# ╔═╡ 503b45d0-8e65-11eb-0e77-15314d82de1a
_ffun2 = ( _fstr2 == "" ? x -> abs(exp(sin(x) + 0.5 * sin(x)^2)) 
				        : Meta.eval(Meta.parse("x -> " * _fstr2)) );

# ╔═╡ f3c1ba14-8e64-11eb-33ea-4341480e50b3
_Ûex2 = let N = 100, f = _ffun2
		F̂ = triginterp(f, N)
		K = kgrid(N) 
		F̂ ./ (1 .+ K.^2)
	end ;

# ╔═╡ b5359ee2-86de-11eb-1446-b10b9815f448
let N = _N2, f = _ffun2
	F̂ = triginterp(f, N)
	K = kgrid(N) 
	Û = F̂ ./ (1 .+ K.^2)
	xp = range(0, 2π, length=200)
	plot(xp, evaltrig.(xp, Ref(_Ûex2)), lw=4, label = L"u", size = (400, 300), 
		title = L"N = %$N", xlabel = L"x")
	plot!(xp, evaltrig.(xp, Ref(Û)), lw=3, label = L"u_N", size = (400, 300))				
	plot!(xgrid(N), evaltrig.(xgrid(N), Ref(Û)), lw=0, ms=3, m=:o, c=2, label = "")
end 

# ╔═╡ 0c84dcde-86e0-11eb-1877-932742501593
md"""
### Convergence of spectral methods 

What can we say about the convergence of the method? Let us start with a very simple argument, which we will then generalise. The key observation is that our approximate solution ``u_N`` satisfies the full DE but with a perturbed right-hand side ``f \approx f_N``. 
```math
\begin{aligned}
	- u'' + u &= f, \\ 
   - u_N'' + u_N &= f_N.
\end{aligned}
```
Because the differential operator is linear, we can subtract the two lines and obtain  the *error equation*
```math 
   -e_N'' + e_N = f - f_N, 
```
where ``e_N = u - u_N`` is the error. At this point, we have several options how to proceed, but since so far we have studied approximation in the max-norm we can stick with that. We have the following result: 

**Lemma 2.3.1:** If ``- u'' + u = f`` with ``f \in C_{\rm per}`` then 
```math
	\|u \|_\infty \leq C \|f \|_\infty,
```
where ``C`` is independent of ``f, u``.

**Proof:** via maximum principle or Fourier analysis. Note the result is far from sharp, but it is enough for our purposes. Via Fourier analysis you would in fact easily get a much stronger result such as ``\|u\|_\infty \leq C \|f\|_{H^{-1}}`` and even that can still be improved.

Applying Lemma 2.3.1 to the error equation we obtain 
```math
	\| e_N \|_\infty \leq C \|f - f_N \|_\infty.
```
For example, if ``f`` is analytic, then we know that 
```math 
	\|f - f_N \|_\infty \leq M_f e^{-\alpha N}
```
for some ``\alpha > 0``, and hence we will also obtain 
```math 
	\| u - u_N \|_\infty \leq C M_f e^{-\alpha N}. 
```
That is, we have proven that our spectral method converges exponentially fast:

**Theorem 2.3.2:** If ``f`` is analytic then there exist ``C, \alpha > 0`` such that 
```math
	\|u - u_N\|_\infty \leq C e^{-\alpha N}.
```
"""

# ╔═╡ bb33932c-8769-11eb-0fb7-a39703fa96cc
md"""
We can test this numerically using the *method of manufactured solutions*. We start from a solution ``u(x)`` and compute the corresponding right-hand side ``f(x) = - u''(x) + u(x)``. Then we solve the BVP for that right-hand side for increasing degree ``N`` and observe the convergence rate.

Here we choose 
```math
	u(x) = \cos\big( (0.2 + \sin^2 x)^{-1} \big)
```
"""

# ╔═╡ a0e33748-8769-11eb-26b4-416a32282bc2
let NN = 10:10:120, u = x -> cos( 1 / (0.2 + sin(x)^2) )
	du(x) = ForwardDiff.derivative(u, x)
	d2u(x) = ForwardDiff.derivative(du, x)
	f(x) = u(x) - d2u(x)
	xerr = range(0, 2π, length = 1_000)
	solve(N) = triginterp(f, N) ./ (1 .+ kgrid(N).^2)
	error(N) = norm(u.(xerr) - evaltrig.(xerr, Ref(solve(N))), Inf)
	errs = error.(NN) 
	plot(NN, errs, yscale = :log10, lw = 3, m=:o, ms=4, label = "error", 
				  size = (400, 300), xlabel = L"N", ylabel = L"\Vert u - u_N \Vert_\infty")
	plot!(NN[5:9], 1_000*exp.( - 0.33 * NN[5:9]), lw=2, ls=:dash, c=:black, label = "rate")
	hline!([1e-15], lw=2, c=:red, label = L"\epsilon")
end

# ╔═╡ 3fb2d5d5-22c1-4511-94f4-14e9a7e467d8
let
	# _cos(dx::Dual) = (cos(dx.x), - sin(dx.x))
	
	# differentiate cos at x
	x = 0.3
	dx = ForwardDiff.Dual(x, 1.0)
	cos(dx), cos(x), - sin(x)
end

# ╔═╡ 9a6facbe-876b-11eb-060a-7b7e717237be
md"""
Try reaching machine precision with a finite difference or finite element method!
"""

# ╔═╡ f63dcd36-8ac8-11eb-3831-57f0a5088c98
md"""
#### An a priori approach ...

How can we determine a good discretisation parameter *a priori*? Suppose, e.g., that we wish to achieve 10 digits of accuracy for our solution. We know from the arguments above that ``\|u - u_N\|_\infty \leq C \|f - f_N \|_\infty``. We don't know the constant, but for sufficiently simple problems we can legitimately hope that it is ``O(1)``. Hence, we could simply check the convergence of the trigonometric interpolant:
"""

# ╔═╡ 33f2ab24-8ac9-11eb-220e-e71d3ebb93fa
let NN = 40:20:140, u = x -> cos( 1 / (0.2 + sin(x)^2) )
	du = x -> ForwardDiff.derivative(u, x)
	d2u = x -> ForwardDiff.derivative(du, x)
	f = x -> u(x) - d2u(x)
	err = Float64[] 
	xerr = range(0, 2π, length = 1_234)
	for N in NN 
		F̂ = triginterp(f, N) 
		push!(err, norm(f.(xerr) - evaltrig.(xerr, Ref(F̂)), Inf))
	end
	ata_table( (NN, L"N"), (err, L"\|f - f_N\|") )
end

# ╔═╡ a0797ece-8ac9-11eb-2922-e3f0295d4787
md"""
The error stagnates, suggesting that we have reached machine precision at around ``N = 120``. And we have likely achieved an accuracy of ``10^{-10}`` for around ``N = 90``. The error plot shows that in fact we have reached ``10^{-10}`` accuracy already for ``N = 70``, but this is ok. We are only after rough guidance here.

An alternative approach is to look at the decay of the Fourier coefficients. By plotting their magnitude we can get a sense at what degree to truncate. Of course we cannot compute the exact Fourier series coefficients, but the coefficients of the trigonometric interpolant closely approximate them (cf Assignment 1).
"""

# ╔═╡ 057c7508-8aca-11eb-0718-21826e314bc7
let N = 150, u = x -> cos( 1 / (0.2 + sin(x)^2) )
	du = x -> ForwardDiff.derivative(u, x)
	d2u = x -> ForwardDiff.derivative(du, x)
	f = x -> u(x) - d2u(x)
	F̂ = triginterp(f, N) 
	K = kgrid(N)
	plot(abs.(K), abs.(F̂), lw=0, ms=2, m=:o, label ="", 
		 xlabel = L"|k|", ylabel = L"|\hat{F}_k|", 
		 yscale = :log10, size = (350, 200))
	hline!([1e-10], lw=2, c=:red, label = "")
end 


# ╔═╡ 59485024-8aca-11eb-2e65-3962c096e9df
md"""
Entirely consistent with our previous results we observe that 
the Fourier coefficients drop below a value of ``10^{-10}`` 
just below ``|k| = 100``. This gives us a second strong indicator 
that for ``N \geq 100`` we will obtain the desired accuracy of 
``10^{-10}``.
"""

# ╔═╡ c9a6c2ce-876b-11eb-182c-d90997ea2cab
md"""
### General Case 
More generally consider a linear operator equation (e.g. differential or integral equation)
```math
	L u = f 
```
which we discretise as 
```math
	L u_N = f_N
```
and where we assume that it transforms under the DFT as
```math
	\hat{L}_k \hat{U}_k = \hat{F}_k
```
where ``u_N = {\rm Re} \sum_k \hat{U}_k e^{i k x}`` and ``f_N = {\rm Re} \sum_k \hat{F}_k e^{i  x}``.

Now we make two closely related assumptions (2. implies 1.): 
1. ``\hat{L}_k \neq 0`` for all ``k``
2. ``L`` is max-norm stable : ``\| u \|_\infty \leq C \| f \|_\infty``

From 1. we obtain that ``\hat{U}`` and hence ``u_N`` are well-defined. From 2. we obtain 
```math
	\| u - u_N \|_\infty \leq C \| f - f_N \|_\infty
```
and the rate of approximation of the right-hand side will determine the rate of approximation of the solution. We can explore more cases and examples in the assignment.
"""


# ╔═╡ 5b3e4e06-8e6e-11eb-0f31-5546b0ae450a
md"""

### Summary Spectral Methods / Perspective

Numerical Analysis and Scientific Computing for (P)DEs : 
* regularity theory: how smooth are the solutions of the DE?
* approximation theory: how well can we approximate the solution in principle?
* discretisation, error analysis: how do we discretize the DE to guarantee convergence of the discretized solution? Optimal rates as predicted by approximation theory?
* Fast algorithms: scaling of computational cost matters! e.g. FFT provides close to optimal computational complexity, in general this is difficult to achieve. 
"""

# ╔═╡ 6deea4c4-86b7-11eb-0fe7-5d6d0f3007ef
md"""

## §2.4 Spectral methods for time-dependent, inhomogeneous and nonlinear problems

In the following we will implement a few examples that go beyond the basic theory above and showcase a few more directions in which one could explore spectral methods. We will see a few techniques to treat cases for which spectral methods are more difficult to use, namely for differential operators with inhomogeneous coefficients and for nonlinear problems.
"""

# ╔═╡ 7620d684-88fe-11eb-212d-efcbc1803608
md"""
### Wave equation 
```math
	u_{tt} = u_{xx}
```
We first discretise in space, 
```math 
	u_{N,tt} = u_{N, xx},
```
then transform to Fourier coefficients, 
```math
	\frac{d^2\hat{U}_k}{d t^2}  = - k^2 \hat{U}_k,
```
and finally discretize in time
```math 
	\frac{\hat{U}_k^{n+1} - 2 \hat{U}_k^n + \hat{U}_k^{n-1}}{\Delta t^2}
	= - k^2 \hat{U}_k^n
```
"""

# ╔═╡ 42513412-88fb-11eb-2591-c90c16e91d6e
let N = 20, dt = 0.2 / N, Tfinal = 30.0, u0 = x -> exp(-10*(1 + cos(x)))
	xp = xgrid(5 * N) 
	k = kgrid(N)
	Û0 = triginterp(u0, N)
	Û1 = Û0
	# start the time-stepping
	anim = @gif for n = 1:ceil(Int, Tfinal / dt) 
		Û0, Û1 = Û1, (2 .- dt^2 * k.^2) .* Û1 - Û0
		plot(xp, evaltrig.(xp, Ref(Û1)), lw = 3, label = "", size = (400, 300), 
			 xlims = [0, 2*π], ylims = [-0.1, 1.1] )
	end every 5
	anim 
	
end

# ╔═╡ ae55b0e4-88ff-11eb-36f0-152089e43c93
md"""

### Inhomogeneous Transport Equation

```math
	u_t + c(x) u_x = 0
```
First discretise in time using the Leapfrog scheme 
```math
	\frac{u^{n+1} - u^{n-1}}{2 \Delta t} + c (u^n)_x = 0.
```
Now we discretise both ``c`` and ``u^n`` using a trigonometric polynomials, ``c \approx c_N`` and ``u^n \approx u^n_N \in \mathcal{T}_N'``. We can easily apply ``d/dx`` in the Fourier domain, ``\hat{U}_k^n \to (i k) \hat{U}_k^n``, but what can we do with the product ``c_N (u^n_N)_x``? The trick is to differentiate in the Fourier domain, but apply the product in real space, i.e., 
* Apply ``d/dx`` in Fourier space
* Convert back to real space
* apply pointwise multiplication at interpolation nodes
"""

# ╔═╡ 00f2a760-8907-11eb-3ed1-376e9bf97fa8
let N = 256,  dt = π/(4N), tmax = 16.0, 
				cfun = x -> 0.2 + sin(x - 1)^2, 
				  u0 = x ->  exp(-100*(x-1)^2)
	X = xgrid(N) 
	K = kgrid(N) 
	C = cfun.(X)
	# initial condition 
	V = u0.(X) 
	Vx = real.( ifft( (im * K) .* fft(V) ) )
	Vold = V + dt * (C .* Vx)
	
	function plot_soln(t, X, v, c)
		P = plot( xaxis = ([0, 2*π], ), yaxis = ([0.0, 1.5],) )
		plot!(X, 0.5*c, lw=1, c=:black, label = L"c/2")
		plot!(X, v, lw=3, label = L"v", size = (500, 300))
		return P
	end

	
	@gif for t = 0:dt:tmax 
		Vx = real.( ifft( (im * K) .* fft(V) ) )
		V, Vold = Vold - 2 * dt * (C .* Vx), V
		plot_soln(t, X, V, C) 
	end every 20 
end

# ╔═╡ babc0fae-88ff-11eb-0516-6b7841fc0a6a
md"""

### Nonlinear BVP

Steady state viscous Burgers equation
```math
		u u_x = \epsilon u_{xx} - 0.1 \sin(x)
```
We write a nonlinear system 
```math
	F_j := u_N(x_j) u_{N,x}(x_j) - \epsilon u_{N,xx}(x_j) + 0.1 sin(x)
```
and use a generic nolinear solver to solve
```math
	F_j = 0, \qquad j = 0, \dots, 2N-1.
```
This is not a magic bullet, often one needs specialized tools to solve these resulting nonlinear systems.
"""


# ╔═╡ f2dab6a0-890a-11eb-1e48-a747d18f6c93
let N = 30, ϵ = 0.1
	function burger(U)
		N = length(U) ÷ 2 
		K = kgrid(N) 
		Ux = real.( ifft( (im * K) .* fft(U) ) )
		Uxx = real.( ifft( (- K.^2) .* fft(U) ) )
		return U .* Ux - ϵ * Uxx + 0.1 * sin.(xgrid(N))
	end
	
	U0 = sin.(xgrid(N))
	# burger(U) == 0 !!!
	U = nsoli(burger, U0, maxit = 1_000)
	norm(burger(U), Inf)
	Û = fft(U) / (2N)
	plot(x -> evaltrig(x, Û), -π, π, lw=3, size = (400, 150), 
	     label = "Residual = " * @sprintf("%.2e\n", norm(burger(U), Inf)), 
		 xlabel = L"x", ylabel = L"u(x)")
end 

# ╔═╡ 7b5b1e6c-86b7-11eb-2b32-610393a24da4
md"""

## §2.5 Outlook: PDEs in higher dimension

Just one example; more in §4 of this course.

### 2D Wave equation 

```math
	u_{tt} = u_{xx} + u_{yy}
```
Discrete fourier transform for ``u(x, y)`` becomes ``\hat{U}_{k_1 k_2}``. 
After discretising in time and space, and transforming to the Fourier domain, 
```math
	\frac{\hat{U}_{k_1k_2}^{n+1} - 2 \hat{U}_{k_1k_2}^n + \hat{U}^{n-1}_{k_1 k_2}}{\Delta t^2} 
	=  -(k_1^2 + k_2^2) \hat{U}_{k_1 k_2}.
```
"""

# ╔═╡ b90093f6-8906-11eb-2e69-6d4b807866a4
let N = 64, u0 = (x, y) -> exp(-10*(1 + cos(x))) * exp.(-10*(1 + cos(y)))
	x = xgrid(N); Xx = kron(x, ones(2*N)'); Xy = Xx'
	k = kgrid(N); Kx = kron(k, ones(2*N)'); Ky = Kx'
	U0 = u0.(Xx, Xy)
	Û0 = fft(U0)
	Û1 = Û0  # zero initial velocity 
	dt = 0.2 / N 
	@gif for n = 1:4_000
		Û0, Û1 = Û1, 2 * Û1 - Û0 - dt^2 * (Kx.^2 + Ky.^2) .* Û1 
		Plots.surface(x, x, real.(ifft(Û1)), zlims = [-0.3, 0.3], color=:viridis, 
					  size = (400, 300))
	end every 5
end

# ╔═╡ Cell order:
# ╠═3f1cfd12-7b86-11eb-1371-c5795b87ef5b
# ╟─76e9a7f6-86a6-11eb-2741-6b8759be971b
# ╟─04b54946-86a7-11eb-1de0-2f61e7b8f790
# ╟─d57bead8-86b3-11eb-3095-1fee9108c6b1
# ╠═10164e58-86ab-11eb-24c7-630872bf513d
# ╠═7b0578ee-8744-11eb-0bf1-555f11fbb0fd
# ╠═13e8fc40-86ab-11eb-1f63-9d2ed7538e7e
# ╠═c3e57120-86ab-11eb-2268-4f7338540556
# ╟─fc6671fa-8748-11eb-3d6b-e50f405b446f
# ╟─96d114ee-8748-11eb-05f8-a72869439a84
# ╟─6da69574-86b4-11eb-3300-9b1d62ede475
# ╠═3c81eca4-86b5-11eb-0e54-d53593b063bc
# ╟─c93dbae2-86b5-11eb-1468-bd709534e1af
# ╟─a18b061c-86b5-11eb-3c44-0bc846854b1b
# ╟─bc30cf3c-86b6-11eb-1f21-ff29b647a839
# ╟─e02f56bc-86b6-11eb-3a66-0d0b94677262
# ╟─240250ae-86b7-11eb-1046-7f29472897fd
# ╟─5ebefefe-86b7-11eb-227f-3d5e02a142fd
# ╟─452c65b2-8806-11eb-2d7a-3f4312071cd1
# ╟─503b45d0-8e65-11eb-0e77-15314d82de1a
# ╟─f3c1ba14-8e64-11eb-33ea-4341480e50b3
# ╟─b5359ee2-86de-11eb-1446-b10b9815f448
# ╟─0c84dcde-86e0-11eb-1877-932742501593
# ╟─bb33932c-8769-11eb-0fb7-a39703fa96cc
# ╟─a0e33748-8769-11eb-26b4-416a32282bc2
# ╠═3fb2d5d5-22c1-4511-94f4-14e9a7e467d8
# ╟─9a6facbe-876b-11eb-060a-7b7e717237be
# ╟─f63dcd36-8ac8-11eb-3831-57f0a5088c98
# ╟─33f2ab24-8ac9-11eb-220e-e71d3ebb93fa
# ╟─a0797ece-8ac9-11eb-2922-e3f0295d4787
# ╟─057c7508-8aca-11eb-0718-21826e314bc7
# ╟─59485024-8aca-11eb-2e65-3962c096e9df
# ╟─c9a6c2ce-876b-11eb-182c-d90997ea2cab
# ╟─5b3e4e06-8e6e-11eb-0f31-5546b0ae450a
# ╟─6deea4c4-86b7-11eb-0fe7-5d6d0f3007ef
# ╟─7620d684-88fe-11eb-212d-efcbc1803608
# ╠═42513412-88fb-11eb-2591-c90c16e91d6e
# ╟─ae55b0e4-88ff-11eb-36f0-152089e43c93
# ╠═00f2a760-8907-11eb-3ed1-376e9bf97fa8
# ╟─babc0fae-88ff-11eb-0516-6b7841fc0a6a
# ╠═7d7d43e2-8a71-11eb-2031-f76c30b64f5e
# ╠═f2dab6a0-890a-11eb-1e48-a747d18f6c93
# ╟─7b5b1e6c-86b7-11eb-2b32-610393a24da4
# ╠═b90093f6-8906-11eb-2e69-6d4b807866a4
