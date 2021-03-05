### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ d65d8190-7aa2-11eb-0e64-837b0676b7ca
begin
	using Plots, LaTeXStrings, PrettyTables, DataFrames
	include("tools.jl")
end;

# ╔═╡ 81a9741a-798f-11eb-22fc-7fd2f87480dc
md"""
# Approximation Theory and Applications

A short course on classical approximation theory and connections to numerical solution of PDE and to inference.

Held (virtually) at Peking Normal University, March and April 2021

Christoph Ortner, ortner@math.ubc.ca, University of British Columbia
"""

# ╔═╡ d28341ea-798f-11eb-3819-a3bc46d12e77
md"""
# § 0. Introduction and Motivation

## § 0.1 Composite Trapezoidal Rule

Consider two functions defined on $[-\pi, \pi]$,
```math
	f_1(x) = \frac{1}{1 + x^2}, \qquad
	f_2(x) = \frac{1}{1 + \sin^2(x)}
```
"""

# ╔═╡ 4127884c-7991-11eb-17f4-c76f7a96a556
begin
	f1(x) = 1 / (1 + x^2)
	f2(x) = 1 / (1 + sin(x)^2)
	plot(f1, -π, π, lw=3, label = L"f_1(x) = 1/(1+x^2)")
	plot!(f2, -π, π, lw = 3, label = L"f_2(x) = 1/(1+\sin^2(x))",
		  size = (550, 250), legend = :outertopright )
end

# ╔═╡ 779c0428-7990-11eb-0437-0d930f606312
md"""
We approximate the integral $I[f_j] := \int_{-\pi}^\pi f_j(x) \,dx$ with a quadrature rule, the [composite trapezoidal rule](https://en.wikipedia.org/wiki/Trapezoidal_rule)
```math
\begin{aligned}
	I[f] \approx I_N[f] := \sum_{n = -N+1}^N \frac{2\pi}{2N}
		\cdot \frac{f(x_{n-1})) + f(x_n)}{2}
\end{aligned}
```
where $x_n = 2\pi n / (2N) = \pi n/N, n = -N, \dots, N$ are the quadrature nodes.

If $f \in C^1$ then it is not too difficult to show that on each sub-interval $(x_{n-1}, x_n)$ of length ``h \approx 1/N`` approximating ``f`` with a piecewise affine function yields an ``O(h^2)`` error and therefore the total error is expected to also scale like ``h^2 \approx N^{-2}``, i.e., we expect that
```math
  |I[f] - I_N[f]| \lesssim N^{-2}
```
"""

# ╔═╡ 8de1aef6-7abf-11eb-186a-f98c6e94f5d1
md"""
To test this numerically we implement the quadrature rule rewritten as follows:
```math
	I_N[f] = \frac{\pi}{2N} \big(f(-\pi) + f(\pi)\big)
		+ \frac{\pi}{N} \sum_{n = -N+1}^{N-1} f(\pi n/N)
```
"""

# ╔═╡ 07ad43fe-7993-11eb-28d0-a3a4b0efcf12
trapezoidal_rule(f, N) =  (
	    0.5*π/N * (f(-π) + f(π))
	+       π/N * sum( f(n*π/N) for n = -N+1:N-1 )  );

# ╔═╡ dc5cad34-7992-11eb-3fc4-a54869ad9b9f
begin
	NN = 3:3:30       # number of quadrature points N means 2N+1 points
	I1 = 2 * atan(π)  # exact value of ∫ f₁
	I2 = √2 * π       # exact value of ∫ f₂
	I1N = trapezoidal_rule.(f1, NN)   # trapezoidal rule approximations
	I2N = trapezoidal_rule.(f2, NN)
	E1N = abs.(I1N .- I1)   # errors
	E2N = abs.(I2N .- I2)
end;

# ╔═╡ 21134062-7ab5-11eb-1cdc-856a378f3c7d
md"""##### Convergence Trapezoidal rule
$(
  ata_table( (NN, "``N``", "%d"),
	       (I1N, "``I_N[f_1]``", "%1.5f"),
		   (E1N, "``E_N[f_1]``", "%1.1e"),
	       (I2N, "``I_N[f_2]``", "%1.5f"),
		   (E2N, "``E_N[f_2]``", "%1.1e"),
	        ))
* ``I_N[f]`` : trapezoidal rule approximation with ``2N+1`` quadrature points
* ``E_N[f] := |I_N[f] - \int_{-\pi}^\pi f(x) \,dx |`` : error
"""

# ╔═╡ edf219ca-7abf-11eb-3d52-83c82513ab0d
md"""Plotting the error will give us a more qualitative view ..."""

# ╔═╡ fc2e003a-7abf-11eb-38c3-2f5db67014e3
begin
	Px = plot(NN, E1N, lw=2, m=:o, ms=4, label = L"E_N[f_1]", yaxis = :log10)
	plot!(NN, E2N.+1e-16, lw=2, m=:o, ms=4, label = L"E_N[f_2]",
	      xlabel = L"N")
	P1 = plot!(deepcopy(Px), NN[3:end], 0.04*NN[3:end].^(-2), lw=2, ls=:dash, c=:black,
			   label = L"N^{-2}", ylims = [1e-6, 1e-1], xaxis = :log10)
	P2 = plot!(Px, NN[2:6], 0.1*exp.(-sqrt(pi)*NN[2:6]), lw=2, c=:black, ls = :dash,
			   label = L"e^{-\sqrt{\pi} N}", legend = :right, ylims = [1e-16, 1e-1])
	plot(P1, P2, size = (500, 300))
end

# ╔═╡ 1f961e64-7abf-11eb-0429-5d79965db4ca
md"""
An unexpected outcome? By the end of the first part of this course we should
be able to explain this result.
"""

# ╔═╡ b1d433f0-7ac2-11eb-06ca-5ffe4fbf5bff
md"""
## §0.2 What is Approximation Theory

[Wikipedia:](https://en.wikipedia.org/wiki/Approximation_theory) In mathematics, approximation theory is concerned with how functions can best be approximated with simpler functions, and with quantitatively characterizing the errors introduced thereby. Note that what is meant by best and simpler will depend on the application.

For the purpose of computational mathematics we should start by asking what operations are available to us on a computer: +, -, *, /. Everything else must be built from those. This means that the only functions we can implement immediately are polynomials and rational functions: 
```math
	p_N(x) = a_0 + a_1 x + \dots + a_N x^N, \qquad r_{NM}(x) = \frac{p_N(x)}{q_M(x)}
```
We could (maybe should?) build this entire course based on polynomials. Instead, I decided to use trigonometric polynomials; more on this in the next notebook.

Why should we approximate a general function ``f`` by a polynomial? There are many reasons: 
* ``f`` might be expensive to evaluate, hence replacing it with a cheap but accurate "surrogate" ``p_N`` would give us computationally efficient access to ``f``
* ``f`` might be unknown, but we know it solves some equation, e.g. a PDE, ``L[f] = 0``. We may be able to construct an approximate equation for a polynomial ``L_N[p_N] = 0`` and prove that solving this implies ``p_N \approx f``.
* ``f`` might be unknown but we have some observations (data) about it. We might then "fit" a polynomial ``p_N`` to this data in order to infer further information about ``f``. 

What implicit or explicit assumptions are we making in these tasks? How should we optimize our approximation parameters (the polynomial degree ``N`` in this case). What can we say about the accuracy of approximation, i.e. how close is ``p_N`` to ``f``? How can we optimally sample ``f`` to obtain good approximations? These are the kind of questions that approximation theory and numerical analysis are concerned with.
"""

# ╔═╡ c86ce998-7ac2-11eb-1b8d-6b1702945027
md"""
## §0.3 Course Outline (Tentative)

* §1 : Approximation with Trigonometric Polynomials
    - regularity and approximation rates, Jackson Theorems, Paley-Wiener theorems
    - best approximation and near-best approximation
    - trigonometric interpolation, fast fourier transform
* §2 : Spectral methods for PDEs
    - discretisation of BVPs and IBVPs using trigonometric polynomials
    - examples of error analysis
* §3 : Approximation and least squares
    - approximation from least squares, role of noise
    - max-norm approximation using irlsq
    - inverse problems
* §4 : Approximation in high dimension
    - approximation in ``d`` dimensions using tensor products
    - spectral methods in 2D, 3D
    - curse of dimensionality, sparse grids
* §5 : Extended research talk
    - approximation of symmetric functions
    - the atomic cluster expansion
    - linear regression of ab initio potential energy surfaces
"""

# ╔═╡ 2caa010c-7aa8-11eb-1554-9b8864597364
md"""
## §0.4 Resources

### Julia

* https://julialang.org
* https://juliaacademy.com
* https://juliadocs.github.io/Julia-Cheat-Sheet/

Although you won't need it for this course, I recommend VS Code for serious work with Julia. (I still use Atom myself but most development has now moved to VS Code and I will probably follow soon.)

### Pluto.jl

* https://github.com/fonsp/Pluto.jl
* https://www.wias-berlin.de/people/fuhrmann/SciComp-WS2021/assets/nb01-first-contact-pluto.html
* https://computationalthinking.mit.edu/Spring21/

### Course Material

* https://github.com/cortner/atashort.git
"""

# ╔═╡ Cell order:
# ╟─d65d8190-7aa2-11eb-0e64-837b0676b7ca
# ╟─81a9741a-798f-11eb-22fc-7fd2f87480dc
# ╟─d28341ea-798f-11eb-3819-a3bc46d12e77
# ╟─4127884c-7991-11eb-17f4-c76f7a96a556
# ╟─779c0428-7990-11eb-0437-0d930f606312
# ╟─8de1aef6-7abf-11eb-186a-f98c6e94f5d1
# ╠═07ad43fe-7993-11eb-28d0-a3a4b0efcf12
# ╠═dc5cad34-7992-11eb-3fc4-a54869ad9b9f
# ╟─21134062-7ab5-11eb-1cdc-856a378f3c7d
# ╟─edf219ca-7abf-11eb-3d52-83c82513ab0d
# ╟─fc2e003a-7abf-11eb-38c3-2f5db67014e3
# ╟─1f961e64-7abf-11eb-0429-5d79965db4ca
# ╟─b1d433f0-7ac2-11eb-06ca-5ffe4fbf5bff
# ╟─c86ce998-7ac2-11eb-1b8d-6b1702945027
# ╟─2caa010c-7aa8-11eb-1554-9b8864597364
