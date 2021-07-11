### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ ef92bbdc-9364-11eb-320e-2b65d470e649
begin
	using Plots, LaTeXStrings, PrettyTables, DataFrames, LinearAlgebra,
          PlutoUI, BenchmarkTools, ForwardDiff, Printf, Random, FFTW
	include("../tools.jl")
end

# ╔═╡ 05d9982c-92a5-11eb-1093-83e835ceedb6
md"""
## Assignment 5

(there is no assignment 4)

* When you have completed the assignment please export it as a static HTML (click the triangle/circle symbol in the upper right corner), and then submit it to your TA. She will post instructions how to submit on wechat.
* Please make sure that your `Pluto` server is running within the `atashort` environment. Follow the instructions on the course website on how to ensure this.
* For the submitted assignment all Julia code must be visible, but the MD+Latex code should be hidden.
* Make sure to enter your personal information in the cell below, before starting with the assignment.

Further detailed instructions how to complete the assignment are given below.
"""

# ╔═╡ e82fcfda-a1f4-11eb-313a-c9f00d4b3054
md"""
**REMARK:** This final assignment is a little more challenging than the previous ones in that I am asking you to extend a bit more beyond lecture material, e.g. produce code more independently, extend the code from the lectures more significiantly. I am also giving you less concrete instructions than before. Your best strategy is to go back to the lecture materials, find similar pieces of code and then modify and/or extend them.
"""

# ╔═╡ 17cdc80a-92a5-11eb-0239-7b19f2498447
begin
	# please enter the following information before you start
	NAME = "Hua Luogeng"
	STUDENTID = "1.618033988749"
	COLLABORATORS = "Shing-Tung Yau, Tu Youyou"
end;

# ╔═╡ b8c386fa-92a5-11eb-231d-0d9119d82718
md"""
## Problem 5.1:  [5 + 5]

(a) Consider a function ``f \in C_{\rm per}(\mathbb{R}^3)``. Assume that all one-dimensional slices, 
```math
	x_1 \mapsto f(x_1, \bar x_2, \bar x_3), \quad 
	x_2 \mapsto f(\bar x_1, x_2, \bar x_3), \quad 
	x_3 \mapsto f(\bar x_1, \bar x_2, x_3), 
```
with ``\bar x_1, \bar x_2, \bar x_3`` held fixed, belong to the space ``C^{p, \sigma}`` i.e. are ``p`` times continuously differentiable with the ``p``-th derivative ``\sigma``-Hölder continuous with all slices having a single, uniform, modulus of continuity ``\omega(r) = M r^\sigma``. Prove that 
```math
	\| f - I_N f \|_\infty \leq C (\log N)^3 N^{-p-\sigma}.
```

(b) Produce a numerical example demonstrating that your result is sharp.

*HINT:* To avoid evaluating the interpolant on a large 3D grid, you may wish to estimate the error on a randomly chosen set of points, maybe between 1,000 to 10,000. Be careful not go to too high polynomial degree otherwise your tests will take a very long time to run.
"""

# ╔═╡ 85b37744-92a6-11eb-0ae1-075407098d85
md"""
Enter your solution for part (a) here.

"""

# ╔═╡ 2d884940-a1fc-11eb-3c99-65021f92840c
# copy important functions from the lectures in here that you want to use

# ╔═╡ 973846d2-92a6-11eb-14a6-417320cd1ef6
# implement your solution to part (b) here 

# ╔═╡ 4c6b16c2-a1f9-11eb-05e1-73b5310e1d02
md"""
## Problem 5.2: [10]

Solve the Swift--Hohenberg equation (see e.g. phase field crystal model!)
```math
	\frac{\partial u}{\partial t} = 
	\delta u - (I + \epsilon^2 \Delta)^2 u - u^3
```
for ``x \in (0, 2\pi)^2`` with periodic boundary conditions and for ``t \in [0, 1000]``. The final output of the cell should be the final solution. (If you submit an HTML then you may instead submit the entire animation of the solution.)

Here, ``(I + \epsilon^2  \Delta) u = u + \epsilon^2  \Delta u``. Choose the parameters:
* ``\epsilon = 0.1`` (the spatial scale)
* ``\delta = 0.3``
* Initial condition: each node should take a random value in ``[-0.0005, 0.0005]``.

(For your own entertainment plot not only the final solution, but create an animation using the `@gif` macro as we did in the lectures.)

You are free to develop your own discretisation, but I recommend the following semi-implicit time discretisation
```math
	\frac{u^{n+1} - u^n}{\tau} + (I + \epsilon^2 \Delta)^2 u^{n+1}
		= \delta u^n - (u^n)^3.
```
Now discretize this in space using a Fourier spectral method.
"""

# ╔═╡ f1ec31a4-a1fd-11eb-0264-cf84dde08894
# put more codes from the lectures that you want to use 
# into this cell



# ╔═╡ 75276756-a220-11eb-2edc-d51cd4bdcc28
# insert your solution to Problem 5.2 here


# ╔═╡ 6301164a-a201-11eb-24bc-1f140fb41175
md"""
## Problem 5.3  [5]

A natural extension of analyticity to the multi-variate setting is to require not only that one-dimensional slices are analytic, but that a function ``f \in C_{\rm per}(\mathbb{R}^d)``  can be extended to the complex plane in all ``d`` variables simultaneously. We won't go into the details of this extension, but only mention that one can then prove that such functions belong to a Korobov class of the form
```math
	\mathcal{A}(\rho) 
	:= 
	\Big\{ f \in C_{\rm per}(\mathbb{R}^d)
		\,\big|\, 
		M_f := \sum_{{\bf k} \in \mathbb{Z}^d} \rho^{\| {\bf k} \|_1} |\hat{f}_{\bf k}| < \infty  \Big\},
```
where ``\|{\bf k}\|_1 = \sum_{t = 1}^d |k_t|``.

Suppose that you know ``f`` belongs to such a class, but you don't know the parameter ``\rho``. Propose how you would *a priori* choose a sparse subset ``\mathcal{K} \subset \mathbb{Z}^d`` of the tensor trigonometric basis (multi-)indices. Define the resulting sparse trigonometric polynomial approximant and derive the resulting error estimate in the max-norm.
"""

# (BONUS) Suppose that ``N`` is the largest univariate degree you have chosen in (a), i.e., 
# ```math
# 	N = \max_{{\bf k} \in \mathcal{K}} \max_{t = 1,\dots, d} |k_t|
# ```
# Show that the cardinality of ``\mathcal{K}`` (and hence the computational cost associated with your approximation) is 
# ```math
# 	\# \mathcal{K} = \binom{N + d}{d}
# ```
# *HINT:* This is reasonably straightforward to prove by induction, or very very quickly with the right combinatorial insight.

# ╔═╡ df372380-a206-11eb-0e76-e73cd38a30f9
md"""
Enter your solution for 5.3 here.

"""

# ╔═╡ Cell order:
# ╠═ef92bbdc-9364-11eb-320e-2b65d470e649
# ╟─05d9982c-92a5-11eb-1093-83e835ceedb6
# ╟─e82fcfda-a1f4-11eb-313a-c9f00d4b3054
# ╠═17cdc80a-92a5-11eb-0239-7b19f2498447
# ╟─b8c386fa-92a5-11eb-231d-0d9119d82718
# ╠═85b37744-92a6-11eb-0ae1-075407098d85
# ╠═2d884940-a1fc-11eb-3c99-65021f92840c
# ╠═973846d2-92a6-11eb-14a6-417320cd1ef6
# ╟─4c6b16c2-a1f9-11eb-05e1-73b5310e1d02
# ╠═f1ec31a4-a1fd-11eb-0264-cf84dde08894
# ╠═75276756-a220-11eb-2edc-d51cd4bdcc28
# ╟─6301164a-a201-11eb-24bc-1f140fb41175
# ╠═df372380-a206-11eb-0e76-e73cd38a30f9
