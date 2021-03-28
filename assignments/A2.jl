### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 43447f58-85d1-11eb-3f03-1b051b053a55
begin
	using Plots, LaTeXStrings, PrettyTables, DataFrames, LinearAlgebra, PlutoUI, FFTW,
			Printf
	include("../tools.jl")
end

# ╔═╡ 54973c5c-85d1-11eb-10a0-5755fd06dd68
md"""
## Assignment 2

* When you have completed the assignment please export it as a static HTML (click the triangle/circle symbol in the upper right corner), and then submit it to your TA. She will post instructions how to submit on wechat.
* Please make sure that your `Pluto` server is running within the `atashort` environment. Follow the instructions on the course website on how to ensure this.
* For the submitted assignment all Julia code must be visible, but the MD+Latex code should be hidden.
* Make sure to enter your personal information in the cell below, before starting with the assignment.

Further detailed instructions how to complete the assignment are given below.
"""

# ╔═╡ dc1fde3c-85fd-11eb-16ec-fde6cc79e2f7
begin
	# please enter the following information before you start
	NAME = "Go Seigen"
	STUDENTID = "2718281828459"
	COLLABORATORS = "Nie Weiping, Chang Hao"
end;

# ╔═╡ 43554bae-85d2-11eb-13d0-79e9415cd5a0
# Use the following codes from the lectures

begin
	"interpolation nodes"
	xgrid(N) = range(0, 2*π-π/N, length = 2*N)

	"fourier coefficient indices"
	kgrid(N) = [0:N; -N+1:-1]

	"""
	construct the coefficients of the trigonometric interpolant
	"""
	triginterp(f, N) = fft(f.(xgrid(N))) / (2*N)

	"""
	to evaluate a trigonometric polynomial just sum coefficients * basis
	we the take the real part because we assume the function we are
	approximating is real.
	"""
	evaltrig(x, F̂) = sum( real(F̂k * exp(im * x * k))
						  for (F̂k, k) in zip(F̂, kgrid(length(F̂) ÷ 2)) )

end;

# ╔═╡ 6b9bda66-85d1-11eb-2edf-23d00c859783
md"""

This assignment consists of three differential and/or integral equations, each on the domain ``(0, 2\pi]`` and supplied with periodic boundary conditions. For each of them
* formulate and then implement a spectral method solving that problem
* visualize the solution, by producing a publication quality plot, in particular make sure to label the axis and provide a title; make sure to plot the solution on ``[0, 2\pi]`` so that it becomes easy for the TA to check correctness of your solution.
* Aim to ensure that the accuracy of your numerical solution is within 10 digits, and briefly justify why your approximation parameters are sufficient to achieve that. (this need not be rigorous)

In order to be able to test the correctness of your solutions you need to store them in globally accessible variables `Û1, Û2, Û3`. For example, if you implement your code in a `let` block then you can achieve this as follows:
```julia
# cell for solutin to problem 2.1
Û1 = let N = 23784, f = x -> peanuts(x)
	F̂ = triginterp(...)
	# ... some code ...
	# ... some more code ....
	Û = ... # your implementation of that solution must be the last line of the let block
end
```
But you cannot return `Û1` and plot it in the same cell (a restriction of `Pluto.jl`), so intead plot the solution in the next cell
```julia
let Û = Û1
	xp = range(...)
	plot(xp, triginterp.(xp, Ref(Û)), ...)
end
```
"""

# ╔═╡ 572ea648-8aa0-11eb-33f1-f7996f86a02c
md"""
### Problem 2.1

```math
   0.1 * u^{(4)} + 3 u''' - u'' + u = \frac{\sin(x)}{1+\cos^2(2x)}
```
"""

# ╔═╡ 56c8503c-8aa0-11eb-2d32-ef884df99e2d
md""""
Write down the formulation of your numerical method.
"""

# ╔═╡ 81afeb16-85d2-11eb-0edc-b724928dfd5e
# ENTER YOUR SOLUTION HERE
# You may put all codes into a single cell, or create additional cells
# as required. Make sure to make the solution globally accessible as Û1.


# ╔═╡ 9f511a28-8e74-11eb-1f87-c183f5f29c24
md"Hidden Test Cell. Do not edit!!"

# ╔═╡ 64a9ca04-8e70-11eb-1e26-53b2a144d92a
# TEST CODE - DO NOT EDIT THIS CELL!!!
# if this cell throws an error then you haven't made Û1 globally accessible
if !(@isdefined Û1)
	md"""
	WARNING: You haven't made the solution `Û1` globally accessible!
	"""
else
	norm( Û1[ [1, 2, 4, 6, end-2, end-4] ] - [1.9326945442418932e-17 + 0.0im, -0.09266522648167674 - 0.06486565853717369im, 0.0008356537474328632 + 0.00018673250405598489im, -3.0800050569498326e-5 - 7.268811934401587e-6im, 0.0008356537474328632 - 0.00018673250405598489im, -3.0800050569498326e-5 + 7.268811934401589e-6im] )
end

# ╔═╡ cf211bc6-8aa1-11eb-233c-f770b953e1f6
md"""
### Problem 2.2
```math
	- u'' = \sin(\cos(x)) \qquad \text{subject to} \quad \int_{-\pi}^\pi u(x) \, dx = 0.
```
Note that without the integral constraint, this problem does not have a unique solution.
"""

# ╔═╡ 0f3d798e-8aa2-11eb-3639-532a67522aaf
md""""
Write down the formulation of your numerical method here.
"""

# ╔═╡ 115bd9ae-8aa2-11eb-0033-117c016146c4
# ENTER YOUR SOLUTION HERE
# You may put all codes into a single cell, or create additional cells
# as required. Make sure to make the solution globally accessible as Û2

# ╔═╡ 46c19652-8ac4-11eb-1cf0-538e8a4bc5fa
let N = 25, f = x -> sin(cos(x))
	K = kgrid(N)
	F̂ = triginterp(f, N)
	str = @sprintf("%.2e", abs(F̂[1]))
	plot(abs.(K), abs.(F̂), yscale = :log10, label = "",
		 xlabel = L"|k|", ylabel = L"|\hat{F}_k|", size = (300, 150),
		 title = L"|\hat{F}_0| = %$(str)")
end

# ╔═╡ 8e46cbd8-8e74-11eb-306d-37efdb9b527f
md"Hidden Test Cell. Do not edit!!"

# ╔═╡ 086b3cda-8e72-11eb-323f-6d73dfdacbc4
# TEST CODE - DO NOT EDIT THIS CELL!!!
# if this cell throws an error then you haven't made Û2 globally accessible
if !(@isdefined Û2)
	md"""
	WARNING: You haven't made the solution `Û2` globally accessible!
	"""
else
	norm( Û2[ [1, 2, 4, 6, end-2, end-4] ] - [0.0 + 0.0im, 0.44005058574493344 - 4.5991499343720825e-17im, -0.002173705998074266 + 3.3934480921988494e-19im, 9.990309208449343e-6 + 1.8011143436980003e-19im, -0.002173705998074266 - 5.188463235162655e-19im, 9.990309208449295e-6 - 1.6265024400035173e-19im] )
end

# ╔═╡ 1314ccf6-8aa2-11eb-0783-e5a8a61c8e31
md"""
### Problem 2.3
```math
	u(x) + \int_{-\pi}^\pi u(y) g(x - y) \, dy = f(x)
```
where
```math
\begin{aligned}
	f(x) &= \sin^7(x)	\\
	g(x) &= \exp\Big((\cos(x)-1) \Big)
\end{aligned}
```
"""

# ╔═╡ cf664410-8aa3-11eb-2c84-333314a1c1c9
begin
	plot(x -> exp((cos(x)-1)), -π, π, size = (400, 150), lw=2, label = L"g(x)")
	plot!(x -> sin(x)^7, -π, π, lw=2, label = L"f(x)")
end

# ╔═╡ f9af683a-8ab4-11eb-1a4a-f7b804e677cf
md"""
**Hint:** Transform the operator
```math
	u \mapsto \int_{-\pi}^{\pi} u(y) g(x - y) \, dy
```
to the Fourier domain. I.e., how can it be represented in terms of Fourier series (or trigonometric polynomial) coefficients?
"""

# ╔═╡ a5ad134a-8ac3-11eb-19ea-ad981988846e
md""""
Write down the formulation of your numerical method.
"""

# ╔═╡ a9ffdbd0-8ac3-11eb-34cb-6b75b22be21a
# ENTER YOUR SOLUTION HERE
# You may put all codes into a single cell, or create additional cells
# as required. Make sure to make the solution globally accessible as Û3.


# ╔═╡ a4e222c0-8e74-11eb-1418-55e7143af435
md"Hidden Test Cell. Do not edit!!"

# ╔═╡ a607fca6-8e74-11eb-3ee5-735083bcb300
# TEST CODE - DO NOT EDIT THIS CELL!!!
# if this cell throws an error then you haven't made Û2 globally accessible
if !(@isdefined Û3)
	md"""
	WARNING: You haven't made the solution `Û3` globally accessible!
	"""
else
	norm( Û3[ [1, 2, 4, 6, end-2, end-4] ] - [4.970691347583156e-18 + 0.0im, -2.51120670874932e-17 - 0.22637233401190246im, 5.954698439214167e-17 + 0.1627353431907648im, -3.3776223630366365e-17 - 0.05468203913911227im, 7.8004012739961e-17 - 0.16273534319076477im, -4.483521382148193e-17 + 0.05468203913911227im] )
end

# ╔═╡ Cell order:
# ╟─43447f58-85d1-11eb-3f03-1b051b053a55
# ╟─54973c5c-85d1-11eb-10a0-5755fd06dd68
# ╠═dc1fde3c-85fd-11eb-16ec-fde6cc79e2f7
# ╠═43554bae-85d2-11eb-13d0-79e9415cd5a0
# ╟─6b9bda66-85d1-11eb-2edf-23d00c859783
# ╟─572ea648-8aa0-11eb-33f1-f7996f86a02c
# ╠═56c8503c-8aa0-11eb-2d32-ef884df99e2d
# ╠═81afeb16-85d2-11eb-0edc-b724928dfd5e
# ╟─9f511a28-8e74-11eb-1f87-c183f5f29c24
# ╟─64a9ca04-8e70-11eb-1e26-53b2a144d92a
# ╟─cf211bc6-8aa1-11eb-233c-f770b953e1f6
# ╠═0f3d798e-8aa2-11eb-3639-532a67522aaf
# ╠═115bd9ae-8aa2-11eb-0033-117c016146c4
# ╟─46c19652-8ac4-11eb-1cf0-538e8a4bc5fa
# ╟─8e46cbd8-8e74-11eb-306d-37efdb9b527f
# ╟─086b3cda-8e72-11eb-323f-6d73dfdacbc4
# ╟─1314ccf6-8aa2-11eb-0783-e5a8a61c8e31
# ╟─cf664410-8aa3-11eb-2c84-333314a1c1c9
# ╟─f9af683a-8ab4-11eb-1a4a-f7b804e677cf
# ╠═a5ad134a-8ac3-11eb-19ea-ad981988846e
# ╠═a9ffdbd0-8ac3-11eb-34cb-6b75b22be21a
# ╟─a4e222c0-8e74-11eb-1418-55e7143af435
# ╟─a607fca6-8e74-11eb-3ee5-735083bcb300
