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
## Assignment 3

* When you have completed the assignment please export it as a static HTML (click the triangle/circle symbol in the upper right corner), and then submit it to your TA. She will post instructions how to submit on wechat.
* Please make sure that your `Pluto` server is running within the `atashort` environment. Follow the instructions on the course website on how to ensure this.
* For the submitted assignment all Julia code must be visible, but the MD+Latex code should be hidden.
* Make sure to enter your personal information in the cell below, before starting with the assignment.

Further detailed instructions how to complete the assignment are given below.
"""

# ╔═╡ 17cdc80a-92a5-11eb-0239-7b19f2498447
begin
	# please enter the following information before you start
	NAME = "Confucius"
	STUDENTID = "1.618033988749"
	COLLABORATORS = "Sun Tzu, Laozi"
end;

# ╔═╡ 57799b78-92a5-11eb-07ce-131a96977550
# code from the lectures that you can reuse 
begin
	
	# trigonometric interpolation codes
	# ------------------------------------
	
	xgrid(N) = range(0, 2π-π/N, length=2N)
	kgrid(N) = [0:N; -N+1:-1] 
	triginterp(f, N) = fft(f.(xgrid(N))) / (2N)
	
	# L2 projection codes 
	# ------------------------------------
	
	kgridproj(N) = [0:N; -N:-1]
	
	trigbasis(x, N) = [exp(im * x * k) for k = kgridproj(N)]

	function designmatrix(X, N)
		A = zeros(ComplexF64, length(X), 2*N+1)
		for (m, x) in enumerate(X)
			A[m, :] .= trigbasis(x, N)
		end
		return A
	end

	function lsqfit(X, F, N)
		A = designmatrix(X, N)
		return qr(A) \ F   # this performs the  R \ (Q' * F) for us
	end

	trigprojeval(x, c) = real(sum(c .* trigbasis(x, (length(c)-1) ÷ 2)))
	
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


	function lsqfit_rand(f::Function, N::Integer, M::Integer) 
		X = 2*π*rand(M)
		return lsqfit(X, f.(X), N) 
	end 

	L2err_rand(f, N, M; xerr = range(0, 2π, length=31*M)) = 
		sqrt( sum(abs2, f.(xerr) - trigprojeval.(xerr, Ref(lsqfit(f, N, M)))) / (2*M) )
end

# ╔═╡ f6aab344-92a5-11eb-18d1-efdff2101228
md"""
We already explored in assignment 1 how to define the best approximation in ``L^2``: if ``f`` has Fourier series 
```math
	f(x) = \sum_{k \in \mathbb{Z}} \hat{f}_k e^{i k x} 
```
then the ``L^2``-best approximation trigonometric polynomial of degree ``N`` is simply the ``N``th partial sum, 
```math
\Pi_N f := \sum_{k = -N}^N \hat{f}_N e^{i k x}. 
```
In Problems 3.1 and 3.2 we will explore some (best) approximation error rates
"""

# ╔═╡ b8c386fa-92a5-11eb-231d-0d9119d82718
md"""
## Problem 3.1:

(i) For ``f`` analytic in the strip ``\Omega_\alpha``, prove an approximation rate for ``\|f - \Pi_N f\|_{L^2}``. 

(ii) For ``f(x) = (1 + \exp(10 \sin(x)))^{-1}`` show numerically that your rate is sharp.
"""

# ╔═╡ 85b37744-92a6-11eb-0ae1-075407098d85
md"""
Enter your solution for part (i) here.

"""

# ╔═╡ 973846d2-92a6-11eb-14a6-417320cd1ef6
# implement your solution to part (ii) here 

# ╔═╡ 83c8932c-92a6-11eb-066c-3d61ba7b78a1
md"""
## Problem 3.2

(i) Consider the functions ``f_p(x) = |\sin(x)|^p``, ``p = 1, 3``. Numerically investigate the rate of decay of the fourier coefficients. 

(ii) Deduce the resulting ``L^2``-best-approximation rates for ``f_1, f_3``. 
"""

# (iii) **(Bonus)** Consider ``f_0(x) = {\rm sign}(x)`` repeated ``2\pi``-periodically. Can you deduce an ``L^2``-best approximation rate from (i, ii)?



# ╔═╡ 7d464426-9365-11eb-16ea-e71cde7ac777
# Enter your solution to Part (i) here  ...

# ╔═╡ 847f5a84-9365-11eb-1fcf-e9e465927077
md"""
then state your conclusion from part (i) here
"""

# ╔═╡ 9287c6ea-9365-11eb-0778-419dfa99f617
md"""
Enter your solution to part (ii) here.
"""

# ╔═╡ 364ad784-92a8-11eb-1ffe-b5e00a65317d
md"""
## Problem 3.3 

Consider ``f_1(x) = |\sin(x)|``, produce a sequence of least squares fits ``t_{NM} \in \mathcal{T}_N`` fit to ``M`` datapoints. Ensure that your fit attains the best ``L^2``-approximation rate.

To construct the data, use the command (implemented in a hidden cell)
```julia
X, F = generate_data(M)
```
This will produce random points ``X = (x_m)_{m = 1}^M`` and ``F = (f_1(x_m))_{m = 1}^M``, with ``x_m \sim U(0, 2\pi)``, iid. 
"""

# ╔═╡ c1aa26e8-936b-11eb-3516-dba1bed1e0d8
# DO NOT MODIFY THIS CELL!!!!
begin 
	function generate_data(M)
		Random.seed!(1)
		X = 2*π*rand(M)
		return X, abs.(sin.(X))
	end
end;

# ╔═╡ 4cc4d56a-936a-11eb-1d97-dfb235cb79fc
# enter your solution to Problem 3.3 here


# ╔═╡ 5df54ff8-92a8-11eb-3c38-119fada0294a
md"""
## Problem 3.4  (Bonus)

In a hidden cell below, datapoints `X4, F4` are generated. They are also plotted below. Product a least squares fit to this data with the aim to reproduce the original function from which it is sampled. Plot the fit you have produced on ``(0, 2\pi)``. 

**Remarks:** We have not covered the material to carry out 3.4 properly, so be prepared to do some more reading/thinking! Be careful not to overfit! You will need to carefully think about how to use the data you are given to test the error. Look up concepts such as training and test set, cross validation, etc. 
"""

# ╔═╡ 4816cf0e-9370-11eb-1f5c-737274d1305b
begin
X4 = [3.7123863524902223, 4.817927873339844, 3.557774615766995, 2.8908014638091757, 4.989010676679563, 5.366761306847189, 1.2603192275217214, 1.8762488462013605, 1.5509237841735541, 3.6421880596843383, 4.077045535649395, 0.06852372552009488, 0.41734824985602237, 6.011458676784539, 4.063279273445183, 0.7067695731083345, 1.734290781706025, 4.094526966738957, 0.35589512649008187, 5.294925813896795, 5.972157626335788, 6.061198965244163, 5.942482130170634, 4.963111212705916, 5.159503087969803, 0.21463429175249235, 0.5940405470750193, 1.9787398229795927, 0.803053278840933, 2.351084468768402]
F4 = [0.26093968552643243, 0.9574979519723349, 0.23008293280197242, 0.2103648591804754, 0.7702080578491584, 0.4029212520852062, 0.7281450856762034, 0.7343537290410407, 0.9984230269070353, 0.24518218368398817, 0.41515300302820185, 0.20075292888916427, 0.23026590823702836, 0.2122305832098886, 0.4062444985859044, 0.30182122101796494, 0.9041816566162303, 0.42692621937794806, 0.22151399629633695, 0.45235885700614836, 0.21619915392143704, 0.20806870415647868, 0.2196173028871428, 0.8024254218180432, 0.572161450459671, 0.20753170643094307, 0.26688862697084265, 0.6136645696249089, 0.34136750723775205, 0.3356199940057369]
end;

# ╔═╡ af2ad658-9371-11eb-33a4-fbe08c0e5853
plot(X4, F4, ms=3, m = :o, lw=0, label = "dataset", 
	size = (350, 200), legend = :outertopright)

# ╔═╡ 3ab65276-9370-11eb-1961-67b4b83ba76c
# Implement your solution here

# ╔═╡ Cell order:
# ╟─ef92bbdc-9364-11eb-320e-2b65d470e649
# ╟─05d9982c-92a5-11eb-1093-83e835ceedb6
# ╠═17cdc80a-92a5-11eb-0239-7b19f2498447
# ╠═57799b78-92a5-11eb-07ce-131a96977550
# ╟─f6aab344-92a5-11eb-18d1-efdff2101228
# ╟─b8c386fa-92a5-11eb-231d-0d9119d82718
# ╠═85b37744-92a6-11eb-0ae1-075407098d85
# ╠═973846d2-92a6-11eb-14a6-417320cd1ef6
# ╟─83c8932c-92a6-11eb-066c-3d61ba7b78a1
# ╠═7d464426-9365-11eb-16ea-e71cde7ac777
# ╠═847f5a84-9365-11eb-1fcf-e9e465927077
# ╠═9287c6ea-9365-11eb-0778-419dfa99f617
# ╟─364ad784-92a8-11eb-1ffe-b5e00a65317d
# ╟─c1aa26e8-936b-11eb-3516-dba1bed1e0d8
# ╠═4cc4d56a-936a-11eb-1d97-dfb235cb79fc
# ╟─5df54ff8-92a8-11eb-3c38-119fada0294a
# ╟─4816cf0e-9370-11eb-1f5c-737274d1305b
# ╟─af2ad658-9371-11eb-33a4-fbe08c0e5853
# ╠═3ab65276-9370-11eb-1961-67b4b83ba76c
