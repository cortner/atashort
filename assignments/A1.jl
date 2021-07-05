### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 43447f58-85d1-11eb-3f03-1b051b053a55
begin
	using Plots, LaTeXStrings, PrettyTables, DataFrames, LinearAlgebra, PlutoUI
	include("../tools.jl")
end

# ╔═╡ 54973c5c-85d1-11eb-10a0-5755fd06dd68
md"""
## Assignment 1

* When you have completed the assignment please export it as a static HTML (click the triangle/circle symbol in the upper right corner), and then submit it to your TA. She will post instructions how to submit on wechat.
* Please make sure that your `Pluto` server is running within the `atashort` environment. Follow the instructions on the course website on how to ensure this.
* For the submitted assignment all Julia code must be visible, but the MD+Latex code should be hidden.
* Make sure to enter your personal information in the cell below, before starting with the assignment.

Further detailed instructions how to complete the assignment are given below. 
"""

# ╔═╡ dc1fde3c-85fd-11eb-16ec-fde6cc79e2f7
begin 
	# please enter the following information before you start	
	NAME = "Ping Xiao Po"
	STUDENTID = "314159265"
	COLLABORATORS = "Chan Kong-sang, Lee Jun-fan, Li Lianjie"
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
	function triginterp(f, N)
		X = xgrid(N)
		# nodal values at interpolation nodes
		F = f.(X) 
		# system matrix
		A = [ exp(im * x * k) for k in kgrid(N), x in X ]
		# coefficients are given by F̂ = A' * F as discussed above!
		return (A' * F) / (2*N)
	end 
	
	
	"""
	to evaluate a trigonometric polynomial just sum coefficients * basis
	we the take the real part because we assume the function we are 
	approximating is real.
	"""
	evaltrig(x, F̂) = sum( real(F̂k * exp(im * x * k))
						  for (F̂k, k) in zip(F̂, kgrid(length(F̂) ÷ 2)) )
	
	"""
	implementation of a naive error function
	"""
	function triginterperror(f, N; Nerr = 1_362)
		xerr = range(0, 2π, length=Nerr)    # compute error on this grid
		F̂ = triginterp(f, N)                # trigonometric interpolant
		return norm(f.(xerr) - evaltrig.(xerr, Ref(F̂)), Inf)  # max-error on xerr grid
	end	
		
end;

# ╔═╡ 6b9bda66-85d1-11eb-2edf-23d00c859783
md"""

### Problem 1.1 [4]

For the following two functions ``g \in \{g_1, g_2\}``
* construct the trigonometric interpolant ``I_N g`` of degree ``N``
* numerically demonstrate the convergence as ``N \to \infty``
* explain the observed rate of convergence by reference to the smoothness of ``g``

```math
\begin{aligned}
    g_1(x) &= \frac{1}{1 + \sin^4 x + \frac{1}{4} \sin^8 x} \\ 
	g_2(x) &= x \sin^2(x)
\end{aligned}
```
where ``g_2`` is defined on ``[-\pi, \pi]`` and extended ``2\pi``-periodically. 
"""

# ╔═╡ 5c0a9cdc-85d2-11eb-38fe-d3d36f1b2376
# You may wish to start from the following example code copy-pasted 
# from the lectures and modify it as needed.

# let NN = [1:2:5; (2).^(3:7)], flist = allf, flabels = flabels
# 	P = plot( xaxis  = (L"N~{\rm (degree)}", ),
#     	      yaxis  = (:log, L"\Vert f - I_N f ~\Vert_{\infty}"), 
#         	  legend = :outertopright, 
# 			  xscale = :log10, yscale = :log10, 
# 			  size = (500, 300))
# 	for (f, lab) in zip(flist, flabels)
#     	err = triginterperror.(f, NN)
# 		plot!(P, NN, err, lw=2, m=:o, ms=3, label = lab)
# 	end
# 	hline!(P, [1e-15], c = :red, lw=3, label = L"\epsilon")
# end 

# ╔═╡ 81afeb16-85d2-11eb-0edc-b724928dfd5e
# ENTER YOUR SOLUTION HERE 
# You may put all codes into a single cell, or create additional cells
# as required.


# ╔═╡ 8e5707f0-85d2-11eb-094b-77b3288efaa8
md"""

Provide a written explanation for the results you observe. Try to justify your arguments rigorously. Alternatively, or in addition, use computational experiments / visualisation to explain your results.
"""

# ╔═╡ 42877f90-88e2-11eb-12f5-efff00cd2db1
md"""
**ANSWER:** Enter your explanation here.

"""

# ╔═╡ b6f731c6-85d2-11eb-23c5-15457814f0e0
md"""

### Problem 1.2 [4] 
Let ``f \in C_{\rm per}``. 
1. Prove that 
```math
	\frac{1}{2\pi} \int_{-\pi}^\pi e^{i k x} e^{-i l x} \,dx = \delta_{kl}.
```

2. Assume that ``f`` is given by its *Fourier series*, 
```math
	f(x) = \sum_{k \in \mathbb{Z}} \hat{f}_k e^{i k x}
```
and that this series converges absolutely (and hence uniformly! Why?). Using step 1 show that the *Fourier coefficients* are given by 
```math
	\hat{f}_k = \frac{1}{2\pi} \int_{-\pi}^\pi f(x) e^{-i k x} \, dx. 
```


3. Assume again that ``f`` is given by its Fourier series. Using step 1 again prove the Plancherel theorem,
```math

	\|f\|_{L^2} = \bigg( \int_{-\pi}^\pi |f(x)|^2 \,dx \bigg)^{1/2} 
	= \bigg(2\pi \sum_{k \in \mathbb{Z}} |\hat{f}_k|^2\bigg)^{1/2} =: \sqrt{2\pi} \|\hat{f}\|_2.
```

4. Still assuming that ``f`` is given by its Fourier series define the approximation operator (``L^2``-projection)
```math
	\Pi_N f(x) := \sum_{k = -N}^N \hat{f}_k e^{i k x}
```
Using step 3 deduce that this is the best approximation in the ``L^2``-norm, i.e., 
```math
	\| f - \Pi_N f\|_{L^2} = \inf_{t_N \in \mathcal{T}_N} \| f - t_N \|_{L^2}.
```

5. **[BONUS]** Have we proven that all continuous functions can be written as a Fourier series? What step is missing? Can you complete it? 
"""

# ╔═╡ d37cf0fa-85d3-11eb-1d1c-b9d1cd764e9e
md"""
**Answer:**

ENTER YOUR SOLUTION HERE USING MARKDOWN AND LATEX

"""

# ╔═╡ de81c3b8-85d3-11eb-072c-17aa0e22db82
md"""

## Problem 1.3 [2] 

1. Prove the discrete analogue of Problem 1.2(1),
```math
	\sum_{j = 0}^{2N-1} e^{i k x_j} e^{- i l x_j} = 2N \delta_{kl}, \qquad k, l = -N+1, \dots N.
```
where ``x_j = \pi j / N``. 

2. We are using this in our implementation of the trigonometric interpolation operator. Where?

3. **[BONUS]** Can you extend the calculation of step 1 a little bit, and then apply The Paley--Wiener theorem to explain the fast convergence of the composite trapezoidal rule for periodic functions we saw in the introduction?
"""

# ╔═╡ 68833984-85d4-11eb-3f73-ed3ee969d0b5
md"""

**Answer:**

ENTER YOUR SOLUTION HERE USING MARKDOWN AND LATEX

"""

# ╔═╡ Cell order:
# ╟─43447f58-85d1-11eb-3f03-1b051b053a55
# ╟─54973c5c-85d1-11eb-10a0-5755fd06dd68
# ╠═dc1fde3c-85fd-11eb-16ec-fde6cc79e2f7
# ╠═43554bae-85d2-11eb-13d0-79e9415cd5a0
# ╟─6b9bda66-85d1-11eb-2edf-23d00c859783
# ╠═5c0a9cdc-85d2-11eb-38fe-d3d36f1b2376
# ╠═81afeb16-85d2-11eb-0edc-b724928dfd5e
# ╟─8e5707f0-85d2-11eb-094b-77b3288efaa8
# ╠═42877f90-88e2-11eb-12f5-efff00cd2db1
# ╠═b6f731c6-85d2-11eb-23c5-15457814f0e0
# ╠═d37cf0fa-85d3-11eb-1d1c-b9d1cd764e9e
# ╟─de81c3b8-85d3-11eb-072c-17aa0e22db82
# ╠═68833984-85d4-11eb-3f73-ed3ee969d0b5
