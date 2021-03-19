### A Pluto.jl notebook ###
# v0.12.21

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

Instructions how to complete the assignment are given below. When you have completed the assignment please export it as a static HTML, and submit it to your TA.

For the submitted assignment all Julia code must be visible, but the MD+Latex code should be hidden.

Make sure to enter your personal information in the cell below, before starting with the assignment.
"""

# ╔═╡ dc1fde3c-85fd-11eb-16ec-fde6cc79e2f7
begin 
	# please enter the following information	
	NAME = "Mickey Mouse"
	STUDENTID = "314159265"
	COLLABORATORS = "Jackie, Jet, Po"
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
	implementation of a basic error function 
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
# You may wish to start from the following example code copy-pasted from the lectures.
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


# ╔═╡ 1d296dd8-85d8-11eb-3c7f-fd170297a8df
let NN = 3:3:30, f = x -> 1 / (1 + sin(x)^4 + 0.25 * sin(x)^8)
	plot( xaxis  = (L"N~{\rm (degree)}", ),
    	      yaxis  = (:log, L"\Vert f - I_N f ~\Vert_{\infty}"), 
        	  legend = :outertopright, 
			  yscale = :log10, 
			  size = (500, 300))
	err = triginterperror.(f, NN)
	plot!(NN, err, lw=2, m=:o, ms=3, label = L"g_1")
	plot!(NN[3:end], exp.(- 0.9065 * NN[3:end]), lw=2, c=:black, ls=:dash, label ="predicted")
end 

# ╔═╡ 805c374e-85d8-11eb-34d2-97908f742920
let NN = (2).^(2:8), f = x -> x < π ? x * sin(x)^2 : (x-2*π) * sin(x)^2
	plot( xaxis  = (L"N~{\rm (degree)}", ),
    	      yaxis  = (:log, L"\Vert f - I_N f ~\Vert_{\infty}"), 
        	  legend = :outertopright, 
			  yscale = :log10, xscale = :log10, 
			  size = (500, 300))
	err = triginterperror.(f, NN)
	plot!(NN, err, lw=2, m=:o, ms=3, label = L"g_2")
	plot!(NN[3:end], NN[3:end].^(-2), lw=2, c=:black, ls=:dash, label ="predicted")
end 

# ╔═╡ 8e5707f0-85d2-11eb-094b-77b3288efaa8
md"""

Provide a written explanation for the results you observe. Try to justify your arguments rigorously. Alternatively use computational experiments / visualisation to explain your results.

* ``g_1(x) = 1 / (1 + 1/2 \sin^4(x))^2``, so we need to look at ``\sin(x) = \sqrt[4]{-2}``. We have 
```math
	\sqrt[4]{-2} = 2^{1/4} \times \{ e^{i \pi/4}, e^{i 3\pi/4}, e^{i 5 \pi/4}, e^{i 7\pi/4} \}
```
Computing the ``\sin^{-1}`` of these four roots we get four singularities at ``\pm 0.623674 \pm i 0.906547``. Thus ``g_1`` is analytic in ``\Omega_\alpha`` with ``\alpha = 0.906547``, and this is the rate we observe and have indicated in the figure.

* ``g_2`` appears is smooth in ``(-\pi, \pi)`` but when extended ``2\pi``-periodically. At ``x = \pi`` we have ``g_2(x) = g_2'(x) = 0`` but ``g_2'`` is no longer differentiable, but only Lipschitz, see figure below! a rigorous proof is not difficult but tedious :(. According to Jackson's second theorem we should therefore obtain the rate ``N^{-2}``.
"""

# ╔═╡ 3bdb44fa-85d7-11eb-23c4-eb563cc14783
# justification of the claim that g_2' is only Lipschitz but not C1
begin
	dg2(x) = ( x < π ? sin(x)^2 + 2 *  x     * sin(x) * cos(x)     # x < π
		             : sin(x)^2 + 2 * (x-2π) * sin(x) * cos(x)  )  # x > π
	plot(dg2, π-0.2, π+0.2, size = (300, 200), legend = :bottomright, 
			label = "", xlabel = "x", title = L"g_2' \mathrm{~is~Lipschitz}")	
end

# ╔═╡ f960efcc-85d5-11eb-2ef2-f53ee53efe26
# computing the inverse of the sin function
asin.( 2.0^0.25 * exp.( im * π/4 * [1, 3, 5, 7] ) )

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
and that this series converges absolutly (and hence uniformly! Why?). Using step 1 show that the *Fourier coefficients* are given by 
```math
	\hat{f}_k = \frac{1}{2\pi} \int_{-\pi}^\pi f(x) e^{-i k x} \, dx. 
```


3. Assume again that ``f`` is given by its Fourier series. Using step 1 again prove the Plancherel theorem,
```math

	\|f\|_{L^2} = \bigg( \int_{-\pi}^\pi |f(x)|^2 \,dx \bigg)^{1/2} 
	= \sum_{k \in \mathbb{Z}} |\hat{f}_k|^2.
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

ENTER YOUR SOLUTION HERE USING MARKDOWN AND LATEX

"""

# ╔═╡ 51470f16-85e8-11eb-119a-35dc77dd53bb
md"""
1. This is a trivial one-line calculation 
```math
\int_{-\pi}^\pi e^{i k x} e^{-i l x} 
	= 
\int_{-\pi}^\pi e^{i (k-l) x} 
   = 
\begin{cases}
    \big[ \frac{1}{i (k-l)} e^{i (k-l) x} \big]_{-\pi}^\pi = 0, & k \neq l, \\ 
    2 \pi, & k = l.
\end{cases}
```

2. Another basic calculation (since we assumed everything converges sufficiently strongly that limits can be interchanged) 
```math
	\frac{1}{2\pi} \int_{-\pi}^\pi  \sum_l \hat{f}_l e^{i l x} e^{-i k x} \,dx 
	= 
	\sum_l \hat{f}_l \frac{1}{2\pi} \int_{-\pi}^\pi e^{i l x} e^{-i k x} \,dx
	= \sum_l \hat{f}_l \delta_{kl} = \hat{f}_k.
```

3. Yet another elementary calculation, use ``|z|^2 = z z^*``, then 
```math
\begin{aligned}
	\int_{-\pi}^\pi |f|^2 \,dx
	&= 
	\int_{-\pi}^\pi \bigg| \sum_k \hat{f}_k e^{i k x} \bigg| \,dx \\ 
	&= 
	\sum_{k, l} \hat{f}_k \hat{f}_l^* \int_{-\pi}^\pi e^{i k x} e^{-i l x} \,dx \\ 
	&= 
	\sum_{k, l} \hat{f}_k \hat{f}_l^* \delta_{kl} 
	= \sum_k |\hat{f}_k|^2.
\end{aligned}
```

4. If ``f`` is given by the Fourier series, then 
```math
  f - \Pi_N f = \sum_{|k| > N} \hat{f}_k e^{i k x}.
```
If ``t_N`` is *any* other trigonometric polynomial then we get ``t_N = \Pi_N f + s_N`` for another trigonometric polynomial and we have 
and by step (3) 
```math 
	\|f - t_N\|^2_{L^2} = \sum_{k = -N}^N |\hat{s}_k|^2 + \sum_{|k| > N} |\hat{f}_k|^2.
```
We see that the error is minimized by taking ``\hat{s}_k = 0`` i.e. ``s_N = 0`` i.e. ``t_N = \Pi_N f``.

5. [BONUS] In a sentence, Jackson's theorem states that the trigonometric polynonmials are dense in ``C_{\rm per}`` and therefore every ``f`` can be written as a Fourier series. A more detailed answer might proceed as follows:  Assume that there exists ``f \in C_{\rm per}`` that is not represented by its Fourier series. This would mean that ``\|f - \Pi_N f \|_{L^2} \not\to 0``. Let ``t_N \to f`` uniformly by Jackson's theorem, then 
```math
\begin{aligned}
	\| f - \Pi_N f \|_{L^2}
	&\leq \|f - t_N \|_{L^2} + \|t_N - \Pi_N t_N \|_{L^2} + \| \Pi_N t_N - \Pi_N f \|_{L^2}
	\\
	&\leq 
	2 \|f - t_N \|_{L^2} \leq 4\pi \|f - t_N \|_\infty \to 0.
\end{aligned}
```
which is a contradiction. We used that ``t_N - \Pi_N t_N  = 0`` and that ``\Pi_N`` is a projection i.e. has operator norm 1.
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

ENTER YOUR SOLUTION HERE USING MARKDOWN AND LATEX

"""

# ╔═╡ 2295c95e-85e9-11eb-1dcc-491509a3fc6b
md""" 

### Model Solution

1. analogous as in continuous case: If ``k = l`` then the statement is trivial. If ``k \neq l`` then 
```math
	\sum_{j=0}^{2N-1} \big[e^{i (k-l) \pi/N}\big]^j
	 = \frac{[e^{i (k-l) \pi/N}]^{2N} - 1}{e^{i (k-l) \pi/N - 1}}
	 = 0, 
```
where we used ``e^{i (k-l) \pi/N}]^{2N} = e^{i 2\pi (k-l)} = 1``.

2. This is precisely the statement that the matrix ``A`` defining the trigonometric interpolant is orthogonal up to scaling, i.e., that ``A A^H = 2N I``.

3. Let ``f`` be given by its Fourier series. 
```math
	\int_{-\pi}^\pi f \,dx = 2\pi \hat{f}_0.
```
Further, 
```math
\begin{aligned}
	\frac{2\pi}{2N} \sum_{j = 0}^{2N-1} f(x_j) 
	&= \sum_k \hat{f}_k \frac{2\pi}{2N} \sum_{j = 0}^{2N-1} e^{i k \pi j/N} 
\end{aligned} 
```
Now we use that 
```math
\frac{1}{2N} \sum_{j = 0}^{2N-1} e^{i k \pi j/N} = 
	\begin{cases}
		1, & k = 0, \pm 2N, \pm 4N, \dots  \\ 
		0, & \text{otherwise}
	\end{cases}
```
to obtain 
```math
\begin{aligned}
	\frac{2\pi}{2N} \sum_{j = 0}^{2N-1} f(x_j) 
	&= 
	2 \pi \sum_{k \in 2N \mathbb{Z}} \hat{f}_k.
\end{aligned} 
```
Next, using the Paley-Wiener theorem we have that ``|\hat{f}_k|\leq M_\alpha e^{-\alpha |k|}`` and this gives 
```math
	\bigg|\frac{2\pi}{2N} \sum_{j = 0}^{2N-1} f(x_j)  - \int_{-\pi}^\pi f \,dx \bigg|
	\leq M_\alpha \sum_{k \in 2N \mathbb{Z} \setminus \{0\}}  e^{-\alpha |k|}
	\leq \frac{M_\alpha}{\alpha} e^{- 2 \alpha N}.	
```

"""

# ╔═╡ Cell order:
# ╟─43447f58-85d1-11eb-3f03-1b051b053a55
# ╟─54973c5c-85d1-11eb-10a0-5755fd06dd68
# ╠═dc1fde3c-85fd-11eb-16ec-fde6cc79e2f7
# ╠═43554bae-85d2-11eb-13d0-79e9415cd5a0
# ╟─6b9bda66-85d1-11eb-2edf-23d00c859783
# ╠═5c0a9cdc-85d2-11eb-38fe-d3d36f1b2376
# ╠═81afeb16-85d2-11eb-0edc-b724928dfd5e
# ╠═1d296dd8-85d8-11eb-3c7f-fd170297a8df
# ╠═805c374e-85d8-11eb-34d2-97908f742920
# ╟─8e5707f0-85d2-11eb-094b-77b3288efaa8
# ╠═3bdb44fa-85d7-11eb-23c4-eb563cc14783
# ╠═f960efcc-85d5-11eb-2ef2-f53ee53efe26
# ╟─b6f731c6-85d2-11eb-23c5-15457814f0e0
# ╠═d37cf0fa-85d3-11eb-1d1c-b9d1cd764e9e
# ╟─51470f16-85e8-11eb-119a-35dc77dd53bb
# ╟─de81c3b8-85d3-11eb-072c-17aa0e22db82
# ╠═68833984-85d4-11eb-3f73-ed3ee969d0b5
# ╟─2295c95e-85e9-11eb-1dcc-491509a3fc6b
