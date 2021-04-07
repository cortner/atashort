### A Pluto.jl notebook ###
# v0.12.21

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

# ╔═╡ 465e7582-95c7-11eb-0c7b-ed7dddc24b4f
begin
	using Plots, LaTeXStrings, PrettyTables, DataFrames, LinearAlgebra,
          PlutoUI, BenchmarkTools, ForwardDiff, Printf, Random, FFTW
	include("tools.jl")
	
	xgrid(N) = [ j * π / N  for j = 0:2N-1 ]
	kgrid(N) = [ 0:N; -N+1:-1 ]
	triginterp(f, N) = fft(f.(xgrid(N))) / (2*N)
	evaltrig(x, F̂) = sum( real(F̂ₖ * exp(im * x * k))
						  for (F̂ₖ, k) in zip(F̂, kgrid(length(F̂) ÷ 2)) )
	function trigerr(f, N; xerr = range(0, 2π, length=13*N)) 
		F̂ = triginterp(f, N) 
		return norm( evaltrig.(xerr, Ref(F̂)) - f.(xerr), Inf )
	end
end;

# ╔═╡ 58d79c34-95c7-11eb-0679-eb9741761c10
md"""
## §4 Miscallaneous 

In this section of the mini-course we will cover a small number of topics, without going too deeply into any of them, simply to get some exposure to some important and fun ideas:
* Chebyshev polynomials and Chebyshev points 
* Rational Approximation 
* Max-norm approximation with iteratively reweighted least squares
"""

# ╔═╡ 9c169770-95c7-11eb-125a-4f174da56d36
md"""
## §4.1 Approximation with Algebraic Polynomials

The witch of Agnesi: 
```math
	f(x) = \frac{1}{1+ 25 x^2}, \qquad x \in [-1, 1]. 
```
"""

# ╔═╡ 05ec7c1e-95c8-11eb-176a-3372a765d4d7
begin
	agnesi(x) = 1 / (1 + 25 * x^2)
	plot(agnesi, -1, 1, lw=3, label = "", 
		 size = (350, 200), title = "Witch of Agnesi", 
		 xlabel = L"x", ylabel = L"f(x)")
end

# ╔═╡ 311dd26e-95ca-11eb-2ff5-03a53a662a62
md"""
We can rescale ``f`` and repeat it periodically and then use trigonometric polynomials to approximate it. Because the periodic extension is only ``C^{0,1}``, i.e., Lipschitz, but no more we only get a rate of ``N^{-1}``. 
"""

# ╔═╡ 4428c460-95c8-11eb-16fc-6327acc80bb6
let f = x -> agnesi((x - π)/π), NN = (2).^(2:10)
	plot(NN, trigerr.(Ref(f), NN), lw=2, m=:o, ms=4,
		 size = (350, 250), xscale = :log10, yscale = :log10, 
		 label = "error", title = "Approximation of Agnesi", 
		 xlabel = L"N", ylabel = L"\Vert f- I_N f \Vert_\infty")
	plot!(NN[5:end], 3e-2*NN[5:end].^(-1), lw=2, ls=:dash, c=:black, 
		  label = L"\sim N^{-1}" )
end

# ╔═╡ 3533717c-9630-11eb-3fc6-3f4b29eaa63a
md"""
And no surprise - the periodic repetition of ``f`` is only Lipschitz:
"""

# ╔═╡ 467c7f76-9630-11eb-1cdb-6dedae3c459f
let f = x -> agnesi((mod(x+1, 2)-1)/2)
	plot(f, -1, 4, lw=2, label = "", 
		 size = (400, 200), xlabel = L"x", ylabel = L"f(x)")
end

# ╔═╡ 6cbcac96-95ca-11eb-05f9-c7a781b38061
md"""
But it is a little ridiculous that we get such a poor rate: After all ``f`` is an analytic function on it domain of definition ``[-1, 1]``. Can we replicate the excellent approximation properties that trigonometric polynomials have for periodic analytic functions?

Our first idea is to use algebraic polynomials 
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

# ╔═╡ 77dcea8e-95ca-11eb-2cf1-ad342f0f6d7d
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

# ╔═╡ 27f7e2c8-95cb-11eb-28ca-dfe90be89670
md"""
## The Chebyshev Trick

The idea is to lift ``f`` to the complex unit circle: 

```math 
	g(\theta) := g(\cos\theta)
```
"""

# ╔═╡ cc8fe0f4-95cd-11eb-04ec-d3fc8cced91b
let
	tt = range(0, 2π, length=300)
	P1 = plot(cos.(tt), sin.(tt), lw=3, label = "",  
		 size = (250, 250), xlabel = L"x = {\rm Re} z", ylabel = L"{\rm Im} z")
	plot!([-1,1], [0,0], lw=3, label = "", legend = :bottomright)
	for x in -1:0.2:1
		plot!([x,x], [sqrt(1-x^2), -sqrt(1-x^2)], c=2, lw=1, label = "")

	end
	P1
end

# ╔═╡ dfc0866c-95cb-11eb-1869-7bd67468c233
let
	tt = range(0, 2π, length=300)
	xx = range(-1, 1, length=200)
	P2 = plot(cos.(tt), sin.(tt), 0*tt, c = 1, lw = 3, label = "")
	# for x in -1:0.1:1
	# 	f = agnesi(x)
	# 	plot!([x,x], [sqrt(1-x^2), -sqrt(1-x^2)], [f, f], 
	# 		  c=:grey, lw=1, label = "")
	# end 
	plot!([-1, 1], [0, 0], [0, 0], c = 2, lw = 3, label = "")
	plot!(xx, 0*xx, agnesi.(xx), c = 2, lw = 2, label = "Agnesi")
	plot!(cos.(tt), sin.(tt), agnesi.(cos.(tt)), c=3, lw=3, label= "Lifted", 
		  size = (400, 300))
end 

# ╔═╡ f3cd1326-95cd-11eb-192a-efc3371d17b6
md"""
For our Agnesi example we obtain 
```math
g(\theta) = \frac{1}{1+ 25 \cos^2(\theta)}
```
which we already know to be analytic. This is a general principle. The regularity of ``f`` transforms into the same regularity for ``g`` but we gain periodicity. We will state some general results below. For now, let us continue to investigate:

Since ``g`` is periodic analytic we can apply a trigonometric approximation: 
```math
	g(\theta) \approx \sum_{k = -N}^N \hat{g}_k e^{i k \theta} 
```
and we know that we obtain an exponential rate of convergence. 
"""

# ╔═╡ 5181b8f4-95cf-11eb-30fe-cd93f4ed0012
md"""
### Chebyshev Polynomials

We could simply use this to construct approximations for ``f(x) = g(\cos^{-1}(x))``, but it is very instructive to transform this approximation back to ``x`` coordinates. To that end we first note that 
```math
	g(-\theta) = f(\cos(-\theta)) = f(\cos\theta) = g(\theta)
```
and moreover, ``g(\theta) \in \mathbb{R}``. These two facts allow us to show that ``\hat{g}_{-k} = \hat{g}_k`` and moreover that $\hat{g}_k \in \mathbb{R}$. Thus, we obtain ,
```math
	f(x) = g(\theta) = \hat{g}_0 + \sum_{k = 1}^N \hat{g}_k (e^{i k \theta} + e^{-i k \theta})
	= \hat{g}_0 + \sum_{k = 1}^N \hat{g}_k \cos(k \theta).
```
Now comes maybe the most striking observation: if we define basis functions ``T_k(x)`` via the identify 
```math
	T_k( \cos \theta ) = \cos( k \theta )
```
so that we can write 
```math
	f(x) = \sum_{k = 0}^N \tilde{f}_k T_k(x)
```
where $\tilde{f}_0 = \hat{g}_0$ and $\tilde{f}_k = 2 \hat{g}_k$ of $k \geq 1$, then we have the following result: 

**Lemma:** The function ``T_k(x)`` is a polynomial of degree ``k``. In particular, they form a basis of the space of polynomials, called the *Chebyshev Basis*. An equivalent definition is the 3-term recursion 
```math
\begin{aligned}
	T_0(x) &= 1,   \\
	T_1(x) &= x,   \\ 
	T_k(x) &= 2 x T_{k-1}(x) + T_{k-2}(x) \quad \text{for } k = 2, 3, \dots.
\end{aligned}
```
**Proof:** see [LN, Lemma 4.2]
"""


# ╔═╡ 7661921a-962a-11eb-04c0-5b7e486aea05
md"""
### Chebyshev Interpolant 

Next, we transform the trigonometric interpolant ``I_N g(\theta)`` to ``x`` coordinates. We already know now that it will be a real algebraic polynomial and we can derive a more derict way to define it: namely, the interpolation nodes are given by 
```math
	x_j = \cos( \theta_j ) = \cos( \pi j / N ), \qquad j = 0, \dots, 2N-1
```
but because of the reflection symmetry of ``\cos`` they are repeated, i.e., ``x_j = x_{-j}`` and in fact we only need to keep 
```math 
	x_j = \cos( \pi j / N ), \qquad j = 0, \dots, N.
```
which are called the *Chebyshev nodes* or *Chebyshev points*. Thus, trigonometric interpolation of ``g(\theta)`` corresponds to polynomial interpolation of ``f(x)`` at those points: 
```math
	I_N f(x_j) = f(x_j), \qquad j = 0, \dots, N.
```
"""

# ╔═╡ 4ca65bec-962b-11eb-3aa2-d5500fd24677
md"""
If we write ``I_N f(x) = p_N(x) = \sum_{n = 0}^N c_n T_n(x)`` then we obtain an ``(N+1) \times (N+1)`` linear system for the coefficients ``\boldsymbol{c}=  (c_n)_{n = 0}^N``, 
```math
	\sum_{n = 0}^N c_n T_n(x_j) = f(x_j), \qquad j = 0, \dots, N
```
The resulting polynomial ``I_N f`` is called the *Chebyshev interpolant*. A naive implementation can be performed as follows:
"""

# ╔═╡ 69dfa184-962b-11eb-0ed2-c17aa6c15a45
begin 
	# NAIVE IMPLEMENTATION OF CHEBYSHEV INTERPOLATION
	# Two alternative implemenations are given in `tools.jl`:
	#    - fast Chebyshev transform (basically FFT); cf [LN, §4.3]
	#    - barycentric interpolation; cf [LN, §4.4]

	"""
	reverse the nodes so they go from -1 to 1, i.e. left to right
	"""
	chebnodes(N) = [ cos( π * n / N ) for n = N:-1:0 ]
	
	function chebinterp_naive(f, N)	
		X = chebnodes(N) 
		F = f.(X) 
		A = zeros(N+1, N+1)
		T0 = ones(N+1) 
		T1 = X 
		A[:, 1] .= T0 
		A[:, 2] .= T1 
		for n = 2:N
			T0, T1 = T1, 2 * X .* T1 - T0
			A[:, n+1] .= T1 
		end
		return A \ F
	end 
	
	function chebeval(x, F̃) 
		T0 = one(x); T1 = x 
		p = F̃[1] * T0 + F̃[2] * T1 
		for n = 3:length(F̃)
			T0, T1 = T1, 2*x*T1 - T0 
			p += F̃[n] * T1 
		end 
		return p 
	end 
end

# ╔═╡ 3ef9e58c-962c-11eb-0172-a700d2f7e72c
let f = agnesi, NN = 6:6:60
	xerr = range(-1, 1, length=2013)
	err(N) = norm(f.(xerr) - chebeval.(xerr, Ref(chebinterp_naive(f, N))), Inf)
	plot(NN, err.(NN), lw=2, m=:o, ms=4, label = "error", 
		 size = (300, 250), xlabel = L"N", ylabel = L"\Vert f - I_N f \Vert_\infty", 
		 yscale = :log10,
		 title = "Chebyshev Interpolant")
	plot!(NN[6:end], 2*exp.( - 0.2 * NN[6:end]), lw=2, ls=:dash, c=:black, 
		  label = L"\exp( - N/5 )")
end

# ╔═╡ 19509e38-962d-11eb-1f5c-811188a57261
md"""
See [LN, Sec. 4.1, 4.2] and whiteboard summary to understand this rate fully. Concepts to understand include 
* Chebyshev series: essentially the Fourier series for ``g(\theta)``
* Chebyshev coefficients: essentially the Fourier coefficients of ``g(\theta)``
* Jukousky map: the transformation between ``[-1,1]`` and the unit circle exteneded into the complex plane
* Bernstein ellipse: the tranformation of the strip ``\Omega_\alpha`` for ``g``

For the sake of completeness, we briefly state the two main approximation theorems which can be obtained fairly easily (though it would take some time) from the ideas we developed for far:

**Theorem [Jackson for Algebraic Polynomials]** Let ``f \in C^r([-1,1])`` and suppose that ``f^{(r)}`` has modulus of continuity ``\omega``, then 
```math
	\inf_{p_N \in \mathcal{P}_N} \| f - p_N \|_\infty \lesssim N^{-r} \omega(N^{-1}).
```

**Theorem:** Let ``f \in C([-1,1])`` also be analytic and bounded in the interior of the Bernstein ellipse 
```math
	E_\rho := \bigg\{ z = x + i y \in \mathbb{C} : 
  				\bigg(\frac{x}{\frac12 (\rho+\rho^{-1})}\bigg)^2
				+ \bigg(\frac{y}{\frac12 (\rho-\rho^{-1})}\bigg)^2 \leq 1 \bigg\}
```
where ``\rho > 1``, then 
```math
	\inf_{p_N \in \mathcal{P}_N} \| f- p_N \|_\infty \lesssim \rho^{-N}.
```

"""

# ╔═╡ 16591e8e-9631-11eb-1157-992452ed6514
md"""
With these results in hand we can perform a few more tests. We have already shown exponential convergence for an analytic target function; now we experiment with $$C^{j,\alpha}$$ smoothness, 
```math
\begin{aligned}
	f_1(x) &= |\cos(\pi x)|^{t} 
\end{aligned}
```
where ``t`` is given by:  $(@bind _t1 Slider(0.5:0.01:4.0, show_value=true))
"""

# ╔═╡ 7b835804-9694-11eb-0692-8dab0a07203e
let t = _t1, f = x -> abs(cos(π * x))^t, NN = (2).^(2:10)
	xerr = range(-1, 1, length=2013)
	err(N) = norm(f.(xerr) - chebeval.(xerr, Ref(chebinterp_naive(f, N))), Inf)
	plot(NN, err.(NN), lw=2, m=:o, ms=4, label = "error", 
		 size = (300, 250), xlabel = L"N", ylabel = L"\Vert f - I_N f \Vert_\infty", 
		 yscale = :log10, xscale = :log10, 
		 title = L"f(x) = |\cos(\pi x)|^{%$t}")
	plot!(NN[4:end], Float64.(NN[4:end]).^(-t), lw=2, ls=:dash, c=:black, 
		  label = L"N^{-%$t}")	
end

# ╔═╡ c1e69df8-9694-11eb-2dde-673888b5f7e2
md"""
## §4.2 An Example from Computational Chemistry

According to the Fermi-Dirac model, the distribution of particles over energy states in systems consisting of many identical particles that obey the Pauli exclusion principle is given by 
```math
	f_\beta(E) = \frac{1}{1 + \exp(\beta (E-\mu))},
```
where ``\mu`` is a chemical potential and ``\beta = \frac{1}{k_B T}`` the inverse temperature. 

Choose a ``\beta``: $(@bind _beta0 Slider(5:100, show_value=true))
"""

# ╔═╡ 5522bc18-9699-11eb-2d67-759be6fc8d62
plot(x -> 1 / (1 + exp(_beta0 * x)), -1, 1, lw=2, label = "", 
	title = L"\beta = %$_beta0", size = (300, 200), 
	xlabel = L"E", ylabel = L"f_\beta(E)")

# ╔═╡ 4fe01af2-9699-11eb-1fb7-9b43a947b51a
md"""
We will return to the chemistry context momentarily but first simply explore the approximation of the Fermi-Dirac function by polynomials. To this end we set ``\mu = 0`` and assume ``E \in [-1, 1]``.

To determine the rate of convergence, we have to find the largest possible Bernstein ellipse. The singularlities of ``f`` are given by 
```math
	i \pi / \beta (1 + 2n), \qquad n \in \mathbb{Z}.
```
This touches the semi-minor axis provided that 
```math
	\frac12 (\rho - 1/\rho) = \pi / \beta
	\qquad \Leftrightarrow \qquad 
	\rho = \sqrt{(\pi/\beta)^2 + 1} + \pi/\beta.
```

Choose a ``\beta``: $(@bind _beta1 Slider(5:100, show_value=true))
"""

# ╔═╡ b3bde6c8-9697-11eb-3ca2-1bacec3f2ad1
let β = _beta1, f = x -> 1 / (1 + exp(β * x)), NN = 6:6:110
	ρ = sqrt( (π/β)^2 + 1 ) + π/β
	xerr = range(-1, 1, length=2013)
	err(N) = norm(f.(xerr) - chebeval.(xerr, Ref(chebinterp_naive(f, N))), Inf)
	plot(NN, err.(NN), lw=2, m=:o, ms=4, label = "error", 
		 size = (300, 250), xlabel = L"N", ylabel = L"\Vert f - I_N f \Vert_\infty", 
		 yscale = :log10, 
		 title = L"\beta = %$(β)")
	plot!(NN[4:end], ρ.^(-NN[4:end]), lw=2, ls=:dash, c=:black, 
		  label = L"\rho^{-N}")	
end 

# ╔═╡ 0c74de06-9699-11eb-1234-eb64590644d7
md"""
Although the rate is always exponential it deteriorates severely as ``\beta \to \infty``. Unfortunately these large values of ``\beta`` are imporant and realistic for applications. We will return to this momentarily.
"""

# ╔═╡ 330aec36-9699-11eb-24de-794562156ef4
md"""
### Approximation of a Matrix Function

Approximating ``f_\beta`` by a polynomial is in itself not useful. Since we have very accurate and performant means to evaluate ``e^x`` this is simply not necessary. But there is a different reason why a polynomial approximation is useful. 

Let ``H \in \mathbb{R}^{P \times P}`` be a hamiltonian describing the interaction of ``P`` fermions (e.g. electrons). Many properties of this quantum mechanical system can be extracted from the *density matrix*, 
```math
	\Gamma := f_\beta(H).
```
A simple way to define this matrix function is to diagonalize ``H = Q \Lambda Q^*`` and then evaluate 
```math
	\Gamma = Q f_\beta(\Lambda) Q^* 
		= Q {\rm diag} \big( f_\beta(\lambda_1), \dots, f_\beta(\lambda_P) \big) Q^*
```
This requires diagonalisation of ``H`` an operation that costs ``O(P^3)`` operations. 

**Proposition:** If ``\|f_\beta - p_N\|_\infty \leq \epsilon`` on ``\sigma(H)`` then 
```math
	\| f_\beta(H) - p_N(H) \|_{\rm op} \leq \sup_{E \in \sigma(H)} \big| f_\beta(E) - p_N(E) \big|
```
"""

# ╔═╡ 53439934-975f-11eb-04f9-43dc19404d6b
md"""
On the other hand we can evaluate ``p_N(H)`` with ``N`` matrix multiplications. If ``H`` is dense then we don't gain anything, but if ``H`` is sparse e.g. that it has only ``O(P)`` entries, then one can show that 
```math
	{\rm cost} \approx N^2 P \approx \beta^2 \log^2(\epsilon) P.
```
where ``\epsilon`` is the desired accuracy. For small or moderate ``\beta`` this is an excellent result allowing us to trade accuracy against computational cost, when ``P`` is very large.

However when ``\beta`` is large, then the poor convergence rate makes this a less attractive algorithm. E.g., if ``\sigma(H) \subset [-1,1]`` and we require an accuracy of ``\epsilon = 10^{-6}`` (this is a typical target) then we require that 
```math
 	\beta \leq \frac{P}{13}
```
for the polynomial algorithm to be more efficient that the diagonalisation algorithm, which is quite restrictive. Note that ``\beta \in [30, 300]`` is a physically realistic range.
"""

# ╔═╡ e3466610-9760-11eb-1562-61a6a1bf76bf
let 
	# TODO : implement this idea	
end

# ╔═╡ 27a272c8-9695-11eb-3a5f-014392471e7a
md"""
### Rational Approximation of the Fermi-Dirac Function 
"""

# ╔═╡ 3fb12378-9695-11eb-0766-9faf92928ad2
md"""
## §4.3 Best Approximation via IRLSQ
"""

# ╔═╡ Cell order:
# ╟─465e7582-95c7-11eb-0c7b-ed7dddc24b4f
# ╟─58d79c34-95c7-11eb-0679-eb9741761c10
# ╟─9c169770-95c7-11eb-125a-4f174da56d36
# ╟─05ec7c1e-95c8-11eb-176a-3372a765d4d7
# ╟─311dd26e-95ca-11eb-2ff5-03a53a662a62
# ╟─4428c460-95c8-11eb-16fc-6327acc80bb6
# ╟─3533717c-9630-11eb-3fc6-3f4b29eaa63a
# ╟─467c7f76-9630-11eb-1cdb-6dedae3c459f
# ╟─6cbcac96-95ca-11eb-05f9-c7a781b38061
# ╟─77dcea8e-95ca-11eb-2cf1-ad342f0f6d7d
# ╟─27f7e2c8-95cb-11eb-28ca-dfe90be89670
# ╟─cc8fe0f4-95cd-11eb-04ec-d3fc8cced91b
# ╟─dfc0866c-95cb-11eb-1869-7bd67468c233
# ╟─f3cd1326-95cd-11eb-192a-efc3371d17b6
# ╟─5181b8f4-95cf-11eb-30fe-cd93f4ed0012
# ╟─7661921a-962a-11eb-04c0-5b7e486aea05
# ╟─4ca65bec-962b-11eb-3aa2-d5500fd24677
# ╠═69dfa184-962b-11eb-0ed2-c17aa6c15a45
# ╟─3ef9e58c-962c-11eb-0172-a700d2f7e72c
# ╟─19509e38-962d-11eb-1f5c-811188a57261
# ╟─16591e8e-9631-11eb-1157-992452ed6514
# ╟─7b835804-9694-11eb-0692-8dab0a07203e
# ╟─c1e69df8-9694-11eb-2dde-673888b5f7e2
# ╟─5522bc18-9699-11eb-2d67-759be6fc8d62
# ╟─4fe01af2-9699-11eb-1fb7-9b43a947b51a
# ╟─b3bde6c8-9697-11eb-3ca2-1bacec3f2ad1
# ╟─0c74de06-9699-11eb-1234-eb64590644d7
# ╟─330aec36-9699-11eb-24de-794562156ef4
# ╟─53439934-975f-11eb-04f9-43dc19404d6b
# ╠═e3466610-9760-11eb-1562-61a6a1bf76bf
# ╠═27a272c8-9695-11eb-3a5f-014392471e7a
# ╠═3fb12378-9695-11eb-0766-9faf92928ad2
