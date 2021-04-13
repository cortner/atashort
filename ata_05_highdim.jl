### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 3f1cfd12-7b86-11eb-1371-c5795b87ef5b
begin
	using Plots, LaTeXStrings, PrettyTables, DataFrames, LinearAlgebra,
		  PlutoUI, BenchmarkTools, ForwardDiff, Printf, FFTW
	include("tools.jl")
end;

# ╔═╡ 76e9a7f6-86a6-11eb-2741-6b8759be971b
md"""
## §2 Approximation in Moderate and High Dimension

After our interlude on polynomial approximation, rational approximation, etc, we now return to trigonometric approximation by in dimension ``d > 1``. That is, we will consider the approximation of functions 
```math
	f \in C_{\rm per}(\mathbb{R}^d)
	:= \big\{ g \in C(\mathbb{R}^d) \,|\, 
			  g(x + 2\pi \zeta) = g(x) \text{ for all } \zeta \in \mathbb{Z}^d \big\}.
```


Our three goals for this lecture are

* 2.1 Approximation in moderate dimension, ``d = 2, 3``
* 2.2 Spectral methods for solving PDEs in two and three dimensions
* 2.3 Approximation in "high dimension"
"""


# ╔═╡ 551a3a58-9b4a-11eb-2596-6dcc2edd334b
md"""
**TODO:** Review of Fourier series, trigonometric interpolation.
"""

# ╔═╡ a7e7ece4-9b48-11eb-26a8-213556563a81
md"""
## § 2.1 Approximation in Moderate Dimension

Our first task is to figure out how to construct approximations from trigonometric polynomials which are inherently one-dimensional objects.

### 2D Case

We will first develop all ideas in 2D, and then quickly generalize them to ``d`` dimensions.

Let ``f \in C_{\rm per}(\mathbb{R}^2)``, i.e., ``f(x_1, x_2)`` is continuous and ``2\pi`` periodic in each of the two coordinate directions. In particular, if we "freeze" the coordinate ``x_1 = \hat{x}_1`` then we obtain a one-dimensional function 
```math
	x_2 \mapsto f(\hat{x}_1, x_2) \in C_{\rm per}(\mathbb{R}).
```
which we can approximate by a trigonometric interpolant, 
```math
	f(\hat{x}_1, x_2) \approx I_N^{(2)} f(\hat{x}_1, x_2) 
	= \sum_{k_2 = -N}^N c_k(\hat{x}_1) e^{i k_2 x_2},
```
where the superscript ``(2)`` in ``I_N^{(2)}`` indicates that the interpolation is performed with respect to the ``x_2`` coordinate. 
"""

# ╔═╡ 102f6eca-9b4a-11eb-23b8-210bd9100faa
md"""
Since trigonometric interpolation is a continuous operation it follows that ``c_k(x_1)`` is again a continuous and ``2\pi``-periodic function of ``x_1``. This takes a bit of work to prove, and we won't really need it later so let's not worry too much about it. But if we accept this as fact, then we can now approximate each ``c_k(x_1)`` again by its trigonometric interpolant, 
```math
	c_{k_2}(x_2)  \approx I_N^{(1)} c_{k_2}(x_1) 
			= \sum_{k_1 = -N}^N c_{k_1 k_2} e^{i k_1 x_1}.
```
Inserting this identity above we deduce 
```math
	f(x_1, x_2) \approx I_N^{(1)} I_N^{(2)} f(x_1, x_2) = 
			\sum_{k_1, k_2 = -N}^N c_{k_1 k_2} e^{i (k_1 x_1 + k_2 x_2)}.
```
Indeed we will momentarily see that this in fact has very similar (excellent) approximation properties as we have seen in the univariate setting. 

We have constructed an approximation to ``f(x_1, x_2)`` in terms of a multi-variate basis 
```math
	e^{i (k_1 x_1 + k_2 x_2)} = e^{i k_1 x_1} e^{i k_2 x_2}.
```
Next, let us write down the interpolation conditions for ``I^{(1)} I^{(2)} f``: 
"""

# ╔═╡ 974d10ae-9b9c-11eb-324f-21c03e345ca4
md"""
* First, ``I^{(2)}_N f`` imposes the interpolation condition 
```math
	I^{(2)}_N f(x_1, \xi_j) = f(x_1, \xi_j), \qquad j = 0, \dots, 2N-1, \quad \forall x_1 \in [0, 2\pi).
```
where ``\xi_j = \pi/N`` are the university equispaced interpolation nodes, 
* Next, we have applied ``I^{(1)}_N`` giving ``I^{(1)}_N I^{(2)}_N f`` which restricts this identity only to the interpolation nodes, i.e., 
```math
	I^{(1)}_N I^{(2)}_Nf(\xi_{j_1}, \xi_{j_2}) = f(\xi_{j_1}, \xi_{j_2}) = f(\xi_i, \xi_j), 
	\qquad j_1, j_2 = 0, \dots, 2N-1.
```
That is, we are imposing identity between target ``f`` and interpolant ``I^{(1)}_N I^{(2)}_Nf`` on the tensor product grid 
```math
	\{ \boldsymbol{\xi}_{j_1 j_2}  = (\xi_{j_1}, \xi_{j_2}) \,|\, 
	   j_1, j_2 = 0, \dots, 2N-1 \}.
```
"""

# ╔═╡ 6e96c814-9b9a-11eb-1200-bf99536f9369
begin
	xgrid(N) = range(0, 2π - π/N, length=2N)
	xygrid(Nx, Ny=Nx) = (xgrid(Nx) * ones(2Ny)'), (ones(2Nx) * xgrid(Ny)')
end;

# ╔═╡ 4741a420-9b9a-11eb-380e-07fc50b805c9
let N = 8
	X = xgrid(8)
	P1 = plot(X, 0*X .+ π, lw=0, ms=3, m=:o, label = "x grid",
			  title = "Univariate grids", xlims = [-0.1, 2π], ylims = [-0.1, 2π])
	plot!(P1, 0*X .+ π, X, lw=0, ms=3, m=:o, label = "y grid")
	X, Y = xygrid(N)
	P2 = plot(X[:], Y[:], lw=0, m=:o, ms=3, label = "", size = (300,300), 
			  title = "Tensor grid (x,y)", xlims = [-0.1, 2π], ylims = [-0.1, 2π])
	plot(P1, P2, size = (400, 200))
end 

# ╔═╡ 95edae16-9b9d-11eb-2698-8d52b0f18a57
md"""
In particular we have seen that the order of applying the two interpolation operators does not matter, and we can now simply write ``I_N f = I_N^{(1)} I_N^{(2)} f``. 

Our next question is how to determine the coefficients of the 2D trigonometric interpolant: 
```math
	I_N f(x_1, x_2) = \sum_{k_1, k_2 = -N}^N \hat{F}_{k_1 k_2} e^{i (k_1 x_1 + k_2 x_2)}.
```
We can of course write down the interpolation conditions again and solve the linear system
```math
	A \hat{F} = F, \qquad \text{where} \qquad 
	A_{j_1 j_2, k_1 k_2} = \exp\big(i (k_1 \xi_{j_1} + k_2 \xi_{j_2})\big)
```
In 1D we used the fact that the corresponding operator could be inverted using the FFT. This still remains true:
"""

# ╔═╡ 2ff639c8-9b9f-11eb-000e-37bbbea50dc5
begin
	"univariate k-grid"
	kgrid(N) = [ 0:N; -N+1:-1 ]
		
	"""
	Evaluation of a two-dimensional trigonometric polynomial
	Note that we only need the univariate k-grid since we just evaluate 
	it in a double-loop!
	"""
	function evaltrig(x, y, F̂::Matrix) 
		Nx, Ny = size(F̂)
		Nx = Nx ÷ 2; Ny = Ny ÷ 2
		return sum( real(exp(im * (x * kx + y * ky)) * F̂[jx, jy])
			        for (jx, kx) in enumerate(kgrid(Nx)), 
		                (jy, ky) in enumerate(kgrid(Ny)) )
	end
	
	"""
	2D trigonometric interpolant via FFT 
	"""
	triginterp2d(f::Function, Nx, Ny=Nx) = 
			fft( f.(xygrid(Nx, Ny)...) ) / (4 * Nx * Ny)
end;

# ╔═╡ bedaf224-9b9e-11eb-0a7d-ad170bfd73a7
md"""
Maybe this is a good moment to check - numerically for now - whether the excellent approximation properties that we enjoyed in 1D are retained! We start with a 2D version of the periodic witch of Agnesi.
"""

# ╔═╡ f7454b4a-9bc2-11eb-2ab9-dfc016fd57de
md"""
We will later prove a theoretical result that correctly predicts this rate.

Our first reaction is that this is wonderful, we obtain the same convergence rate as in 1D. But we have to be careful! While the rate is the same in terms of the degree ``N`` the **cost** associated with evaluating ``f`` now scales like ``N^2``. So in terms of the **cost**, the convergence rate we observe here is 
```math
	\epsilon = {\rm error} \sim \exp\Big( - \alpha \sqrt{{\rm cost}} \Big)
```
or equivalently, 
```math
	{\rm cost}  \sim \alpha^{-1} |\log \epsilon|^2.
```
We call this *polylogarithmic cost*. This can become prohibitive in high dimension; we will return to this in the third part of the lecture.
"""

# ╔═╡ 1506eaf4-9b9a-11eb-170d-95834314fb84
md"""
### General Case

Let's re-examine the approximation we constructed, 
```math
	f(x_1, x_2) \approx \sum_{k_1, k_2} c_{k_1 k_2} e^{i k_1 x_1} e^{i k_2 x_2}.
```
These 2-variate basis functions
```math
e^{i k_1 x_1} e^{i k_2 x_2}
```
are *tensor products* of the univariate basis ``e^{i k x}``. One normally writes
```math
	(a \otimes b)(x_1, x_2) = a(x_1) b(x_2).
```
If we define ``\varphi_k(x) := e^{i k x}`` then 
```math
	(\varphi_{k_1} \otimes \varphi_{k_2})(x_1, x_2) = \varphi_{k_1}(x_1) \varphi_{k_2}(x_2)
	= e^{i k_1 x_1} e^{i k_2 x_2}
```
"""

# ╔═╡ 1eb00bb6-9bc8-11eb-3af9-c33ef5b6ab15
md"""
Now suppose that we are in ``d`` dimensions, i.e., ``f \in C_{\rm per}(\mathbb{R}^d)``, then we can approxiate it by the ``d``-dimensional tensor products: let 
```math
\begin{aligned}
	\varphi_{\bf k}({\bf x}) &:= \Big(\otimes_{t = 1}^d \varphi_{k_t} \Big)({\bf x}), \qquad \text{or, equivalently,} \\ 
	\varphi_{k_1 \cdots k_d}(x_1, \dots, x_d)
	&= 
	\prod_{t = 1}^d \varphi_{k_t}(x_t)
	= \prod_{t = 1}^d e^{i k_t x_t}
	= \exp\big(i {\bf k} \cdot {\bf x}).
\end{aligned}
```
"""

# ╔═╡ 28aa73fe-9bc8-11eb-3266-f725f73a3159
md"""
The interpolation condition generalises similarly: 
```math
   \sum_{{\bf k} \in \{-N,\dots,N\}^d} \hat{F}_{\bf k} e^{i {\bf k} \cdot \boldsymbol{\xi} } = f(\boldsymbol{\xi}) 
	\qquad \forall \boldsymbol{\xi} = (\xi_{j_1}, \dots, \xi_{j_d}), \quad 
	j_t = 0, \dots, 2N-1.
```
And the nodal interpolant can be evaluated using the multi-dimensional Fast Fourier transform. Here we implement it just for 3D since we won't need it beyond three dimensions for now.
"""

# ╔═╡ 9b31019a-9bc8-11eb-0372-43ea0c0d8fc3
begin
	
	function xyzgrid(Nx, Ny=Nx, Nz=Nx) 
		X = [ x for x in xgrid(Nx), y = 1:2Ny, z = 1:2Nz ]
		Y = [ y for x in 1:2Nx, y in xgrid(Ny), z = 1:2Nz ]
		Z = [ z for x in 1:2Nx, y = 1:2Nz, z in xgrid(Nz) ]
		return X, Y, Z
	end

	"""
	Evaluation of a three-dimensional trigonometric polynomial
	Note that we only need the univariate k-grid since we just evaluate 
	it in a double-loop!
	"""
	function evaltrig(x, y, z, F̂::Array{T,3}) where {T} 
		Nx, Ny, Nz = size(F̂)
		Nx = Nx ÷ 2; Ny = Ny ÷ 2; Nz = Nz ÷ 2
		return sum( real(exp(im * (x * kx + y * ky + z * kz)) * F̂[jx, jy, jz])
			        for (jx, kx) in enumerate(kgrid(Nx)), 
		                (jy, ky) in enumerate(kgrid(Ny)),
						(jz, kz) in enumerate(kgrid(Nz)) )
	end
	
	"""
	2D trigonometric interpolant via FFT 
	"""
	triginterp3d(f::Function, Nx, Ny=Nx, Nz=Nx) = 
			fft( f.(xyzgrid(Nx, Ny, Nz)...) ) / (8 * Nx * Ny * Nz)

	
end

# ╔═╡ bd5807fc-9b9e-11eb-1434-87af91d2d296
let N = 8, f = (x, y) -> exp(-cos(x)^2-cos(y)^2-cos(x)*cos(y))
	# evaluate the function at the interpolation points 
	X, Y = xygrid(N)
	F = f.(X, Y)
	# transform to trigonometric polynomial coefficients
	F̂ = fft(F) / (2N)^2
	# evaluate the trigonometric polynomial at the interpolation nodes
	Feval = evaltrig.(X, Y, Ref(F̂))
	# check that it satisfies the interpolation condition
	F ≈ Feval
	# and while we are at it, we can also check that this is the same 
	# as inverting the FFT
	Finv = real.(ifft(F̂)) * (2N)^2
	F ≈ Feval ≈ Finv
end

# ╔═╡ bdcd0340-9b9e-11eb-0ee6-e7e0993e8fe8
let f = (x, y) -> 1 / (1 + 10 * cos(x)^2 + 10 * cos(y)^2), NN = 4:2:20
	Xerr, Yerr = xygrid(205)
	err(N) = norm( f.(Xerr, Yerr) - evaltrig.(Xerr, Yerr, Ref(triginterp2d(f, N))), Inf )
	plot(NN, err.(NN), lw=3, ms=4, m=:o, label = "",
		 xlabel = L"N", ylabel = L"\Vert f - I_N f \Vert_\infty", 
		 yscale = :log10, size = (400, 250) )		
	plot!(NN[5:end], 2* exp.( - NN[5:end] / sqrt(10) ), lw = 2, ls=:dash, 
		  label = L"\sim \exp( - N / \sqrt{10} )")
end

# ╔═╡ c9b0923c-9bce-11eb-3861-4d201b1765a8
md"""
Let us test this implementation on a three-dimensional periodic witch of agnesi: 
```math
	f(x, y, z) = \frac{1}{1 + c (\cos^2 x + \cos^2 y + \cos^2 z)}
```
"""

# ╔═╡ abc29124-9bc8-11eb-268d-c5115c341b58
let f = (x,y,z) -> 1 / (1 + 10 * (cos(x)^2+cos(y)^2+cos(z)^2)), N = 4
	# evaluate the function at the interpolation points 
	X, Y, Z = xyzgrid(N)
	F = f.(X, Y, Z)
	# transform to trigonometric polynomial coefficients
	F̂ = fft(F) / (2N)^3
	# evaluate the trigonometric polynomial at the interpolation nodes
	Feval = evaltrig.(X, Y, Z, Ref(F̂))
	# check that it satisfies the interpolation condition; also check that 
	# this is the same as inverting the FFT
	Finv = real.(ifft(F̂)) * (2N)^3
	F ≈ Feval ≈ Finv		
end

# ╔═╡ 8de3bc38-9b50-11eb-2ed2-436e4a3da804
md"""
Of course what we are doing here is **complete madness** - the cost of each call 
```julia
evaltrig(x, y, z, Ref(F̂))
```
is ``O(N^3)``! We will therefore try to avoid evaluating trigonometric polynomials on general sets, but only on grids. And when we do that, then we can just use the `ifft`!! Then we can evaluate the trigonometric polynomial at ``O(M^3)`` gridpoints for ``O( (M \log M)^3 )`` operations instead of ``O( (M N)^3 )``.
"""

# ╔═╡ a0c7fb86-9b50-11eb-02c6-37fc9fc78d57
begin 
	"""
	Note this function can evaluate the trigonometric interpolant on a 
	grid that is much finer than the one we used to construct it!
	The implementation is quite messy, but the idea is simple.
	"""
	function evaltrig_grid(F̂::Matrix, Nx, Ny=Nx)
		
	end 
	
end

# ╔═╡ aed51e0c-9b50-11eb-2b2b-0553df4ec03b
md"""
### Approximation results

Approximation in dimension ``d > 1`` is much more subtle than in one dimension. Here, we can only focus on some basic results, but we will return to explore some of that complexity in the third part of this lecture. 

In this first result we are making a very simple statement: if ``f`` has a uniform one-dimensional regularity along all possible one-dimensional coordinate direction slices of ``d``-dimensional space, then the one-dimensional convergence results are recovered up to possibly worse constants (which we sweep under the carpet) and ``d``-dependent logarithmic factors.

**Theorem:** Suppose that ``f \in C_{\rm per}(\mathbb{R}^d)`` that
```math
    x_t \mapsto f({\bf x}) 
	\qquad 
	\begin{cases}
		\in C^p, \\ 
		\text{is analytic and bdd in } \Omega_\alpha,
	\end{cases}
	\qquad \forall {\bf x} \in [0, 2\pi)^d
```
where ``x_t \mapsto f({\bf x})`` means that we keep all other coordinates fixed! Then, 
```math
	\| f - I_N f \|_{L^\infty(\mathbb{R}^d)} 
	\lesssim 
	(\log N)^d \cdot 
	\begin{cases}
		  N^{-p}, \\ 
		 e^{-\alpha N}
	\end{cases}
```

**Proof:** Whiteboard or [LN, Sec. 8.1, 8.2].

Note: for the sake of simplicity we won't go into the subtleties of additional factors due to the modulus of continuity, but this can be easily incorporated.
"""

# ╔═╡ b80ba91e-9b50-11eb-2e67-bfd1a71f069e
md"""
## §6.2 Spectral Methods in 2D and 3D

"""

# ╔═╡ e47464b6-9bd2-11eb-2404-6b4a6459ee31
md"""
## §6.3 Approximation in High Dimension



"""

# ╔═╡ Cell order:
# ╟─3f1cfd12-7b86-11eb-1371-c5795b87ef5b
# ╟─76e9a7f6-86a6-11eb-2741-6b8759be971b
# ╟─551a3a58-9b4a-11eb-2596-6dcc2edd334b
# ╟─a7e7ece4-9b48-11eb-26a8-213556563a81
# ╟─102f6eca-9b4a-11eb-23b8-210bd9100faa
# ╟─974d10ae-9b9c-11eb-324f-21c03e345ca4
# ╠═6e96c814-9b9a-11eb-1200-bf99536f9369
# ╟─4741a420-9b9a-11eb-380e-07fc50b805c9
# ╟─95edae16-9b9d-11eb-2698-8d52b0f18a57
# ╠═2ff639c8-9b9f-11eb-000e-37bbbea50dc5
# ╠═bd5807fc-9b9e-11eb-1434-87af91d2d296
# ╟─bedaf224-9b9e-11eb-0a7d-ad170bfd73a7
# ╟─bdcd0340-9b9e-11eb-0ee6-e7e0993e8fe8
# ╟─f7454b4a-9bc2-11eb-2ab9-dfc016fd57de
# ╟─1506eaf4-9b9a-11eb-170d-95834314fb84
# ╟─1eb00bb6-9bc8-11eb-3af9-c33ef5b6ab15
# ╟─28aa73fe-9bc8-11eb-3266-f725f73a3159
# ╠═9b31019a-9bc8-11eb-0372-43ea0c0d8fc3
# ╟─c9b0923c-9bce-11eb-3861-4d201b1765a8
# ╠═abc29124-9bc8-11eb-268d-c5115c341b58
# ╟─8de3bc38-9b50-11eb-2ed2-436e4a3da804
# ╠═a0c7fb86-9b50-11eb-02c6-37fc9fc78d57
# ╟─aed51e0c-9b50-11eb-2b2b-0553df4ec03b
# ╠═b80ba91e-9b50-11eb-2e67-bfd1a71f069e
# ╠═e47464b6-9bd2-11eb-2404-6b4a6459ee31
