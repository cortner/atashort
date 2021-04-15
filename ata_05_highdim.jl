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

# ╔═╡ 76e9a7f6-86a6-11eb-2741-6b8759be971b
md"""
## §5 Approximation in Moderate and High Dimension

After our interlude on polynomial approximation, rational approximation, etc, we now return to trigonometric approximation by in dimension ``d > 1``. That is, we will consider the approximation of functions 
```math
	f \in C_{\rm per}(\mathbb{R}^d)
	:= \big\{ g \in C(\mathbb{R}^d) \,|\, 
			  g(x + 2\pi \zeta) = g(x) \text{ for all } \zeta \in \mathbb{Z}^d \big\}.
```


Our three goals for this lecture are

* 5.1 Approximation in moderate dimension, ``d = 2, 3``
* 5.2 Spectral methods for solving PDEs in two and three dimensions
* 5.3 Approximation in "high dimension"
"""


# ╔═╡ 551a3a58-9b4a-11eb-2596-6dcc2edd334b
md"""
**TODO:** Review of Fourier series, trigonometric interpolation.
"""

# ╔═╡ a7e7ece4-9b48-11eb-26a8-213556563a81
md"""
## § 5.1 Approximation in Moderate Dimension

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
	function evaltrig_grid(F̂::Matrix, Mx, My=Mx)
		Nx, Ny = size(F̂); Nx ÷= 2; Ny ÷= 2
		Ix = 1:Nx+1; Jx = Nx+2:2Nx; Kx = (2Mx-Nx+2):2Mx 
		Iy = 1:Ny+1; Jy = Ny+2:2Ny; Ky = (2My-Ny+2):2My		
		Ĝ = zeros(ComplexF64, 2Mx, 2My)
		Ĝ[Ix, Iy] = F̂[Ix, Iy] 
		Ĝ[Ix, Ky] = F̂[Ix, Jy] 
		Ĝ[Kx, Iy] = F̂[Jx, Iy]
		Ĝ[Kx, Ky] = F̂[Jx, Jy]
		return real.(ifft(Ĝ) * (4 * Mx * My))
	end 
	
	function trigerr(f::Function, F̂::Matrix, Mx, My=Mx)
		G = evaltrig_grid(F̂, Mx, My)
		X, Y = xygrid(Mx, My) 
		return norm(f.(X, Y) - G, Inf)		
	end
end

# ╔═╡ 047a3e4a-9c83-11eb-142c-73e070fd4731
let f = (x, y) -> sin(x) * sin(y), N = 2, M = 30
	F̂ = triginterp2d(f, N)
	# coarse grid function 
	xc = xgrid(N)
	Fc = real.(ifft(F̂) * (2N)^2)
	# fine grid function 
	xf = xgrid(M)
	Ff = evaltrig_grid(F̂, M)
	plot( surface(xc, xc, Fc, colorbar=false), 
		  surface(xf, xf, Ff, colorbar=false), size = (500, 200) )
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
## §5.2 Spectral Methods in 2D and 3D

We now return to the solution of (partial) differential equations using trigonometric polynomials, i.e. spectral methods. The ideas carry over from the one-dimensional setting without essential changes. 

### §5.2.0 Review Differentiation Operators

Recall from §2 that the fundamental property of Fourier spectral methods that we employed is that, if 
```math
	u_N(x) = \sum_{k = -N}^N \hat{U}_k e^{i k x}
```
is a trigonometric polynomial, then 
```math 
	u_N'(x) = \sum_{k = -N}^N \big[ i k \hat{U}_k \big]  e^{i k x}, 
```
that is, all homogeneous differentiation operators are *diagonal* in Fourier space. This translates of course to ``d`` dimensions: if 
```math 
	u_N({\bf x}) = \sum_{{\bf k}} \hat{U}_{\bf k} e^{ i {\bf k} \cdot {\bf x}}, 
```
then 
```math 
	\frac{\partial u_N}{\partial x_t}({\bf x}) = 
		\sum_{{\bf k}} \big[ i k_t \hat{U}_{\bf k} \big] e^{ i {\bf k} \cdot {\bf x}}, 
```
More generally, if ``L`` is a homogeneous differential operator, 
```math
	Lu = \sum_{\bf a} c_{\bf a} \prod_{t=1}^d \partial_{x_t}^{a_t} u, 
```
then this becomes 
```math 
	\widehat{Lu}(k) = \hat{L}(k) \hat{u}(k).
```
where 
```math
	\hat{L}(k) = \sum_{\bf a} c_{\bf a} \prod_{t=1}^d (i k_{t})^{a_t} 
```
We will now heavily use this property to efficiently evaluate differential operators.
"""

# ╔═╡ 677ed7ee-9c15-11eb-17c1-bf6e27df8dba
md"""
### §5.2.1 Homogeneous elliptic boundary value problem

We begin with a simple boundary value problem the *biharmonic equation*, a 4th order PDE modelling thin structures that react elastically to external forces.
```math
	\Delta^2 u = f, \qquad \text{ with PBC.}	
```

[TODO whiteboard: derive the multiplier] 

If ``f_N, u_N`` are trigonometric polynomials and ``f_N = \Delta^2 u_N`` then we can write 
```math
	\hat{F}_{\bf k} = |{\bf k}|^4 \hat{U}_{\bf k}
```
This determines ``\hat{U}_k`` except when ``k = 0``. Since the PDE determines the solution only up to a constant we can either prescribe that constant or pick any constant that we like. It  is common to require that ``\int u_N = 0``, which amounts to ``\hat{U}_0 = 0``. 

"""

# ╔═╡ 1abf39ea-9c19-11eb-1bb0-178d4f3e7de4
begin
	# kgrid(N) = [0:N; -N+1:-1]
	kgrid2d(Nx, Ny=Nx) = (
			[ kx for kx in kgrid(Nx), ky in 1:2Ny ], 
			[ ky for kx in 1:2Nx, ky in kgrid(Ny) ] )
end

# ╔═╡ ba97522c-9c16-11eb-3734-27688484ef6f
let N = 64, M = 40, f = (x, y) -> exp(-3(cos(x)sin(y))) - exp(-3(sin(x)cos(y)))
	F̂ = triginterp2d(f, N)
	Kx, Ky = kgrid2d(N)
	L̂ = (Kx.^2 + Ky.^2).^2
	L̂[1] = 1
	Û = F̂ ./ L̂
	Û[1] = 0
	U = real.(ifft(Û) * (2N)^2)
	x = xgrid(N)
	contourf(x, x, U, size = (300,300), colorbar=false)
end

# ╔═╡ 9c7a4006-9c19-11eb-15ac-890dce21f2ec
md"""

### Error Analysis

The error analysis proceeds essentially as in the one-dimensional case. We won't give too many details here, but only confirm that neither the results nor the techniques change fundamentally.

To proceed we need just one more ingredient: the multi-dimensional Fourier series. We just state the results without proof: Let ``f \in C_{\rm per}(\mathbb{R}^d)`` then 
```math
	f({\bf x}) = \sum_{{\bf k} \in \mathbb{Z}^d} \hat{f}_{\bf k} e^{i {\bf k} \cdot {\bf x}},
```
where the convergence is in the least square sense (``L^2``), and uniform if ``f`` is e.g. Hölder continuous. 

Thus, for sufficiently smooth ``f`` we can write the solution of the biharmonic equation ``u`` also as a Fourier series with coefficients 
```math
	\hat{u}_{\bf k} = 
	\begin{cases}
		\hat{f}_{\bf k} / |{\bf k}|^{-4}, & {\bf k} \neq 0, \\ 
 	    0, & \text{otherwise.}
	\end{cases}
```
Note that this requires ``\hat{f}_{\bf 0} = 0``. 

Since ``|{\bf k}|^{-4}`` is summable it follows readily that the equation is max-norm, stable, i.e., 
```math
	\|u\|_\infty \leq C \|f\|_\infty,
```
and we cannow argue as in the 1D case that 
```math
	\Delta^2 (u - u_N) = f - f_N \qquad \Rightarrow \qquad 
	\|u - u_N \|_\infty \leq C \| f - f_N \|_\infty.
```
Thus, the approximation error for ``f - f_N`` translates into an approximation error for the solution. 

"""

# ╔═╡ e6b1374a-9c7f-11eb-1d90-1d6d40914d90
md"""
We can test the result for a right-hand side where we have a clearly defined rate, e.g., 
```math
	f(x, y) = \frac{1}{1 + 10 (\cos^2 x + \cos^2 y)}.
```
According to our results above we expect the rate 
```math 
	e^{-\alpha N}, \qquad \text{where} \qquad 
	\alpha = \sinh^{-1}(1 / \sqrt{10})
```
"""

# ╔═╡ 56eeada8-9c80-11eb-1422-79f53b49b47e
let f = (x,y) -> 1 / (1 + 10 * (cos(x)^2 + cos(y)^2)), NN = 4:4:40, M = 300
	err(N) = trigerr(f, triginterp2d(f, N), M)
	plot(NN, err.(NN), lw=2, ms=4, m=:o, label = "error", 
		 yscale = :log10, size = (300, 250), 
	     xlabel = L"N", ylabel = L"\Vert f - I_N f \Vert_\infty")
	α = asinh(1 / sqrt(10))
	plot!(NN[5:end], 2 * exp.( - α * NN[5:end]), c=:black, lw=2, ls=:dash, label = L"\exp(-\alpha N)")
end

# ╔═╡ 64a5c40a-9c84-11eb-22fa-ff8524822e62
md"""
We will leave the domain of rigorous error analysis now and focus more on issues of implementation. Although everything we do here can be easily extended to 3D, we will focus purely on 2D since the runtimes are more manageable and the visualization much easier.
"""

# ╔═╡ 6c189d08-9c15-11eb-1d47-a902e8997cc3
md"""
### §5.2.2 A 2D transport equation

```math
	u_t + {\bf v} \cdot \nabla u = 0
```
We discretize this as 
```math
	\frac{d\hat{U}_{\bf k}}{dt}
	+ i ({\bf v} \cdot {\bf k}) \hat{U}_{\bf k} = 0
```
And for the time-discretisation we use the leapfrog scheme resulting in 
```math
	\frac{\hat{U}_{\bf k}^{n+1} - \hat{U}_{\bf k}^{n-1}}{2 \Delta t}
	= - i ({\bf v} \cdot {\bf k}) \hat{U}_{\bf k}^n.
```
"""

# ╔═╡ 993e30d8-9c86-11eb-130a-41a1bb825e7c
let u0 = (x, y) -> exp(- 3 * cos(x) - cos(y)), N = 20, Δt = 3e-3, Tf = 2π
		v = [1, 1]
	
	Kx, Ky = kgrid2d(N)
	dt_im_vdotk = 2 * Δt * im * (v[1] * Kx + v[2] * Ky)
	Û1 = triginterp2d(u0, N)
	Û0 = Û1 + 0.5 * dt_im_vdotk .* Û1
	
	xx = xgrid(N)
	
	t = 0.0
	@gif for _ = 1:ceil(Int, Tf/Δt)
		Û0, Û1 = Û1, Û0 - dt_im_vdotk .* Û0
		contourf(xx, xx, real.(ifft(Û1) * (2N)^2), 
			     colorbar=false, size = (300, 300), 
				 color=:viridis)
	end every 15
end

# ╔═╡ 704b6176-9c15-11eb-11ba-ffb4491a6003
md"""

### §5.2.3 The Cahn--Hilliard Equation

Next, we solve a nonlinear evolution equation: the Cahn--Hilliard equation, which models phase separation of two intermixed liquids,
```math
	(-\Delta)^{-1} u_t = \epsilon \Delta u - \frac{1}{\epsilon} (u^3 - u)
```
The difficulty here is that a trivial semi-implicit time discretisation
```math
	u^{(n+1)} + \epsilon \tau \Delta^2 u^{(n+1)} = 
	u^{(n)} + \frac{\tau}{\epsilon} \Delta (u^3 - u)
```
has time-step restriction ``O( \epsilon N^{-2} )``. We can stabilise with a (local) convex-concave splitting such as
```math
	(1 + \epsilon \tau \Delta^2 - C \tau \Delta) u^{(n+1)}
	= (1-C \tau \Delta) u^{(n)} + \frac{\tau}{\epsilon} \Delta (u^3 - u)^{(n)}
```
Since ``(u^3-u)' = 3 u^2 - 1 \in [-1, 2]`` we need ``C \geq 2/\epsilon`` to get ``\tau``-independent stability. We then choose the time-step ``\tau = h \epsilon`` to make up for the loss of accuracy.

In reciprocal space, the time step equation becomes
```math
	(1+\epsilon \tau |k|^4 + C \tau |k|^2) \hat{u}^{(n+1)} 
	= 
	\big(1+C\tau |k|^2 + \frac{\tau}{\epsilon} |k|^2\big) \hat{u}^{(n)} 
	- \frac{\tau}{\epsilon} |k|^2 (\widehat{u^3})^{(n)}
```
(For something more serious we should probably implement a decent adaptive time-stepping strategy.)	
"""


# ╔═╡ fd07f1d2-9c89-11eb-023f-e96ccdd43a7e
let N = 64, ϵ = 0.1,  Tfinal = 8.0
	h = π/N     # mesh size 
	C = 2/ϵ     # stabilisation parameter
	τ = ϵ * h   # time-step 

	# real-space and reciprocal-space grids, multipliers
	xx = xgrid(N)
	Kx, Ky = kgrid2d(N)
	Δ = - Kx.^2 - Ky.^2

	# initial condition
	U = rand(2N, 2N) .- 0.5

	# time-stepping loop
	@gif for n = 1:ceil(Tfinal / τ)
		F̂ =  (1 .- C*τ*Δ) .* fft(U) + τ/ϵ * (Δ .* fft(U.^3 - U))
		U = real.(ifft( F̂ ./ (1 .+ ϵ*τ*Δ.^2 - C*τ*Δ) ))
		contourf(xx, xx, U, color=:viridis, 
			     colorbar=false, size = (400,400))
	end every 3
end

# ╔═╡ 7489004a-9c15-11eb-201d-91f34cb40c6f
md"""

### §5.2.4 A nonlinear eigenvalue problem 



"""

# ╔═╡ e47464b6-9bd2-11eb-2404-6b4a6459ee31
md"""
## §5.3 Approximation in High Dimension

Let us return to the approximation of an analytic function ``f \in C_{\rm per}(\mathbb{R}^d)``. We have proven above that 
```math
	\|f - I_N f \|_\infty \lesssim (\log N)^d e^{- \alpha N}
```
for some ``\alpha > 0``, which appears to be an excellent generalisation of our 1D results. However, this is deceptive. The number of parameters required to define or to evaluate ``I_N f`` is ``O(N^d)``, and the work to construct it is ``O( (N \log N)^d)``. This cost is **considerable** when ``d`` is large. For example, consider a function in just 10 dimensions, and discretise with just 10 parameters in each coordinate direction (``N = 5``), then we obtain already ``10^10``, or, 10 billion, parameters. Going into slightly higher dimension, say ``d = 100``, we require ``10^100`` parameters. The number of atoms in the universe is estimated to be around ``10^{78}`` to ``10^{82}``. This extreme explosion of computational cost with high dimension is called the [**CURSE OF DIMENSIONALITY**](https://en.wikipedia.org/wiki/Curse_of_dimensionality).


If we want to express this in terms of a error-cost relationship, we define a new convergence parameter ``{\rm COST}`` and rewrite our error estimate as 
```math
	\|f - I_N f \|_\infty \lesssim d^{-d} (\log {\rm COST})^d e^{- \alpha {\rm COST}^{1/d}}
```
We can equally see how this rate deteriorates rapidly with increasing ``d``. 

How is approximation in high dimension then possible? The main point is that it is only possible if we can exploit specific features of the target function ``f``. For example, in the next lecture I will introduce approximation problems in dimension ``d \approx 100``, provided it satisfies some very strong symmetries.
"""

# ╔═╡ 228aba30-9cd9-11eb-0bdc-e7b4237757ce
md"""
### A Greedy Algorithm

A simple way --- at least conceptually --- to try and find "sparsity" in the high-dimensional function is via Fourier analysis. Suppose we have the Fourier series of ``f``, i.e., 
```math
	f({\bf x}) = \sum_{{\bf k} \in \mathbb{Z}^d} \hat{f}_{\bf k} e^{i {\bf k} \cdot {\bf x}}
```
then we can ask what the effect would be of projecting ``f`` not onto the "canonical" space
```math
	{\rm span} \big\{ e^{i {\bf k} \cdot {\bf x}} \,|\, {\bf k} \in \{-N,\dots, N\}^d \big\},
```
but onto an *arbitrary* subset of the full basis. I.e., we first choose a finite set  
```math
	\mathcal{K} \subset \mathbb{Z}^d
```
and then define the corresponding ``L^2``-projection 
```math
	t_{\mathcal{K}} := \sum_{{\bf k} \in \mathcal{K}}  \hat{f}_{\bf k} e^{i {\bf k} \cdot {\bf x}}.
```
Since the multi-variate trigonometric polynomials are orthonormal and bounded, the error in the ``L^2`` and max-norms will be [ASSIGNMENT]
```math
	\begin{aligned}
		\| f - t_{\mathcal{K}} \|_{L^2( (0, 2\pi)^2 )} 
			&= \bigg( \sum_{{\bf k} \in \mathbb{Z}^d \setminus \mathcal{K}} 
						|\hat{f}_k|^2 \bigg)^{1/2}, \qquad \text{and} \\ 
		\| f - t_{\mathcal{K}} \|_{\infty}
			&\leq \sum_{{\bf k} \in \mathbb{Z}^d \setminus \mathcal{K}}  |\hat{f}_k|.
	\end{aligned}
```
Thus, we see that to get the best ``M``-term approximation in the ``L^2`` sense, or the best possible upper bound as derived above in the max-norm, we should fill ``mathcal{K}`` with the indices ``{\bf k}`` corresponding to the ``M`` largest Fourier coefficients. 

This leads to the following, purely hypothetical, "greedy" algorithm: 

**Greedy Best Approximation Algorithm:**
* Compute all fourier coefficients ``\hat{f}_{\bf k}``
* Sort then in descenting order: ``{\bf k}_1, {\bf k}_2, \dots`` such that ``|\hat{f}_{{\bf k}_m}| \geq |\hat{f}_{{\bf k}_{m+1}}|``.
* Define ``\mathcal{K}_M := \{{\bf k}_1, \dots, {\bf k}_M\}`` and the resulting best ``M``-term approx ``t_{\mathcal{K}_M}``.

**Remarks:**
* The greedy algorithm produces the best ``M``-term approximation in the ``L^2``-sense but it is by no means clear that it gives the best ``M``-term approximation in the max-norm (or indeed quasi-best...) 
* Since the basis functions are most efficiently computed via a recursion it is not even clear that the best ``M``-term approximation gives also the best cost error ratio. 
* It is not always clear how to select "the most important" Fourier coefficients in practise.

[[DeVore, 1997]](https://www.cambridge.org/core/journals/acta-numerica/article/abs/nonlinear-approximation/C8E028C39B8A849690D0EC418516A934)
"""

# ╔═╡ 1e27020a-9cd9-11eb-2b27-8956a221427c
md"""
### Mixed Regularity and Sparse Grids

Because it can be difficult to implement a best ``M``-term approximation in practise we are also interested in how general function classes lead to generically good choices of the sparse basis set ``\mathcal{K}``. 

A very general and broadly applicable idea is to exploit the existence of **mixed derivatives.**

For example, suppose that ``f \in C_{\rm per}`` and that the derivatives 
```math
	f, \partial_{x_1} f, \partial_{x_2} f,  \partial_{x_1} \partial_{x_2} f \in L^2
```
then it is straightforward to show that [LN, Sec. 8.4] and **[WHITEBOARD]**
```math
	\sum_{{\bf k} \in \mathbb{Z}^2} 
	(1 + |k_1|)^2 (1 + |k_2|)^2 |\hat{f}_{\bf k}|^2 
	\leq 
	\|f\|_2^2 + \| \partial_{x_1} f \|_2^2 + \|\partial_{x_2} f\|_2^2 
	+ \| \partial_{x_1} \partial_{x_2} f \|_2^2 < \infty.
```
"""

# ╔═╡ 80cf2b4a-9d55-11eb-2908-8f19c79bca96
md"""
This gives us a rough estimate on how the Fourier coefficients must be decaying, i.e. it suggests that they might decay like some function of ``(1 + |k_1|)^{-1} (1 + |k_2|)^{-1}``. The idea now is to use this "characteristic decay" to produce the sparse basis ``\mathcal{K}`` instead of the Fourier coefficients themselves. That is, we can choose 
```math
	\mathcal{K}_N := \big\{ {\bf k} : (1 + |k_1|) (1 + |k_2|) \leq N \big\}.
```
This is a variant of the *hyperbolic cross approximation*. Named after the level sets of the characteristic decay function ``\omega({\bf k}) = (1 + |k_1|) (1 + |k_2|)``.
"""

# ╔═╡ abddcd1e-9d55-11eb-158a-098df3e34ffe
let N = 16, ω = kk -> (1+abs(kk[1])) * (1+ abs(kk[2]))
	k = -N:N
	Kx = k * ones(2N+1)'; Ky = (Kx')[:]; Kx = Kx[:]
	Ihc = findall(ω.([ [kx, ky] for (kx, ky) in zip(Kx, Ky) ]) .<= N+1)
	scatter(Kx, Ky, ms=3, label = "Tensor Grid")
	scatter!(Kx[Ihc], Ky[Ihc], ms=3, label = "Hyperbolic Cross", 
				size = (480,300), legend = :outertopright, 
				xlabel = L"k_1", ylabel = L"k_2")
end

# ╔═╡ a59bcffe-9d56-11eb-10d5-8da3f1e7a7ea
md"""
More generally, one can try to classify functions in high dimension in terms of *Korobov classes*: given a weight ``\omega : \mathbb{Z}^d \to (0, 1]`` we consider the functions
```math
	\begin{aligned}
	\mathcal{A}_{\omega}^{(2)} 
	&:= \bigg\{ f : \sum_{{\bf k} \in \mathbb{Z}^d} \omega({\bf k })^2 |\hat{f}_{\bf k}|^2 < \infty \bigg\} \\ 
	\mathcal{A}_{\omega}^{(\infty)} 
	&:= \bigg\{ f : \sum_{{\bf k} \in \mathbb{Z}^d} \omega({\bf k }) |\hat{f}_{\bf k}| < \infty \bigg\} 
\end{aligned}
```

Suppose e.g. that ``f \in \mathcal{A}^{(p)}_\omega`` then a natural way to choose ``\mathcal{K}`` would be 
```math
	\mathcal{K}_\epsilon := \big\{ {\bf k} : \omega({\bf k}) \leq \epsilon^{-1} \big\}, 
```
and this would lead to the error bound **[WHITEBOARD]**
```math
	\| f - t_{\mathcal{K}_\epsilon} \|_p \leq \epsilon.
```

What remains to be done now is to understand the computational cost associated with these *sparse approximations*, or maybe somewhat easier, what is the size of the basis set --- e.g. ``\mathcal{K}_\epsilon`` above --- that we selected? For that we return to the hyperbolic cross example but generalize it in various ways.
"""

# ╔═╡ 4e8fd706-9d5a-11eb-392d-5fb9f8ba4b30
md"""
### Hyperbolic cross in ``d`` dimensions

A fairly general class of functions where some explicit results are possible are those that possess all mixed derivatives up to some order ``r``, that is, 
```math
	\frac{\partial^j f}{\partial x_{a_1} \cdots \partial x_{a_j}} \in L^2, 
	\qquad j \leq r, \qquad a_i \in \{1, \dots,  d\}.
```
In this case ``f \in \mathcal{A}^{(2)}_{\omega^{\rm hc}_r}`` with 
```math 
	\omega^{\rm hc}_r({\bf k}) = \prod_{i = 1}^d (1 + |k_i|)^r.
```
In this case it is common to write 
```math
	\mathcal{K}_N^{\rm hc} := \big\{ {\bf k} \,|\, \omega^{\rm hc}_1({\bf k}) \leq N \big\},
```
and the parameter ``r`` becomes a regularity parameters. Let ``t_N^{\rm hc}`` denote the corresponding projection onto the sparse subspace.

We then have the following results, which we state without proof [LN, Sec. 8.4]: 
* If ``f \in \mathcal{A}^{(p)}_{\omega^{\rm hc}_r}``, ``p \in \{2, \infty\}``, then 
```math
	\| f - t_N^{\rm hc} \|_p \leq C N^{-r}
```
* The size of the basis, i.e. the number of terms can be estimated by 
```math
	\# \mathcal{K}_N^{\rm hc} \leq N (\log N)^{d-1}.
```
* We can use ``\# \mathcal{K}_N^{\rm hc} \geq N`` to obtain 
```math 
	N \geq \frac{\# \mathcal{K}_N^{\rm hc}}{ (\log \# \mathcal{K}_N^{\rm hc})^{d-1} }
```
and hence 
```math
	\| f - t_N^{\rm hc} \|_p 
	\lesssim \bigg( \frac{\# \mathcal{K}_N^{\rm hc}}{ (\log \# \mathcal{K}_N^{\rm hc})^{d-1} } \bigg)^{-r}
	\approx \bigg( \frac{{\rm COST}}{ [\log {\rm COST}]^{d-1} } \bigg)^{-r}
```

Similar results can be obtained for analytic functions [LN, Sec. 8].
"""

# ╔═╡ 5570b0e0-9e09-11eb-2bd5-fb1448144270
begin
	# some simple codes to experiment with sparse approximation in 2D
	
	"""
	plot an image showing where large coefficients are concentrated
	"""
	function imcoeffs(C; logscale=true)
		if logscale 
			img = log.(abs.(C) .+ 1e-15)
		else
			img = abs.(C)
		end
		a, b = extrema(img)
		img = (img .- a) ./ (b - a)
		return Gray.(img)		
	end

	"just the naive tensor approximation"
	trig2d_err(f, N, Nerr = 4*N) = trigerr(f, triginterp2d(f, N), Nerr)

	"""
	approximation error for the greedy approximation algorithm 
	"""	
	function greedy2d_err(f, M, Nmax, Nerr = 3 * Nmax)
		F̂ = triginterp2d(f, Nmax) 
		Idel = sortperm(abs.(F̂)[:])[1:(length(F̂)-M)]
		F̂[Idel] .= 0
		return trigerr(f, F̂, Nerr)
	end
	
	"""
	approximation error for a sparse approximation algorithm with 
	prescribed weight function
	
	Input:
	* `f`: target function
	* `ω, N`: weight function  and sparse "degree"
	* `Nmax, Nerr`: grid size to get coefficients and errors
	
	Output: error, number of terms
	"""	
	function sparse2d_err(f, ω, N, Nmax, Nerr = 2 * Nmax)
		F̂ = triginterp2d(f, Nmax) 
		Kx, Ky = kgrid2d(Nmax)
		Idel = findall(ω.(Kx, Ky) .> N)
		F̂[Idel] .= 0
		return trigerr(f, F̂, Nerr), (length(F̂) - length(Idel))
	end
end;

# ╔═╡ 148992ca-9d66-11eb-3787-b3db6ad34765
md"""
### Some sparsity patterns ... 

In practise it is rare that the real "sparsity pattern" in the Fourier coefficients closely matches such a simple rule. The reason is quite simply that we have only used one single property of ``f``, but there may be many other influences such as symmetries, or more complex regularity properties that are more difficult to capture in such simple terms.

To understand "real-life" sparsity patterns we will look at a few 2D examples. These are of course extremely limited and in no way reflect the complexity of proper high-dimensional approximation.

First a "random" analytic function 
```math
f(x_1, x_2) = \exp\Big( \sin(2 x_1) \cos( \sin(x_2)) \Big)
```
In the following image we plot the magnitudes of the Fourier coefficients, with the ``x``-axis representing ``k_1`` and the ``y``-axis ``k_1``. It is important to remember here the ordering of the Fourier coefficients, ``(0, 1, \dots, N, -N+1, -N+2, \dots, -1)`` i.e. the four corners represent the small ``{\bf k}`` while the center represents the large ``{\bf k}``.
"""

# ╔═╡ fec961f6-9d5e-11eb-3263-ed551a9e753f
let f = (x1, x2) -> exp(sin(2*x1)*cos(sin(x2))), N = 32
	F̂ = triginterp2d(f, N)
	imcoeffs(F̂)
end

# ╔═╡ fc31d3c4-9e17-11eb-089b-e740ba429e45
let f = (x1, x2) -> exp(sin(2*x1)*cos(sin(x2))),
		Nmax = 64, NN = 3:3:30, MM = 20:50:1000,
		ω = (k1, k2) -> sqrt(k1^2+k2^2) + 1000 * (isodd(k1) || isodd(k2)), 
		NNsp = 3:3:30
	
	errsmax = [ trig2d_err(f, N) for N in NN ] 
	errsgr = [ greedy2d_err(f, M, Nmax)  for M in MM ]
	errs_M_sp = [ sparse2d_err(f, ω, N, Nmax)  for N in NNsp ]
	errssp = [ a[1] for a in errs_M_sp ]
	Msp = [ a[2] for a in errs_M_sp ]
	
	plot( (2*NN).^2, errsmax, lw = 2, label = "tensor", 
		   yscale = :log10, size = (400, 300), 
		   xlabel = "# terms", ylabel = L"\Vert f - t_{\mathcal{K}} \Vert" )
	plot!( MM, errsgr, lw = 2, label = "greedy" )
	plot!( Msp, errssp, lw = 2, label = "sparse" ) 
end

# ╔═╡ eb4f6e28-9d5d-11eb-14af-350b86a80a62
md"""
A multi-dimensional witch of Agnesi, 
```math
	f({\bf x}) = \frac{1}{1 + c |\sin.({\bf x})|^2}
```
"""

# ╔═╡ 8af26c20-9d61-11eb-01dc-9d4af89e0459
let f = (x1, x2) -> 1 / (1 + 50*(sin(x1)^2 + sin(x2)^2)), N = 256
	F̂ = triginterp2d(f, N)
	imcoeffs(F̂[1:2:end, 1:2:end])
end

# ╔═╡ 5c78ea82-9e19-11eb-1027-eb92828adc98
let f = (x1, x2) -> 1 / (1 + 50*(sin(x1)^2 + sin(x2)^2)), N = 256, Nsp = 100, Mgr = 8_000
	F̂ = triginterp2d(f, N)
	Kx, Ky = kgrid2d(N)
	ω = (k1, k2) -> norm([k1, k2]) + 1000 * (isodd(k1) || isodd(k2))
	Idelsp = findall(ω.(Kx, Ky) .> Nsp)
	F̂sp = copy(F̂)
	F̂sp[Idelsp] .= 0
	F̂gr = copy(F̂) 
	Idelgr = sortperm(abs.(F̂gr)[:])[1:(length(F̂)-Mgr)]
	F̂gr[Idelgr] .= 0
	[ imcoeffs(F̂sp[1:2:end, 1:2:end]) Gray.(ones(N, 20)) imcoeffs(F̂gr[1:2:end, 1:2:end]) ]
end

# ╔═╡ 12abd79c-9e0b-11eb-07c6-a7790969d30e
let f = (x1, x2) -> 1 / (1 + 50 * (sin(x1)^2 + sin(x2)^2)), 
		Nmax = 64, NN = 4:4:40, MM = 50:50:2_000,
		ω = (k1, k2) -> sqrt(k1^2+k2^2) + 1000 * (isodd(k1) || isodd(k2)), 
		NNsp = 5:5:60 							# a hack to exploit the reflection symmetries!!
 											
	errsmax = [ trig2d_err(f, N) for N in NN ] 
	errsgr = [ greedy2d_err(f, M, Nmax)  for M in MM ]
	errs_M_sp = [ sparse2d_err(f, ω, N, Nmax)  for N in NNsp ]
	errssp = [ a[1] for a in errs_M_sp ]
	Msp = [ a[2] for a in errs_M_sp ]
	
	plot( (2*NN).^2, errsmax, lw = 2, label = "tensor", 
		   yscale = :log10, size = (400, 300) )
	plot!( MM, errsgr, lw = 2, label = "greedy" )
	plot!( Msp, errssp, lw = 2, label = "sparse" )
end

# ╔═╡ 771181e8-9d5f-11eb-05fb-83a91ead7fa0
md"""
Secondly, an examples taken from electron [transport models](https://arxiv.org/pdf/1907.01314.pdf) (for this function we would again want to evaluate it as a bi-variate matrix function!)
```math
	f(x_1, x_2) = \frac{f_\beta(x_1) - f_\beta(x_2)}{x_1 - x_2} \cdot \frac{1}{x_1 - x_2 + i \eta},
```
where ``f_\beta`` is the Fermi-Dirac function and ``\eta`` a physical parameter (relaxation time of a wave) which is small. The prefactor ``\frac{f_\beta(x_1) - f_\beta(x_2)}{x_1 - x_2}`` is in fact analytic (a bit of an exercise) so we focus on the last term ``\frac{1}{x_1 - x_2 + i \eta}``. 

``\eta = `` $(@bind __eta Slider(0.01:0.01:0.3, show_value=true))
"""

# ``N = `` $(@bind __N Slider(32:32:512, show_value=true))



# ╔═╡ 2d580e02-9d62-11eb-282d-394a4745ffff
let η = __eta, N = 128, f = (x1, x2) -> 1 / (sin(x2) - sin(x1) + im * η)
	F̂ = triginterp2d(f, N)
	imcoeffs(F̂; logscale=true)
end

# ╔═╡ 6bb73e98-9e11-11eb-37f2-650140076bc4
let η = 0.03, f = (x1, x2) -> real(1 / (sin(x1) - sin(x2) + im * η)), 
		Nmax = 256, NN = 10:10:100, MM = 500:500:12_000
	
	errsmax = [ trig2d_err(f, N) for N in NN ] 
	errsgr = [ greedy2d_err(f, M, Nmax)  for M in MM ]
	
	plot( (2*NN).^2, errsmax, lw = 2, label = "tensor", 
		   yscale = :log10, size = (400, 300) )
	plot!( MM, errsgr, lw = 2, label = "greedy" )
end

# ╔═╡ 27c4b85c-9cd9-11eb-2f0d-71d2ca0913f9
md"""
### Outlook

In this last lecture I was focusing mostly on theory. Getting high-dimensional approximation to work in practise, for "real" problems is computationally much more challenging. Many further ideas might be needed, e.g. next week we will talk about a sparse ACE approximation which involves reductions due to: sparse grids, symmetrized basis, recursive evaluation schemes, body-order expansion, to name just a few. The very basic concepts we discussed here will normally work only up around 10 dimensions.

The relatively simple ideas outlined above should therefore only be taken as motivation for further reading. Most work in this domain is necessarily application specific. For example, a famous and notoriously difficult problem (with recent progress made by Google [[1]](https://arxiv.org/abs/2007.15298), [[2]](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.033429)) is the solution of Schrödinger's equation (talk to Professor Chen for more details). But the overarching idea is that "real-world" signals however high-dimensional should be representable in some sparse format. But exactly what these formats are can vary greatly across different applications. Automagically discovering these sparsities is one of the goals of artificial neural networks (success is mixed...). Analysis and modelling can do a lot to help!
"""

# ╔═╡ 72f978e8-9d61-11eb-13ef-4ff5f4ef690a
using Colors

# ╔═╡ 3f1cfd12-7b86-11eb-1371-c5795b87ef5b
begin
	using Plots, LaTeXStrings, PrettyTables, DataFrames, LinearAlgebra,
		  PlutoUI, BenchmarkTools, ForwardDiff, Printf, FFTW, Colors
	include("tools.jl")
end;

# ╔═╡ Cell order:
# ╠═3f1cfd12-7b86-11eb-1371-c5795b87ef5b
# ╠═72f978e8-9d61-11eb-13ef-4ff5f4ef690a
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
# ╠═047a3e4a-9c83-11eb-142c-73e070fd4731
# ╟─aed51e0c-9b50-11eb-2b2b-0553df4ec03b
# ╟─b80ba91e-9b50-11eb-2e67-bfd1a71f069e
# ╟─677ed7ee-9c15-11eb-17c1-bf6e27df8dba
# ╠═1abf39ea-9c19-11eb-1bb0-178d4f3e7de4
# ╠═ba97522c-9c16-11eb-3734-27688484ef6f
# ╟─9c7a4006-9c19-11eb-15ac-890dce21f2ec
# ╟─e6b1374a-9c7f-11eb-1d90-1d6d40914d90
# ╟─56eeada8-9c80-11eb-1422-79f53b49b47e
# ╟─64a5c40a-9c84-11eb-22fa-ff8524822e62
# ╟─6c189d08-9c15-11eb-1d47-a902e8997cc3
# ╠═993e30d8-9c86-11eb-130a-41a1bb825e7c
# ╟─704b6176-9c15-11eb-11ba-ffb4491a6003
# ╠═fd07f1d2-9c89-11eb-023f-e96ccdd43a7e
# ╠═7489004a-9c15-11eb-201d-91f34cb40c6f
# ╟─e47464b6-9bd2-11eb-2404-6b4a6459ee31
# ╟─228aba30-9cd9-11eb-0bdc-e7b4237757ce
# ╟─1e27020a-9cd9-11eb-2b27-8956a221427c
# ╟─80cf2b4a-9d55-11eb-2908-8f19c79bca96
# ╠═abddcd1e-9d55-11eb-158a-098df3e34ffe
# ╟─a59bcffe-9d56-11eb-10d5-8da3f1e7a7ea
# ╟─4e8fd706-9d5a-11eb-392d-5fb9f8ba4b30
# ╠═5570b0e0-9e09-11eb-2bd5-fb1448144270
# ╟─148992ca-9d66-11eb-3787-b3db6ad34765
# ╟─fec961f6-9d5e-11eb-3263-ed551a9e753f
# ╠═fc31d3c4-9e17-11eb-089b-e740ba429e45
# ╟─eb4f6e28-9d5d-11eb-14af-350b86a80a62
# ╠═8af26c20-9d61-11eb-01dc-9d4af89e0459
# ╠═5c78ea82-9e19-11eb-1027-eb92828adc98
# ╠═12abd79c-9e0b-11eb-07c6-a7790969d30e
# ╟─771181e8-9d5f-11eb-05fb-83a91ead7fa0
# ╟─2d580e02-9d62-11eb-282d-394a4745ffff
# ╠═6bb73e98-9e11-11eb-37f2-650140076bc4
# ╟─27c4b85c-9cd9-11eb-2f0d-71d2ca0913f9
