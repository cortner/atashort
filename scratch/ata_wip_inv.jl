### A Pluto.jl notebook ###
# v0.14.0

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

# ╔═╡ 2215d20c-9008-11eb-0ced-83e677e8816e
begin
	using Plots, LaTeXStrings, PrettyTables, DataFrames, LinearAlgebra,
          PlutoUI, BenchmarkTools, ForwardDiff, Printf, Random,
	 	  OrdinaryDiffEq, Zygote, DiffEqSensitivity

	include("tools.jl")
end

# ╔═╡ 3e3723c8-9008-11eb-11dc-d77df12db1d9
md"""

## Discovering an ODE

Suppose we are observing some physical process, measuring two important variables, ``{\bf x} = (u, v)`` that we think will describe the process. After making many experiments and taking many measurements we can plot all those experiments to analyze this process, find some trends, etc...

Although it is not important for our purposes, the example we are considering here is a simplified activator-inhibitor model of the [Chlorite-Iodide-Malonic Acid reaction](https://neurophysics.ucsd.edu/courses/physics_173_273/BZ_Epstein_Review.pdf). I am probably using physically unrealistic parameters here.
"""

# ╔═╡ 3d9e50bc-9008-11eb-0080-1bd2057a9aed
md"""
Number of Trajectories: $(@bind _Ntraj1 Slider(1:30, show_value=true))

Final Time: $(@bind _Tf1 Slider(0.5:0.1:20.0, show_value=true))
"""

# ╔═╡ ad9a97b4-8f80-11eb-3513-5f22a6e9a821
	fref = let a = 5.0, b = 2.0
		x -> [ a - x[1] - 4 * x[1] * x[2] / (1 + x[1]^2),
			   b * (x[1] - x[1] * x[2] / (1 + x[1]^2)) ]
end;


# ╔═╡ 22f496b2-8e7e-11eb-3dca-59872173d433
let Tf = _Tf1, Ntraj = _Ntraj1, xmax = 2.5, ymax = 3.0, noise = 0.01
	P = plot(; size = (300,300), ylims = [0,ymax+0.5], xlims = [0,xmax])
	tp = range(0, Tf, length=ceil(Int,30*Tf))

	Random.seed!(1)
	for n = 1:Ntraj
		x0 = rand(2) .* [xmax, ymax]
		prob = ODEProblem((u,p,t) -> fref(u), x0, (0.0, Tf))
		sol = solve(prob, Tsit5())
		X = sol.(tp)
		x = [ xy[1] for xy in X ]; y = [ xy[2] for xy in X ]
		x += noise * randn(length(x)); y += noise * randn(length(y))
		plot!(P, x, y, c=1, label = "")
		plot!(P, [x[1]], [y[1]], lw=0, m=:o, ms=4, c=1, label = "")
	end
	P
end

# ╔═╡ 1d3a3d66-8e84-11eb-3b7a-4dff2ba4cad3
md"""
We believe that there is a mathematical ODE model,
```math
	\dot x = f(x)
```
where ``x = (x_1, x_2)`` but we have no idea where to start determining ``f``. So what we do is we write down a *universal approximator* for ``f`` and fit its parameters to our experimental observations.

E.g., if ``f : \mathbb{R} \to \mathbb{R}`` is ``2\pi``-periodic, then we could use trigonometric polynomials to parameterise ``f`` but this is not the case here, so I will use tensor product Chebyshev polynomials. More on that in a later lecture. The important thing for now is only that we can find a *basis* and approximate
```math
	f(x) \approx F({\bf c}; x) = \sum_k c_k B_k(x),
```
where ``c_k`` are the unknown parameters and ``B_k : \mathbb{R}^2 \to \mathbb{R}^2`` is a basis of the space of algebraic polynomials. This means we can approximate ``f`` to within arbitrary accuracy using this representation.
"""

# ╔═╡ 0c50f854-8e85-11eb-06e4-3572b8017422
md"""
Now we solve
```math
	\dot{X}(t; x_0, {\bf c}) = F({\bf c}; X)
```
for all the same initial values ``x_0 \in X_0`` as we had above. And then we fit to the observations:
```math
	L({\bf c}) := \sum_{x_0 \in \mathcal{X}_0} \sum_{t \in \mathcal{T}}
		\big| X(t; x_0, {\bf c}) - x(t; x_0) \big|^2   \qquad \longrightarrow \qquad
	{\rm minimize~w.r.t.~}{\bf c}
```
This is a general least squares functional. We are minimizing the sum of squares of prediction minus measurement.
"""

# ╔═╡ f40b0f94-8f80-11eb-1364-abb59871cf32
# params = (Tf = 5.0, Ntraj = 30, xmax = 2.5, ymax = 3.5, TT = 0.0:0.2:5.0,
# 		  Nx = 5, Ny = 5)
params = (Tf = 5.0, Ntraj = 10, xmax = 2.5, ymax = 3.5, TT = 0.0:0.5:5.0,
		  Nx = 5, Ny = 5)

# ╔═╡ df50c834-8f7e-11eb-2b0b-79700e9f6f71
train = let params = params
	Random.seed!(1)
	train = []
	for n = 1:params.Ntraj
		x0 = rand(2) .* [params.xmax, params.ymax-0.5]
		prob = ODEProblem((u,p,t) -> fref(u), x0, (0.0, params.Tf))
		sol = solve(prob, Tsit5())
		X = sol.(params.TT)
		push!(train, (x0 = x0, X = X))
	end
	train
end

# ╔═╡ 28f8cfc0-8f81-11eb-33b4-05f3b55c14ca
function evalcheb22(xy, c, params)
	Nx = params.Nx; Ny = params.Ny
	x = -1 + 2 * xy[1] / params.xmax
	y = -1 + 2 * xy[2] / params.ymax
	
	Tx0 = 1.0;  
	Ty0 = 1.0; Ty1 = y
	f = (   Tx0*Ty0 * [c[1], c[Nx*Ny+1]] 
		  + Tx0*Ty1 * [c[2], c[Nx*Ny+2]] )
	idx = 3
	for m = 3:Ny 
		Ty0, Ty1 = Ty1, 2 * y * Ty1 - Ty0
		f += Tx0 * Ty1 * [c[idx], c[Nx*Ny+idx]] 
		idx += 1
	end 
	
	Tx1 = x;
	Ty0 = 1.0; Ty1 = y
	f += ( Tx1*Ty0 * [c[idx], c[Nx*Ny+idx]]
		 + Tx1*Ty1 * [c[idx+1], c[Nx*Ny+idx+1]] )
    idx += 2 
	for m = 3:Ny
		Ty0, Ty1 = Ty1, 2 * y * Ty1 - Ty0
		f += Tx1 * Ty1 * [c[idx], c[Nx*Ny+idx]]
		idx += 1
	end
	
	
	for n = 3:Nx
		Tx0, Tx1 = Tx1, 2 * x * Tx1 - Tx0
		
		Ty0 = 1.0; Ty1 = y
		f += ( Tx1*Ty0 * [c[idx], c[Nx*Ny+idx]]
			 + Tx1*Ty1 * [c[idx+1], c[Nx*Ny+idx+1]] )
		idx += 2 
		
		for m = 3:Ny
			Ty0, Ty1 = Ty1, 2 * y * Ty1 - Ty0
			f += Tx1 * Ty1 * [c[idx], c[Nx*Ny+idx]]
			idx += 1
		end 
	end
	
	return f
end

# ╔═╡ 8aad91ec-8f81-11eb-34cc-29d9b9a2ed32
loss = let params = params, train = train
	
	c0 = zeros(2 * params.Nx * params.Ny)

	# parameterised ODE model 
	F(u, p, t) = evalcheb22(u, p, params)
	prob(x0) = ODEProblem(F, x0, (0.0, params.Tf))
	soln(p, x0) = solve(prob(x0), Tsit5(), p = p, saveat = params.TT).u

	# the loss functional 
	_loss(p, cfg) = sum(abs2 ∘ norm, soln(p, cfg.x0) - cfg.X)
	loss(p) = sum(cfg -> _loss(p, cfg), train)

	# loss(c0)	
	# @time loss(c0)
	# @time loss(c0)
	# @time Zygote.gradient(loss, c0)
	# @time Zygote.gradient(loss, c0)
	# Zygote.hessian(loss, c0)
end


# ╔═╡ 53f64390-9046-11eb-0f68-233bb9f6f50b
# let loss = loss, α = 1e-7, params=params
# 	c = zeros(2 * params.Nx * params.Ny)
# 	grad_loss = p -> Zygote.gradient(loss, p)[1]
# 	size(grad_loss(c)), size(c)
# 	for n = 1:100 
# 		l = loss(c) 
# 		g = grad_loss(c)
# 		@show l, norm(g, Inf)
# 		c -= α * g 
# 	end		
# end

# ╔═╡ Cell order:
# ╠═2215d20c-9008-11eb-0ced-83e677e8816e
# ╟─3e3723c8-9008-11eb-11dc-d77df12db1d9
# ╟─3d9e50bc-9008-11eb-0080-1bd2057a9aed
# ╟─ad9a97b4-8f80-11eb-3513-5f22a6e9a821
# ╟─22f496b2-8e7e-11eb-3dca-59872173d433
# ╟─1d3a3d66-8e84-11eb-3b7a-4dff2ba4cad3
# ╟─0c50f854-8e85-11eb-06e4-3572b8017422
# ╟─f40b0f94-8f80-11eb-1364-abb59871cf32
# ╟─df50c834-8f7e-11eb-2b0b-79700e9f6f71
# ╠═28f8cfc0-8f81-11eb-33b4-05f3b55c14ca
# ╠═8aad91ec-8f81-11eb-34cc-29d9b9a2ed32
# ╠═53f64390-9046-11eb-0f68-233bb9f6f50b
