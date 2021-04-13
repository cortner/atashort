### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 3d5f2b96-7d32-11eb-33bb-79a7f62f8da2
using LinearAlgebra, Printf, Plots

# ╔═╡ 04b964c4-7d31-11eb-14d8-8b286343ff4f
begin

	function wlsq(A, y, w)
		W = Diagonal(sqrt.(w))
		return qr(W * A) \ (W * y)
	end

	function irlsq(A, y; tol=1e-5, maxnit = 100, γ = 1.0, γmin = 1e-6, verbose=true)
		M, N = size(A)
		@assert M == length(y)
		wold = w = ones(M) / M
		res = 1e300
		x = zeros(N)
		verbose  && @printf("  n   | ||f-p||_inf |  extrema(w) \n")
		verbose  && @printf("------|-------------|---------------------\n")
		for nit = 1:maxnit
			x = wlsq(A, y, w)

			resnew = norm(y - A * x, Inf)
			verbose  && @printf(" %4d |   %.2e  |  %.2e  %.2e \n", nit, resnew, extrema(w)...)

			# update
			wold = w
			res = resnew
			wnew = w .* (abs.(y - A * x).^γ .+ 1e-15)
			wnew /= sum(wnew)
			w = wnew
		end
		return x, w, res
	end

	function cheb_basis(x::T, N) where {T}
		B = zeros(T, N+1)
		B[1] = 1.0
		B[2] = x
		for k = 2:N
			B[k+1] = 2 * x * B[k] - B[k-1]
		end
		return B
	end

	eval_chebpoly(F̃, x) = dot(F̃, cheb_basis(x, length(F̃)-1))

	function fit_cheb(f::Function, indomain::Function, N, Ndat; kwargs...)
		X = filter(indomain, range(-1, 1, length = Ndat))
		y = f.(X)
		A = zeros(length(X), N+1)
		for (irow, x) in enumerate(X)
			A[irow, :] = cheb_basis(x, N)
		end
		F̃, w, res = irlsq(A, y; kwargs...)
		return F̃
	end

end;


# ╔═╡ 4cfc8e54-7d32-11eb-12c5-5187ffd86e40
begin
	β = 30.0
	fβ(x) = 1 / (1 + exp(β * x))
	a = 0.2
	b = 0.03
	indomain(x) = (x < -a) || (x > a) || (-2*b < x < -b)
	F̃ = fit_cheb(fβ, indomain, 60, 10_000)
end

# ╔═╡ 2623664e-7d33-11eb-26a4-5bd7b0f45e32
begin
	xp = range(-1,1,length=300)
	plot(xp, fβ.(xp))
	plot!(xp, eval_chebpoly.(Ref(F̃), xp))
end



# ╔═╡ 069f7b18-7d48-11eb-0572-a39d1001eb3d
begin
	plot([-1, -a, NaN, -2*b, -b, NaN, a, 1], zeros(8), c = :red, lw=3, label = "domain")
	plot!(xp, abs.(fβ.(xp) - eval_chebpoly.(Ref(F̃), xp)), c = 1, label = "error",
			ylims = [-0.0001, 0.001])

end

# ╔═╡ 2be9e652-7d47-11eb-11e8-996d253a84b1
	function get_error(indomain, β, N, Ndat = 10 * N^2; maxnit = 100)
		gβ(x) = 1 / (1 + exp(β * x))
		F̃ = fit_cheb(gβ, indomain, N, Ndat, maxnit = maxnit)
		xtest = filter(indomain, range(-1.0, 1.0, length = ceil(Int, 1.33 * Ndat)))
		return norm( gβ.(xtest) - eval_chebpoly.(Ref(F̃), xtest), Inf )
	end

# ╔═╡ 3ec25bcc-7d33-11eb-2f0d-6543e236bb4e
begin
	β1 = 20.0
	a1 = 0.2
	indomain1(x) = (a1 <= abs(x) <= 1)
	NN = 5:5:50
	err = [ get_error(indomain1, β1, N; maxnit = 150) for N in NN ]
	plot(NN, err, yscale = :log10)
	plot!(NN, exp.( - a1 * NN), lw=2, c=:black, ls = :dash, label = "")
	# plot!(NN, exp.( - sqrt(a1) * sqrt.(NN)), lw=1, c=:black, ls = :dash, label = "")
end


# ╔═╡ 5b9ff01e-7d45-11eb-16f0-f1f9f8ddc543
begin
	# Case 2: mini-interval close to fermi-level
	β2 = 30.0
	a2 = 0.2
	b2 = 0.03
	indomain2(x) = (a2 <= abs(x) <= 1) || (-2*b2 <= x <= -b2)
	NN2 = 20:20:200
	err2 = [ get_error(indomain2, β2, N; maxnit = 150) for N in NN2 ]
	plot(NN2, err2, yscale = :log10)
	plot!(NN2, exp.( - a2 * NN2), lw=2, c=:black, ls = :dash, label = "exp(- 0.5 ctsgap * N)")
	plot!(NN2, exp.( - 1/β2 * NN2), lw=1, c=:black, ls = :dot, label = "exp( - 1/β N )")
	plot!(NN2, exp.( - b2 * NN2), lw=1, c=:black, ls = :dot, label = "exp( - discrgap N )")
	# plot!(NN2, exp.( - sqrt(a2) * sqrt.(NN2)), lw=1, c=:black, ls = :dash, label = "")
end



# ╔═╡ c13fd40e-7d48-11eb-3d2c-fb5f69ad59fb
begin
	# Case 3 : mini-interval centered on fermi-level
	β3 = 30.0
	a3 = 0.2
	b3 = 0.05
	indomain3(x) = (a3 <= abs(x) <= 1) || (-b3 <= x <= b3)
	NN3 = 5:5:50
	err3 = [ get_error(indomain3, β3, N; maxnit = 150) for N in NN3 ]
	plot(NN3, err3, yscale = :log10)
	plot!(NN3, exp.( - a3 * NN3), lw=2, c=:black, ls = :dash, label = "")
	plot!(NN3, exp.( - 1/β3 * NN3), lw=1, c=:black, ls = :dot, label = "")
	# plot!(NN2, exp.( - sqrt(a2) * sqrt.(NN2)), lw=1, c=:black, ls = :dash, label = "")
end



# ╔═╡ Cell order:
# ╠═3d5f2b96-7d32-11eb-33bb-79a7f62f8da2
# ╠═04b964c4-7d31-11eb-14d8-8b286343ff4f
# ╠═4cfc8e54-7d32-11eb-12c5-5187ffd86e40
# ╠═2623664e-7d33-11eb-26a4-5bd7b0f45e32
# ╠═069f7b18-7d48-11eb-0572-a39d1001eb3d
# ╠═2be9e652-7d47-11eb-11e8-996d253a84b1
# ╠═3ec25bcc-7d33-11eb-2f0d-6543e236bb4e
# ╠═5b9ff01e-7d45-11eb-16f0-f1f9f8ddc543
# ╠═c13fd40e-7d48-11eb-3d2c-fb5f69ad59fb
