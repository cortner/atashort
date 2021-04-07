
using Plots, LaTeXStrings, PrettyTables, DataFrames, LinearAlgebra,
		  PlutoUI, BenchmarkTools, ForwardDiff, Printf

import SIAMFANLEquations

function ata_table(data::AbstractMatrix, labels::AbstractVector;
							format = :md, kwargs...)
	if format == :md
		return pretty_table(String, data, labels;
					 backend=:text,
					 tf = tf_markdown,
					 kwargs...) |> Markdown.parse
	elseif format == :html
		return pretty_table(String, data, labels;
					 backend=:html, tf=tf_html_minimalist,
					 nosubheader=true,
					 kwargs...) |> HTML
	else
		error("unknown table format")
	end
end

function ata_table(args...; T = Float64, format = :md, kwargs...)

	labels = []
	data = Matrix{T}(undef, length(args[1][1]), 0)
	formatters = []
	for (iarg, arg) in enumerate(args)
		data = [data arg[1]]
		if length(arg) > 1
			push!(labels, arg[2])
		else
			push!(labels, "_")
		end
		if length(arg) > 2
			# push a formatter
			push!(formatters, ft_printf(arg[3], [iarg]))
		end
	end
	if format == :md
		return pretty_table(String, data, labels;
					 backend=:text,
					 tf = tf_markdown,
				     formatters = tuple(formatters...),
					 kwargs...) |> Markdown.parse
	elseif format == :html
		return pretty_table(String, data, labels;
					 backend=:html, tf=tf_html_minimalist,
					 nosubheader=true,
    				 formatters = formatters,
					 kwargs...) |> HTML
	else
		error("unknown table format")
	end
end

function SIAMFANLEquations.nsoli(f, x0; kwargs...)
	f!(FV, x) = (FV[:] .= f(x); return nothing)
	FS = similar(x0)
	FPS = similar(x0, (length(x0), length(x0)))
	result = nsoli(f!, x0, FS, FPS; kwargs...)
	return result.solution
end


function chebbasis(x, N, a=-1, b=1)
	x = (x - a) / (b - a) - (x - b) / (a - b)
	T = zeros(N)
	T[1] = 1.0
	T[2] = x
	for n = 3:N
		T[n] = 2 * x * T[n-1] - T[n-2]
	end
	return T
end

"""
Barycentric interpolation with a Chebyshev grid with N grid points.
The interpolant is evaluated at points `x`.
"""
function chebbary(x, F::Vector)
    N = length(F)-1
	 X = [ cos(j*π/N) for j = N:-1:0 ]
    p = 0.5 * ( F[1] ./ (x .- X[1]) + (-1)^N * F[N+1] ./(x .- X[N+1]) )
    q = 0.5 * (1.0 ./ (x .- X[1]) + (-1)^N ./ (x .- X[N+1]))
    for n = 1:N-1
        p += (-1)^n * F[n+1] ./ (x .- X[n+1])
        q += (-1)^n ./ (x .- X[n+1])
    end
    return p ./ q
end

"""
generate a grid on which to plot errors; this is chosen to avoid
any grid points since barycentric interpolation is not defined
on those.
"""
errgrid(Np) = range(-1+0.000123, stop=1-0.000321, length=Np)


function chebtransf(F)

end

@info("Finished loading dependencies")
