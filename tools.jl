
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

@info("Finished loading dependencies")
