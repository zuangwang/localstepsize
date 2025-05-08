import Pkg
Pkg.add(["LIBSVMdata", "LinearAlgebra", "Plots", "LaTeXStrings", "TimerOutputs", "CSV", "DataFrames"])

using LinearAlgebra, Plots, LaTeXStrings, LIBSVMdata, Random, TimerOutputs, Base.Threads, Statistics, CSV, DataFrames


M, labels = load_dataset("w8a",
    dense=true,
    replace=false,
    verbose=true,
)

#change the labels
for i in eachindex(labels)
    if labels[i] == 1
        labels[i] = 0
    else
        labels[i] = 1
    end 
end

function estimate_lipschitz(M, labels)
    if ndims(M) == 1
        dim = size(M[1])[2]
        lipschitz_values = [[] for _ in 1:size(M)[1]]
        for i in 1:size(M)[1]
            for j in 1:size(M[i])[1]
                L = 0
                # for j = 1:idx    
                #     L_j = maximum(eigen(exp(-labels[i]*(M[i,:]'*randn(dim)))/(1+exp(-labels[i]*(M[i,:]'*randn(dim))))^2 .* (M[i,:]*M[i,:]')).values)
                #     if L_j > L
                #         L = L_j
                #     end
                # end
                if labels[i][j] != 0
                    L = maximum(eigen(0.25 .* (labels[i][j])^2 .* (M[i][j, :] * M[i][j, :]')).values)
                else
                    L = maximum(eigen(0.25 .* (M[i][j, :] * M[i][j, :]')).values)
                end
                push!(lipschitz_values[i], L)
            end
        end
    else
        dim = size(M)[2]
        lipschitz_values = []
        for j in 1:size(M)[1]
            L = 0
            # for j = 1:idx    
            #     L_j = maximum(eigen(exp(-labels[i]*(M[i,:]'*randn(dim)))/(1+exp(-labels[i]*(M[i,:]'*randn(dim))))^2 .* (M[i,:]*M[i,:]')).values)
            #     if L_j > L
            #         L = L_j
            #     end
            # end
            if labels[j] != 0
                L = maximum(eigen(0.25 .* (labels[j])^2 .* (M[j, :] * M[j, :]')).values)
            else
                L = maximum(eigen(0.25 .* (M[j, :] * M[j, :]')).values)
            end
            push!(lipschitz_values, L)
        end
    end

    return lipschitz_values
end

function sort_with_norms(M, labels)
    if ndims(M) == 1
        dim = size(M[1])[2]
        norms = [[] for _ in 1:size(M)[1]]
        for i in 1:size(M)[1]
            for j in 1:size(M[i])[1]
                push!(norms[i], M[i][j, :]' * M[i][j, :])
            end
        end
    else
        dim = size(M)[2]
        norms = []
        for j in 1:size(M)[1]
            push!(norms, M[j, :]' * M[j, :])
        end
    end

    return norms
end

#lipschitz_values = estimate_lipschitz(M, labels)
norms = sort_with_norms(M, labels)
matrix_saved = hcat(M, labels, norms)
row_sum = M*ones(size(M)[2])
function are_elements_same_with_tolerance(v::Vector{T}, tol::T) where T
    return all(x -> abs(x - v[1]) <= tol, v)
end
tolerance = 1e-5
println(are_elements_same_with_tolerance(row_sum, tolerance))

df = DataFrame(matrix_saved, :auto)
# Get the full path of the current script
script_path = @__DIR__

# Get the directory containing the script
script_dir = dirname(script_path)
path = joinpath(script_dir, "sorted_datasets", "w8a_norms.csv")

CSV.write(path, df)