using LinearAlgebra
using Plots
using LaTeXStrings
using Statistics, CSV, DataFrames, Random, Dates

#algorithm NIDS
function NIDS(X2, X1, G2, G1, mat, μs, Ls, storage)
    stepsize = Array{Float64}(undef, number_of_agents)
    for i = 1:number_of_agents
        stepsize[i] = 1 / Ls[i]
    end
    (N, ~) = size(mat)
    c = 1 / (1 - minimum(eigvals(mat))) / maximum(stepsize)
    tildemat = Matrix(I, N, N) .- c * diagm(stepsize) * (Matrix(I, N, N) .- mat)
    Xhalf = 2 .* X2 .- X1 .- G2 * diagm(stepsize) .+ G1 * diagm(stepsize)
    X3 = consensus(tildemat, Xhalf)
    return X3
end
function NIDS_same(X2, X1, G2, G1, mat, μs, Ls, storage)
    stepsize = Array{Float64}(undef, number_of_agents)
    L_average = sum(Ls)/number_of_agents
    for i = 1:number_of_agents
        stepsize[i] = 1 / L_average
    end
    (N, ~) = size(mat)
    c = 1 / (1 - minimum(eigvals(mat))) / maximum(stepsize)
    tildemat = Matrix(I, N, N) .- c * diagm(stepsize) * (Matrix(I, N, N) .- mat)
    Xhalf = 2 .* X2 .- X1 .- G2 * diagm(stepsize) .+ G1 * diagm(stepsize)
    X3 = consensus(tildemat, Xhalf)
    return X3
end

function generate_X1(X0, G0, μs, Ls, storage)
    stepsize = Array{Float64}(undef, number_of_agents)
    for i = 1:number_of_agents
        stepsize[i] = 1 / Ls[i]
    end
    X1 = X0 .- G0 * diagm(stepsize)
    return X1
end

function NEW_NIDS(X2, X1, G2, G1, mat, μs, Ls, storage)
    stepsize = Array{Float64}(undef, number_of_agents)
    for i = 1:number_of_agents
        stepsize[i] = 1 / Ls[i]
    end
    (N, ~) = size(mat)
    tildemat = (Matrix(I, N, N) .+ mat) ./ 2
    Xhalf = 2 .* X2 .- X1 .- G2 * diagm(stepsize) .+ G1 * diagm(stepsize)
    X3 = consensus(tildemat, Xhalf)
    return X3
end

function COMBINATION(X2, X1, G2, G1, mat, μs, Ls, storage)
    if switch_sign == 1 && initial_sign == 0
        X3 = generate_X1(X2, G2, μs, Ls, storage)
        global initial_sign = 1
    elseif switch_sign == 1 && initial_sign == 1
        X3 = NIDS(X2, X1, G2, G1, mat, μs, Ls, storage)
    else
        X3 = NEW_NIDS(X2, X1, G2, G1, mat, μs, Ls, storage)
    end
    # if abs(norm(gradient(storage[end], A, number_of_agents, y)*ones(number_of_agents))-norm(gradient(storage[end-1], A, number_of_agents, y)*ones(number_of_agents)))<= 10^-10 && initial_sign == 0 #switch criteria
    #     global switch_sign = 1
    # end
    if switch(storage)
        global switch_sign = 1
    end
    return X3
end

function switch(storage)
    norms = []
    for i = 1:size(storage)[1]
        push!(norms, norm(storage[i]*ones(size(storage[1])[2])))
    end
    norms = log10.(norms)
    if size(norms)[1] >= 75 && (norms[end-1] - norms[end])<= (norms[end-4] - norms[end])/5 + 10^-1
        return true
    else
        return false
    end
end

function GD1(X1, G1, mat, μs, Ls, storage)
    L_sum = sum(Ls)
    number_of_agents = size(mat)[1]
    stepsize = ones(number_of_agents) .* (1 / (L_sum / number_of_agents))
    out = consensus(mat, X1 .- G1 * diagm(stepsize))
    return out
end

function GD1(X2, X1, G2, G1, mat, μs, Ls, storage)
    L_sum = sum(Ls)
    number_of_agents = size(mat)[1]
    stepsize = ones(number_of_agents) .* (1 / (L_sum / number_of_agents))
    out = consensus(mat, X2 .- G2 * diagm(stepsize))
    return out
end

function consensus(W, x)
    (a, b) = size(x)
    (N, ~) = size(W)
    if N > 1
        for i in 1:N
            sum_i = 0
            for j in 1:N
                sum_i = sum_i .+ W[i, j] .* x[:, j]
            end
            if i == 1
                y = sum_i
            else
                y = [y sum_i]
            end
        end
    else
        y = x
    end
    return y
end

function random_ball(dim, degree)
    y = randn(dim)  # Sample from normal distribution
    y = y / norm(y) # Normalize to unit sphere
    # r = rand()^(1/dim) # Scale by random radius for uniformity
    return y*degree
end

# Example usage
# dim = 3  # Dimension of the space
# y = random_unit_ball(dim)
# println(y)
# println(norm(y)) # Should be <= 1


#generate data matrix M_i and y_i with default setting x^* 0
function generate_dataMatrix(dimension_of_data, number_of_data, μs, Ls, number_of_agents)
    A = Array{Matrix}(undef, number_of_agents)
    for i = 1:number_of_agents
        # Ensure a ≤ b
        @assert μs[i] ≤ Ls[i] "Minimum eigenvalue must be ≤ maximum eigenvalue"

        # Create an orthonormal basis using QR decomposition
        U, _ = qr(randn(dimension_of_data, dimension_of_data))  # Random orthogonal m×m matrix
        V, _ = qr(randn(number_of_data, number_of_data))  # Random orthogonal n×n matrix

        # Define singular values such that (A'*A) has eigenvalues in [a, b]
        singular_values = sqrt.(range(μs[i], Ls[i], length=number_of_data))

        # Construct A using the SVD: A = U * Σ * V'
        Σ = Diagonal(singular_values)
        A[i] = U[:, 1:number_of_data] * Σ * V'
    end
    return A
end


function generate_dataLabels(xs_constraint, A, number_of_agents)
    y = Array{Any}(undef, number_of_agents)
    dim = size(A[1])[2]
    x = random_ball(dim, 0)
    if xs_constraint == 0
        for i = 1:number_of_agents
            y[i] = pinv(A[i]') * A[i]' * A[i] * random_ball(dim, 10^-5) #different local optimal point
            #y[i] = pinv(A[i]') * A[i]' * A[i] * x #same local optimal point 
        end
    end
    return y
end

function value(state, A, number_of_agents, y, mu)
    dim = size(A[1])[2]
    f = Array{Float64}(undef, number_of_agents)
    for i = 1:number_of_agents
        f[i] = norm(A[i] * state[:, i] .- y[i])./2
    end
    return sum(f)
end
function gradient(state, A, number_of_agents, y, mu)
    dim = size(A[1])[2]
    G = Array{Float64}(undef, dim, number_of_agents)
    for i = 1:number_of_agents
        G[:, i] = A[i]' * (A[i] * state[:, i] .- y[i]) 
    end
    return G
end

function compute_Ls(A, number_of_agents, y, sample_number, mu)
    dim = size(A[1])[2]
    Ls = zeros(number_of_agents)
    for i = 1:number_of_agents
        Ls[i] = maximum(eigen(A[i]' * A[i]).values)
    end
    return Ls
end


## test Example usage
##using generated data
m, n = 100, 50  # A is 5×3
number_of_agents = 2
mu = 0.01
μs= mu .* ones(number_of_agents)  # Desired min/max eigenvalues of A'*A
# Ls = [1.2, 1.5, 1, 2.2, 3, 1.8]
Ls = [3 1/3]
#Ls = [1, 2, 3, 4, 5]
number_of_algorithms = 4
mat = ones(number_of_agents, number_of_agents)./number_of_agents

A = generate_dataMatrix(m, n, μs, Ls, number_of_agents)
y = generate_dataLabels(0, A, number_of_agents)


##using w8a

# number_of_agents = 2
# subset_size = 200
# mu = 0.0

# A = []
# y = []

# #load data
# # Get the full path of the current script
# script_path = @__FILE__

# # Get the directory containing the script
# script_dir = dirname(script_path)
# path = joinpath(script_dir, "w8a_norms.csv")

# data = CSV.read(path, DataFrame)
# mat = Matrix(data)
# M = mat[:, 1:end-2]
# labels = mat[:, end-1]
# norms = mat[:,end]

# #lipschitz_values = estimate_lipschitz(class_M)
# class_M = []
# class_y = []
# class_l = []

# indice_1 = findall(x -> x == 1, labels)
# push!(class_M, M[indice_1, :])
# push!(class_y, labels[indice_1])
# push!(class_l, norms[indice_1])

# indice_2 = findall(x -> x == -1, labels)
# push!(class_M, M[indice_2, :])
# push!(class_y, labels[indice_2])
# push!(class_l, norms[indice_1])

# # for i in 1:3 #random selection
# #     selected_rows = randperm(size(M,1))[1:subset_size]  # Random row indices
# #     println(labels[selected_rows,:])
# #     push!(A, M[selected_rows, :])    # Cut matrix
# #     push!(y, labels[selected_rows])  # Cut labels
# # end

# # for i in 1:1
# #     # selected_rows = sortperm((lipschitz_values[1]))[1:subset_size]  # select rows based on lipschitz_values
# #     selected_rows = findall(x -> 0.2<x<0.8, lipschitz_values[1])
# #     final_selected_rows = randperm(size(class_M[1][selected_rows, :], 1))[1:subset_size]
# #     push!(A, class_M[1][final_selected_rows, :])    # Cut matrix
# #     push!(y, class_y[1][final_selected_rows])  # Cut labels
# # end

# for i = 1:1
#     selected_rows = findall(x -> 2 < x < 5, norms)
#     final_selected_rows = randperm(size(M[selected_rows, :], 1))[1:subset_size]
#     push!(A, M[selected_rows, :][final_selected_rows, :])    # Cut matrix
#     push!(y, labels[selected_rows][final_selected_rows])  # Cut labels
# end

# for i = 2:2
#     selected_rows = findall(x -> 2 < x < 5, norms)
#     final_selected_rows = randperm(size(M[selected_rows, :], 1))[1:subset_size]
#     push!(A, M[selected_rows, :][final_selected_rows, :])    # Cut matrix
#     push!(y, labels[selected_rows][final_selected_rows])  # Cut labels
# end

# # for i in 2:2
# #     # selected_rows = sortperm((lipschitz_values[2]), rev = true)[1:subset_size]  # Random row indices
# #     selected_rows = findall(x -> 1.5<x<2.5, lipschitz_values[2])
# #     final_selected_rows = randperm(size(class_M[1][selected_rows, :], 1))[1:subset_size]
# #     push!(A, class_M[2][final_selected_rows, :])    # Cut matrix
# #     push!(y, class_y[2][final_selected_rows])  # Cut labels
# # end


sample_number = 1
Ls = compute_Ls(A, number_of_agents, y, sample_number, mu)
μs = ones(number_of_agents).*mu
println(Ls)



number_of_algorithms = 4
mat = ones(number_of_agents, number_of_agents) ./ number_of_agents
iteration = 200
run_times = 1
algorithm_list = [GD1, NIDS, NEW_NIDS, COMBINATION]
dim = size(A[1])[2]

# results = []
# for _ in 1:run_times
#     global switch_sign = 0
#     global initial_sign = 0
#     global X = [[] for _ in 1:number_of_algorithms]  # A vector of 5 empty integer arrays
#     global plot_matrix = zeros(iteration+1, number_of_algorithms)

#     for i = 1:iteration+1              
#         if i == 1
#             local vec = random_ball(dim, 1)
#             for j = 1:number_of_algorithms
#                 push!(X[j], (vec)*ones(number_of_agents)')
#             end
#             for j = 1:number_of_algorithms
#                 plot_matrix[i, j] = norm(gradient(X[j][i], A, number_of_agents, y)*ones(number_of_agents))
#             end
#         elseif i == 2
#             push!(X[1], GD1(X[1][1], gradient(X[1][1], A, number_of_agents, y), mat, μs, Ls, X[1]))
#             push!(X[2], generate_X1(X[2][1], gradient(X[2][1], A, number_of_agents, y), μs, Ls, X[2]))
#             push!(X[3], generate_X1(X[3][1], gradient(X[3][1], A, number_of_agents, y), μs, Ls, X[3]))
#             push!(X[4], generate_X1(X[4][1], gradient(X[4][1], A, number_of_agents, y), μs, Ls, X[4]))
#             for j = 1:number_of_algorithms
#                 plot_matrix[i, j] = norm(gradient(X[j][i], A, number_of_agents, y)*ones(number_of_agents))
#             end
#         else
#             for j = 1:number_of_algorithms
#                 push!(X[j], algorithm_list[j](X[j][i-1], X[j][i-2], gradient(X[j][i-1], A, number_of_agents, y), gradient(X[j][i-2], A, number_of_agents, y), mat, μs, Ls, X[j]))
#                 plot_matrix[i, j] = norm(gradient(X[j][i], A, number_of_agents, y)*ones(number_of_agents))
#             end
#         end
#     end
#     push!(results, plot_matrix)
# end

results = []
for _ in 1:run_times
    global switch_sign = 0
    global initial_sign = 0
    global X = [[] for _ in 1:number_of_algorithms]  # A vector of 5 empty integer arrays
    global G = [[] for _ in 1:number_of_algorithms]
    global plot_matrix = zeros(iteration + 1, number_of_algorithms)
    for i = 1:iteration+1
        if i == 1
            vec = random_ball(dim, 1) .* 1
            for j = 1:number_of_algorithms
                push!(X[j], vec .* ones(number_of_agents)')
            end
            for j = 1:number_of_algorithms
                push!(G[j], gradient(X[j][i], A, number_of_agents, y, mu))
                plot_matrix[i, j] = norm(G[j][i]* ones(number_of_agents) ./ number_of_agents)
            end
        elseif i == 2
            push!(X[1], GD1(X[1][1], G[1][1], mat, μs, Ls, G[1]))
            push!(G[1], gradient(X[1][2], A, number_of_agents, y, mu))

            push!(X[2], generate_X1(X[2][1], G[2][1], μs, Ls, G[2]))
            push!(G[2], gradient(X[2][2], A, number_of_agents, y, mu))

            push!(X[3], generate_X1(X[3][1], G[3][1], μs, Ls, G[3]))
            push!(G[3], gradient(X[3][2], A, number_of_agents, y, mu))

            push!(X[4], generate_X1(X[4][1], G[4][1], μs, Ls, G[4]))
            push!(G[4], gradient(X[4][2], A, number_of_agents, y, mu))
            for j = 1:number_of_algorithms
                plot_matrix[i, j] = norm(gradient(X[j][i], A, number_of_agents, y, mu) * ones(number_of_agents) ./ number_of_agents)
                # plot_matrix[i, j] = value(X[j][i], A, number_of_agents, y, mu)
            end
        else
            #@timeit to "algorithm_update" 
            #@timeit to "compute gradient" 
            for j = 1:number_of_algorithms
                push!(X[j], algorithm_list[j](X[j][i-1], X[j][i-2], G[j][i-1], G[j][i-2], mat, μs, Ls, G[j]))
                push!(G[j], gradient(X[j][i], A, number_of_agents, y, mu))

                plot_matrix[i, j] = norm(G[j][i] * ones(number_of_agents) ./ number_of_agents)
                # plot_matrix[i, j] = value(X[j][i], A, number_of_agents, y, mu)
            end
        end
    end
    push!(results, plot_matrix)
    println(size(results)[1])
end

means = zeros(iteration+1, number_of_algorithms)
std_errors = zeros(iteration+1, number_of_algorithms)

for i in 1:iteration+1
    # Extract the i-th element from each of the 10 vectors
    for j in 1:number_of_algorithms
        values_at_position = [results[a][i, j] for a in 1:run_times]
        means[i, j] = mean(values_at_position)
        std_errors[i, j] = std(values_at_position) / sqrt(run_times)  # Standard error
    end
end

# means
# fig = plot()
# plot!(fig, 1:size(means,1), means[:,1], 
# yerr=std_errors[:,1], 
# labels=L"GD", linecolor=:red, line=(2, :solid),markersize=1, markerstrokecolor=:auto)
# plot!(fig, 1:size(means,1), means[:,2], 
# yerr=std_errors[:,2], 
# labels=L"NIDS", linecolor=:black, line=(2, :solid),markersize=1, markerstrokecolor=:auto)
# plot!(fig, 1:size(means,1), means[:,3], 
# yerr=std_errors[:,3], 
# labels=L"NIDS-(I+W)/2", linecolor=:blue, line=(2, :solid),markersize=1, markerstrokecolor=:auto)
# plot!(fig, 1:size(means,1), means[:,4], 
# yerr=std_errors[:,4], 
# labels=L"COMBINATION", linecolor=:black, line=(2, :dash),markersize=1, markerstrokecolor=:auto)
# plot!(xlabel="Iteration", ylabel="Gradient", yscale=:log10, minorgrid=true)

# savefig(fig, "result.svg")
##Check eigenvalues of A'*A
# eigvals_AAtA = eigvals(A[2]' * A[2])
# println("Constructed Matrix A:\n", A[2])
# println("\nEigenvalues of A'*A:\n", eigvals_AAtA)
# println("\nMinimum eigenvalue: ", minimum(eigvals_AAtA))
# println("Maximum eigenvalue: ", maximum(eigvals_AAtA))

means
error_bar_index = 1:5:size(means,1)
index = 1:size(means,1)
fig = plot()
plot!(fig, index[error_bar_index], means[:,1][error_bar_index], 
# yerr=std_errors[error_bar_index],
labels=L"GD", linecolor=:red, line=(2, :solid),markersize=1, markerstrokecolor=:auto)

plot!(fig, index[error_bar_index], means[:,2][error_bar_index], 
# yerr=std_errors[error_bar_index],
labels=L"NIDS", linecolor=:black, line=(1, :solid),markersize=1, markerstrokecolor=:auto)

plot!(fig, index[error_bar_index], means[:,3][error_bar_index], 
# yerr=std_errors[error_bar_index],
labels=L"NIDS-(I+W)/2", linecolor=:blue, line=(2, :solid),markersize=1, markerstrokecolor=:auto)

plot!(fig, index[error_bar_index], means[:,4][error_bar_index], 
# yerr=std_errors[error_bar_index],
labels=L"COMBINATION", linecolor=:black, line=(2, :dash),markersize=1, markerstrokecolor=:auto)
plot!(xlabel="Iteration", ylabel="Gradient", yscale=:log10, minorgrid=true)

script_path = @__DIR__

# Get the directory containing the script
script_dir = dirname(script_path)
timestamp = Dates.format(now(), "YYYYmmdd-HHMMSS")
savepath = joinpath(script_dir, "numerical_experiments", "results", "run_$timestamp")
mkdir(savepath)
# for i in 1:number_of_agents
#     data_path = joinpath(savepath, "subset_$i.csv")
#     matrix_saved = hcat(A[i], y[i])
#     CSV.write(data_path, DataFrame(matrix_saved, :auto))
# end
for i in 1:size(results)[1]
    data_path = joinpath(savepath, "result_$i.csv")
    CSV.write(data_path, DataFrame(results[i], :auto))
end

CSV.write(joinpath(savepath, "means.csv"), DataFrame(means, :auto))
CSV.write(joinpath(savepath, "std_errors.csv"), DataFrame(std_errors, :auto))

fig_path = joinpath(savepath, "pic.svg")
savefig(fig, fig_path)