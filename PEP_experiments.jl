include("PEP_SDP.jl")

#number of iterations performed
K = 20


#number of agents 
N = 4

#mixing matrix 
#all_to_all
# mat = ones(N, N) ./ N

##generate a ring mixing matrix
# A = zeros(Int, N, N)
# for i in 1:N
#     A[i, mod1(i + 1, N)] = 1
#     A[i, mod1(i - 1, N)] = 1
# end
# mat = (A .+ Matrix(I, N, N)) ./ N

##generate a star mixing matrix
# A = zeros(Int, N, N)  # initialize an n x n adjacency matrix

# for i in 2:N
#     A[1, i] = 1
#     A[i, 1] = 1
# end

# mat = (A .+ (N-1) .* Matrix(I, N, N)) ./ N
# mat[1, 1] = 1/N

##geenrate a random mixing matrix
# represents the probability that any given edge exists between a pair of distinct nodes (connectivity)
p = 0.6
g = generate_random_sparse_graph(N, p)
mat = generate_metropolis_mixing_matrix(g)

# mat = [0.5 0 0.25 0.25; 0 0.5 0.25 0.25; 0.25 0.25 0.25 0.25; 0.25 0.25 0.25 0.25]

#initialize local function prarameters
fctParams = Vector{fctParam}(undef, N)

#fctParam(strongly_convex_parameter, Lipschitz_smoothness_parameter, function_class_description)
fctParams[1] = fctParam(0.1, 0.5, "SmoothStronglyConvex")
fctParams[2] = fctParam(0.1, 2, "SmoothStronglyConvex")
# fctParams[3] = fctParam(0.1, 0.5, "SmoothStronglyConvex")
# fctParams[4] = fctParam(0.1, 2, "SmoothStronglyConvex")

#hand tuned switch time for algorithm 1
switch_time = 10

#initial value constraints
R_0 = 100

#optimal vlaue constraints
R_optimal = 1


#run the experiments
#plot_save_pic(run(mat, K, fctParams, R_optimal, R_0, "funcValue", switch_time), fctParams)