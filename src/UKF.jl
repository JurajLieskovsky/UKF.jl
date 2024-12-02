module UKF

using LinearAlgebra
using Debugger
using Infiltrator
using BenchmarkTools

struct SigmaPoint
    val::AbstractArray
    w_μ::Number
    w_Σ::Number
end

function weights(n, λ, α, β)
    # mean
    w_μi = 1 / (2 * (n + λ))
    w_μ0 = λ / (n + λ)

    # covariance
    w_Σi = w_μi
    w_Σ0 = w_μ0 + 1 - α^2 + β

    # concatenated
    w_μ = vcat(w_μ0, w_μi * ones(2 * n))
    w_Σ = vcat(w_Σ0, w_Σi * ones(2 * n))

    return w_μ, w_Σ
end

# Additive unscented transformation, predicion, and update
include("additive.jl")

# General unscented transformation, predicion, and update
include("augmented.jl")

end # module UKF
