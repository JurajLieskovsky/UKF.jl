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
    w_μi = 1 / (2 * (n + λ))
    w_μ0 = λ / (n + λ)
    w_Σi = w_μi
    w_Σ0 = w_μ0 + 1 - α^2 + β

    return w_μi, w_μ0, w_Σi, w_Σ0
end

# Additive unscented transformation, predicion, and update
include("additive.jl")

# General unscented transformation, predicion, and update
include("augmented.jl")

end # module UKF
