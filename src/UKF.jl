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

function unscented_transform(μ, Σ, λ=2, α=1, β=0)
    n = length(μ)

    # weights
    w_μi = 1 / (2 * (n + λ))
    w_μ0 = λ / (n + λ)
    w_Σi = w_μ0
    w_Σ0 = w_μi + 1 - α^2 + β

    # points
    central = SigmaPoint(copy(μ), w_μ0, w_Σ0)

    S = cholesky(Σ).L
    positive = map(s -> SigmaPoint(μ + sqrt(n + λ) * s, w_μi, w_Σi), eachcol(S))
    negative = map(s -> SigmaPoint(μ - sqrt(n + λ) * s, w_μi, w_Σi), eachcol(S))

    return vcat([central], positive, negative)
end

function predict(f, μ, P, u, Q)
    # seed sigma points
    sigma_points = unscented_transform(μ, P)

    # propagate sigma points through dynamics
    map(x -> x.val .= f(x.val, u), sigma_points)

    # calculate new mean and covariance
    μn = mapreduce(x -> x.val * x.w_μ, +, sigma_points)
    Σn = mapreduce(x -> (x.val - μn) * x.w_Σ * (x.val - μn)', +, sigma_points)

    return μn, 0.5 * Σn * Σn' + Q
end

function update(h, μ, P, z, R)
    # seed sigma points
    sigma_states = unscented_transform(μ, P)

    # propagate sigma points through measurements
    sigma_meas = map(x -> SigmaPoint(h(x.val), x.w_μ, x.w_Σ), sigma_states)

    # calculate new mean and covariances
    μ_meas = mapreduce(y -> y.val * y.w_μ, +, sigma_meas)

    Σ_yy = mapreduce(y -> (y.val - μ_meas) * y.w_Σ * (y.val - μ_meas)', +, sigma_meas) + R
    Σ_xy = mapreduce((x, y) -> (x.val - μ) * y.w_Σ * (y.val - μ_meas)', +, sigma_states, sigma_meas)

    # Kalman gain
    K = Σ_xy * inv(Σ_yy)

    # Updated state estimate
    μn = μ - K * (μ_meas - z)
    Σn = P - K * Σ_xy'

    return μn, 0.5 * Σn * Σn'
end

end # module UKF
