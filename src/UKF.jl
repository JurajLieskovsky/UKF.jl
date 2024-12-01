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

function inverse_unscented_transform(sigma_points)
    μ = mapreduce(pt -> pt.val * pt.w_μ, +, sigma_points)
    Σ = mapreduce(pt -> (pt.val - μ) * pt.w_Σ * (pt.val - μ)', +, sigma_points)
    return μ, Σ
end


function predict(f, μ, P, u, Q)
    # seed sigma points
    sigma_points = unscented_transform(μ, P)

    # propagate sigma points through dynamics
    map(x -> x.val .= f(x.val, u), sigma_points) 

    # calculate new mean and covariance
    new_μ, new_P = inverse_unscented_transform(sigma_points)

    return new_μ, 0.5 * new_P * new_P' + Q
end

function update(h, μ, P, z, R)
    # constants
    n = length(μ)
    λ = 2
    α = 1
    β = 0

    # auxilary sigma points
    S = cholesky(P).L
    positive_states = map(s -> μ + sqrt(n + λ) * s, eachcol(S))
    negative_states = map(s -> μ - sqrt(n + λ) * s, eachcol(S))
    auxilary_states = vcat(positive_states, negative_states)

    # measurements
    central_meas = h(μ)
    auxilary_meas = map(x -> h(x), auxilary_states)

    # mean measurement
    w_0 = λ / (n + λ)
    w_i = 1 / (2 * (n + λ))

    μ_meas = w_0 * central_meas
    μ_meas += mapreduce(x -> w_i * x, +, auxilary_meas)

    # (in-place) deviations from the mean
    ## measurements
    central_meas .= central_meas - μ_meas
    map(x -> x .= x - μ_meas, auxilary_meas)

    ## states
    central_state = zeros(eltype(μ), size(μ))
    map(x -> x .= x - μ, auxilary_states)

    # covariances
    ## measurement - measurement
    cov_yy = central_meas * w_0 * central_meas'
    cov_yy .+= mapreduce(dy -> w_i * dy * dy', +, auxilary_meas)
    cov_yy .+= R

    # state - measurement
    cov_xy = central_state * w_0 * central_meas'
    cov_xy .+= mapreduce((dx, dy) -> w_i * dx * dy', +, auxilary_states, auxilary_meas)

    # Kalman gain
    K = cov_xy * inv(cov_yy)

    # Updated state estimate
    updt_μ = μ - K * (μ_meas - z)
    updt_P = P - K * cov_xy'

    return updt_μ, 0.5 * updt_P * updt_P'
end

end # module UKF
