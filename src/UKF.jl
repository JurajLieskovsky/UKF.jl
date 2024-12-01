module UKF

using LinearAlgebra
using Debugger
using Infiltrator

function predict(f, μ, P, u, Q)
    # constants
    n = length(μ)
    λ = 2
    α = 1
    β = 0

    # unscented transform
    S = cholesky(P).L

    positive = map(s -> μ + sqrt(n + λ) * s, eachcol(S))
    negative = map(s -> μ - sqrt(n + λ) * s, eachcol(S))

    points = hcat(μ, positive..., negative...)

    w_μ = 1 / (2 * (n + λ)) * ones(2 * n + 1)
    w_μ[1] = λ / (n + λ)

    w_Σ = diagm(w_μ)
    w_Σ[1,1] += 1 - α^2 + β

    # state propagation
    map(x -> x .= f(x, u), eachcol(points))

    # inverse unscented transform
    new_μ = points * w_μ 

    deviations = mapreduce(x -> x - new_μ, hcat, eachcol(points))
    new_P = deviations * w_Σ * deviations' + Q

    # @infiltrate
    return new_μ, 0.5 * new_P * new_P'
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
    cov_xy .+= mapreduce((dx,dy) -> w_i * dx * dy', +, auxilary_states, auxilary_meas)

    # Kalman gain
    K = cov_xy * inv(cov_yy)

    # Updated state estimate
    updt_μ = μ - K * (μ_meas - z)
    updt_P = P - K * cov_xy'

    return updt_μ, 0.5 * updt_P * updt_P'
end

end # module UKF
