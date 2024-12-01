module UKF

using LinearAlgebra

function predict(f, μ, P, u, Q)
    # constants
    n = length(μ)
    λ = 3 - n

    # auxilary sigma points
    S = cholesky(P).L
    positive = map(s -> μ + sqrt(n + λ) * s, eachcol(S))
    negative = map(s -> μ - sqrt(n + λ) * s, eachcol(S))
    auxilary = vcat(positive, negative)

    # state propagation
    central = f(μ, u)
    map(x -> x .= f(x, u), auxilary)

    # new mean
    w_0 = λ / (n + λ)
    w_i = 1 / (2 * (n + λ))

    new_μ = w_0 * central
    new_μ += mapreduce(x -> w_i * x, +, auxilary)

    # (in-place) deviations from the new mean
    central .= central - new_μ
    map(x -> x .= x - new_μ, auxilary)

    # new state error covariance
    new_P = central * central'
    new_P += mapreduce(d -> d * d', +, auxilary)
    new_P += Q

    return new_μ, new_P
end

function update(h, μ, P, z, R)
    # constants
    n = length(μ)
    λ = 3 - n

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
    cov_yy = central_meas * central_meas'
    cov_yy += mapreduce(dy -> dy * dy', +, auxilary_meas)
    cov_yy += R

    # state - measurement
    cov_xy = central_state * central_meas'
    cov_xy += mapreduce((dx,dy) -> dx * dy', +, auxilary_states, auxilary_meas)

    # Kalman gain
    K = cov_xy * inv(cov_yy)

    # Updated state estimate
    updt_μ = μ - K * (μ_meas - z)
    updt_P = P - K * cov_yy * K'

    return updt_μ, updt_P
end

end # module UKF
