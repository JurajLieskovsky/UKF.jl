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

    # state propagation
    x0 = f(μ, u)
    map(x -> x .= f(x, u), positive)
    map(x -> x .= f(x, u), negative)

    # new mean
    w_0 = λ / (n + λ)
    w_i = 1 / (2 * (n + λ))

    new_μ = w_0 * x0
    new_μ += mapreduce(x -> w_i * x, +, positive)
    new_μ += mapreduce(x -> w_i * x, +, negative)

    # (in-place) deviations from the new mean
    x0 .= x0 - new_μ
    map(x -> x .= x - new_μ, positive)
    map(x -> x .= x - new_μ, negative)

    # new state error covariance
    new_P = x0 * x0'
    new_P += mapreduce(d -> d * d', +, positive)
    new_P += mapreduce(d -> d * d', +, negative)
    new_P += Q

    return new_μ, new_P
end

end # module UKF
