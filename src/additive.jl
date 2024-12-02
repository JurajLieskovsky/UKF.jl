"""
Unscented transformation for processes and measurements with additive noise.
"""
function unscented_transform(μ, Σ, λ=2, α=1, β=0)
    n = length(μ)

    # weights
    w_μi, w_μ0, w_Σi, w_Σ0 = weights(n, λ, α, β)

    # central sigma point
    central = SigmaPoint(copy(μ), w_μ0, w_Σ0)

    # auxilary sigma points
    S = cholesky(Σ).L
    positive = map(s -> SigmaPoint(μ + sqrt(n + λ) * s, w_μi, w_Σi), eachcol(S))
    negative = map(s -> SigmaPoint(μ - sqrt(n + λ) * s, w_μi, w_Σi), eachcol(S))

    return vcat([central], positive, negative)
end

"""
Prediction step for processes with additive noise.
"""
function predict(f, μx, Σx, Σw, u)
    # sigma points
    sigma_points = unscented_transform(μx, Σx)

    # propagation of sigma points through system dynamics
    map(x -> x.val .= f(x.val, u), sigma_points)

    # calculate new mean and covariance
    μ_new = mapreduce(x -> x.val * x.w_μ, +, sigma_points)
    Σ_new = mapreduce(x -> (x.val - μ_new) * x.w_Σ * (x.val - μ_new)', +, sigma_points) + Σw

    return μ_new, 0.5 * Σ_new * Σ_new'
end

"""
Update for measurements with additive noise.
"""
function update(h, μ, P, z, R)
    # state sigma points
    sigma_x = unscented_transform(μ, P)

    # measurements from sigma points
    sigma_y = map(x -> SigmaPoint(h(x.val), x.w_μ, x.w_Σ), sigma_x)

    # new mean and covariances
    μ_y = mapreduce(y -> y.val * y.w_μ, +, sigma_y)

    Σ_yy = mapreduce(y -> (y.val - μ_y) * y.w_Σ * (y.val - μ_y)', +, sigma_y) + R
    Σ_xy = mapreduce((x, y) -> (x.val - μ) * y.w_Σ * (y.val - μ_y)', +, sigma_x, sigma_y)

    # Kalman gain
    K = Σ_xy * inv(Σ_yy)

    # Updated state estimate
    μ_new = μ - K * (μ_y - z)
    Σ_new = P - K * Σ_xy'

    return μ_new, 0.5 * Σ_new * Σ_new'
end

