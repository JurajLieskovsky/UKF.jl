"""
Unscented transformation for processes and measurements with additive noise.
"""
function unscented_transform(μ, Σ, λ=2, α=1, β=0)
    n = length(μ)

    # central sigma point
    central = copy(μ)

    # auxilary sigma points
    S = cholesky(Σ).L
    positive = map(s -> μ + sqrt(n + λ) * s, eachcol(S))
    negative = map(s -> μ - sqrt(n + λ) * s, eachcol(S))

    return (
        vcat([central], positive, negative),
        weights(n, λ, α, β)...
    )
end

"""
Prediction step for processes with additive noise.
"""
function predict(f, μx, Σx, μw, Σw, u)
    # sigma points and their weights
    seeds, w_μ, w_Σ = unscented_transform(μx, Σx)

    # propagation of sigma points through system dynamics
    sigma_points = map(pt -> f(pt, u) + μw, seeds)

    # calculate new mean and covariance
    μ_new = mapreduce((w, pt) -> pt * w, +, w_μ, sigma_points)
    Σ_new = mapreduce((w, pt) -> (pt - μ_new) * w * (pt - μ_new)', +, w_Σ, sigma_points) + Σw

    return μ_new, 0.5 * Σ_new * Σ_new'
end

"""
Update for measurements with additive noise.
"""
function update(h, μx, Σx, μv, Σv, z)
    # state sigma points and sigma point weights
    sigma_x, w_μ, w_Σ = unscented_transform(μx, Σx)

    # measurements from sigma points
    sigma_y = map(x -> h(x) + μv, sigma_x)

    # new mean and covariances
    μ_y = mapreduce((w, y) -> y * w, +, w_μ, sigma_y)

    Σ_yy = mapreduce((w, y) -> (y - μ_y) * w * (y - μ_y)', +, w_Σ, sigma_y) + Σv
    Σ_xy = mapreduce((w, x, y) -> (x - μx) * w * (y - μ_y)', +, w_Σ, sigma_x, sigma_y)

    # Kalman gain
    K = Σ_xy * inv(Σ_yy)

    # Updated state estimate
    μ_new = μx - K * (μ_y - z)
    Σ_new = Σx - K * Σ_xy'

    return μ_new, 0.5 * Σ_new * Σ_new'
end

