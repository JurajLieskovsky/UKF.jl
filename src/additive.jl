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
function predict!(f, μx, Σx, μw, Σw, u)
    # sigma points and their weights
    seeds, w_μ, w_Σ = unscented_transform(μx, Σx)

    # propagation of sigma points through system dynamics
    sigma_points = map(pt -> f(pt, u) + μw, seeds)

    # calculate new mean and covariance
    μx .= mapreduce((w, pt) -> pt * w, +, w_μ, sigma_points)
    Σx .= mapreduce((w, pt) -> (pt - μx) * w * (pt - μx)', +, w_Σ, sigma_points) + Σw

    # ensure symmetry
    Σx .= 0.5 * (Σx + Σx')

    return nothing
end

"""
Update for measurements with additive noise.
"""
function update!(h, μx, Σx, μv, Σv, z)
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
    μx .-= K * (μ_y - z)
    Σx .-= K * Σ_xy'

    # Ensure symmetry
    Σx .= 0.5 * (Σx + Σx')

    return return nothing
end

