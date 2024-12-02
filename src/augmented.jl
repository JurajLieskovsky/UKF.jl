"""
Unscented transformation for processes and measurements with non-additive noise.
"""
function augmented_unscented_transform(μ, Σ, μd, Σd, λ=2, α=1, β=0)
    n = length(μ) + length(μd)

    # central sigma point
    central = copy(μ)

    # standard auxilary sigma points
    S = cholesky(Σ).L
    positive = map(s -> μ + sqrt(n + λ) * s, eachcol(S))
    negative = map(s -> μ - sqrt(n + λ) * s, eachcol(S))

    # disturbance auxilary sigma points
    aug_S = cholesky(Σd).L
    aug_positive = map(s -> μd + sqrt(n + λ) * s, eachcol(aug_S))
    aug_negative = map(s -> μd - sqrt(n + λ) * s, eachcol(aug_S))

    return (
        vcat([central], positive, negative),
        vcat(aug_positive, aug_negative),
        weights(n, λ, α, β)...
    )
end

"""
Prediction step for processes with non-additive noise.
"""
function augmented_predict(f, μx, Σx, μw, Σw, u)
    # sigma points and their weights
    std_seeds, aug_seeds, w_μ, w_Σ = augmented_unscented_transform(μx, Σx, μw, Σw)

    # propagation of points through system dynamics
    std_points = map(pt -> f(pt, u, copy(μw)), std_seeds)
    aug_points = map(pt -> f(copy(μx), u, pt), aug_seeds)

    # concatenation of sigma points
    sigma_points = vcat(std_points, aug_points)

    # calculate new mean and covariance
    μ_new = mapreduce((w, pt) -> pt * w, +, w_μ, sigma_points)
    Σ_new = mapreduce((w, pt) -> (pt - μ_new) * w * (pt - μ_new)', +, w_Σ, sigma_points)

    return μ_new, 0.5 * Σ_new * Σ_new'
end
