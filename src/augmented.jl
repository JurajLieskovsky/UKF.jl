"""
Unscented transformation for processes and measurements with non-additive noise.
"""
function augmented_unscented_transform(μ, Σ, μd, Σd, λ=2, α=1, β=0)
    n = length(μ) + length(μd)

    # weights
    w_μi, w_μ0, w_Σi, w_Σ0 = weights(n, λ, α, β)

    # central sigma point
    central = SigmaPoint(copy(μ), w_μ0, w_Σ0)

    # standard auxilary sigma points
    S = cholesky(Σ).L
    positive = map(s -> SigmaPoint(μ + sqrt(n + λ) * s, w_μi, w_Σi), eachcol(S))
    negative = map(s -> SigmaPoint(μ - sqrt(n + λ) * s, w_μi, w_Σi), eachcol(S))

    # disturbance auxilary sigma points
    aug_S = cholesky(Σd).L
    aug_positive = map(s -> SigmaPoint(μd + sqrt(n + λ) * s, w_μi, w_Σi), eachcol(aug_S))
    aug_negative = map(s -> SigmaPoint(μd - sqrt(n + λ) * s, w_μi, w_Σi), eachcol(aug_S))

    return vcat([central], positive, negative), vcat(aug_positive, aug_negative)
end

"""
Prediction step for processes with non-additive noise.
"""
function augmented_predict(f, μx, Σx, μw, Σw, u)
    # sigma points
    std_seeds, aug_seeds = augmented_unscented_transform(μx, Σx, μw, Σw)

    # propagation of points through system dynamics
    std_points = map(pt -> SigmaPoint(f(pt.val, u, copy(μw)), pt.w_μ, pt.w_Σ), std_seeds)
    aug_points = map(pt -> SigmaPoint(f(copy(μx), u, pt.val), pt.w_μ, pt.w_Σ), aug_seeds)

    # concatenation of sigma points
    sigma_points = vcat(std_points, aug_points)

    # calculate new mean and covariance
    μ_new = mapreduce(x -> x.val * x.w_μ, +, sigma_points)
    Σ_new = mapreduce(x -> (x.val - μ_new) * x.w_Σ * (x.val - μ_new)', +, sigma_points)

    return μ_new, 0.5 * Σ_new * Σ_new'
end
