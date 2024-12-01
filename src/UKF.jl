module UKF

using LinearAlgebra

function unscented_transform(x, P)
    # constants
    n = length(x)
    λ = 3 - n

    # weights
    mean_weights = 1 / (2 * (n + λ)) * ones(2 * n + 1)
    mean_weights[1] = λ / (n + λ)

    cov_weights = diagm(mean_weights)

    # sigma points
    S = cholesky(P).L
    positive = map(s -> x + sqrt(n + λ) * s, eachcol(S))
    negative = map(s -> x - sqrt(n + λ) * s, eachcol(S))

    return vcat([x], positive, negative), mean_weights, cov_weights
end

odd_even_direction(i::Int) = (i % 2 == 0) ? 1 : -1

end # module UKF
