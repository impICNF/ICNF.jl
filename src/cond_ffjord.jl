export CondFFJORD

"""
Implementation of FFJORD (Conditional Version)
"""
struct CondFFJORD{T <: AbstractFloat} <: AbstractCondICNF{T}
    re::Optimisers.Restructure
    p::AbstractVector{T}

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    solvealg_test::SciMLBase.AbstractODEAlgorithm
    solvealg_train::SciMLBase.AbstractODEAlgorithm

    sensealg_test::SciMLBase.AbstractSensitivityAlgorithm
    sensealg_train::SciMLBase.AbstractSensitivityAlgorithm

    acceleration::AbstractResource

    array_mover::Function
    ϵ::AbstractVector{T}

    # trace_test
    # trace_train
end

function CondFFJORD{T}(
        nn,
        nvars::Integer,
        ;
        basedist::Distribution=MvNormal(Zeros{T}(nvars), one(T)*I),
        tspan::Tuple{T, T}=convert(Tuple{T, T}, (0, 1)),

        solvealg_test::SciMLBase.AbstractODEAlgorithm=default_solvealg,
        solvealg_train::SciMLBase.AbstractODEAlgorithm=default_solvealg,

        sensealg_test::SciMLBase.AbstractSensitivityAlgorithm=default_sensealg,
        sensealg_train::SciMLBase.AbstractSensitivityAlgorithm=default_sensealg,

        acceleration::AbstractResource=default_acceleration,

        rng::AbstractRNG=Random.default_rng(),
        ) where {T <: AbstractFloat}
    array_mover = make_mover(acceleration, T)
    nn = fmap(x -> adapt(T, x), nn)
    p, re = destructure(nn)
    CondFFJORD{T}(
        re, p |> array_mover, nvars, basedist, tspan,
        solvealg_test, solvealg_train,
        sensealg_test, sensealg_train,
        acceleration, array_mover, randn(rng, T, nvars) |> array_mover,
    )
end

function augmented_f(icnf::CondFFJORD{T}, mode::TestMode, ys::AbstractMatrix{T})::Function where {T <: AbstractFloat}

    function f_aug(u, p, t)
        m = Chain(
            x -> vcat(x, ys),
            icnf.re(p),
        )
        z = u[1:end - 1, :]
        mz, J = jacobian_batched(m, z, icnf.array_mover)
        trace_J = transpose(tr.(eachslice(J; dims=3)))
        vcat(mz, -trace_J)
    end
    f_aug
end

function augmented_f(icnf::CondFFJORD{T}, mode::TrainMode, ys::AbstractMatrix{T})::Function where {T <: AbstractFloat, T2 <: Integer}

    function f_aug(u, p, t)
        m = Chain(
            x -> vcat(x, ys),
            icnf.re(p),
        )
        z = u[1:end - 1, :]
        mz, back = Zygote.pullback(m, z)
        ϵJ = only(back(icnf.ϵ))
        trace_J = transpose(icnf.ϵ) * ϵJ
        vcat(mz, -trace_J)
    end
    f_aug
end

function inference(icnf::CondFFJORD{T}, mode::TestMode, xs::AbstractMatrix{T}, ys::AbstractMatrix{T}, p::AbstractVector=icnf.p)::AbstractVector where {T <: AbstractFloat}
    xs = xs |> icnf.array_mover
    ys = ys |> icnf.array_mover
    zrs = zeros(T, 1, size(xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, mode, ys)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(xs, zrs), icnf.tspan, p)
    sol = solve(prob, icnf.solvealg_test; sensealg=icnf.sensealg_test)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    Δlogp = fsol[end, :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    logp̂x
end

function inference(icnf::CondFFJORD{T}, mode::TrainMode, xs::AbstractMatrix{T}, ys::AbstractMatrix{T}, p::AbstractVector=icnf.p)::AbstractVector where {T <: AbstractFloat}
    xs = xs |> icnf.array_mover
    ys = ys |> icnf.array_mover
    zrs = zeros(T, 1, size(xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, mode, ys)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(xs, zrs), icnf.tspan, p)
    sol = solve(prob, icnf.solvealg_train; sensealg=icnf.sensealg_train)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    Δlogp = fsol[end, :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    logp̂x
end

function generate(icnf::CondFFJORD{T}, mode::TestMode, ys::AbstractMatrix{T}, n::Integer, p::AbstractVector=icnf.p; rng::AbstractRNG=Random.default_rng())::AbstractMatrix{T} where {T <: AbstractFloat}
    ys = ys |> icnf.array_mover
    new_xs = rand(rng, icnf.basedist, n) |> icnf.array_mover
    zrs = zeros(T, 1, size(new_xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, mode, ys)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(new_xs, zrs), reverse(icnf.tspan), p)
    sol = solve(prob, icnf.solvealg_test; sensealg=icnf.sensealg_test)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    z
end

function generate(icnf::CondFFJORD{T}, mode::TrainMode, ys::AbstractMatrix{T}, n::Integer, p::AbstractVector=icnf.p; rng::AbstractRNG=Random.default_rng())::AbstractMatrix{T} where {T <: AbstractFloat}
    ys = ys |> icnf.array_mover
    new_xs = rand(rng, icnf.basedist, n) |> icnf.array_mover
    zrs = zeros(T, 1, size(new_xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, mode, ys)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(new_xs, zrs), reverse(icnf.tspan), p)
    sol = solve(prob, icnf.solvealg_train; sensealg=icnf.sensealg_train)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    z
end

Flux.@functor CondFFJORD (p,)

function loss(icnf::CondFFJORD{T}, xs::AbstractMatrix{T}, ys::AbstractMatrix{T}, p::AbstractVector=icnf.p; agg::Function=mean) where {T <: AbstractFloat}
    logp̂x = inference(icnf, TrainMode(), xs, ys, p)
    agg(-logp̂x)
end
