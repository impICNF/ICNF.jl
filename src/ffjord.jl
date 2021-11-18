export FFJORD

"""
Implementations of

[Grathwohl, Will, Ricky TQ Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. "Ffjord: Free-form continuous dynamics for scalable reversible generative models." arXiv preprint arXiv:1810.01367 (2018).](https://arxiv.org/abs/1810.01367)
"""
@with_kw struct FFJORD{T} <: AbstractICNF where {T <: AbstractFloat}
    re::Function
    p::AbstractVector{T}

    nvars::Integer
    basedist::Distribution = MvNormal(zeros(T, nvars), Diagonal(ones(T, nvars)))
    tspan::Tuple{T, T} = convert.(T, (0, 1))

    solver_test::OrdinaryDiffEqAlgorithm = default_solver_test
    solver_train::OrdinaryDiffEqAlgorithm = default_solver_train

    sensealg_test::SciMLBase.AbstractSensitivityAlgorithm = default_sensealg
    sensealg_train::SciMLBase.AbstractSensitivityAlgorithm = sensealg_test

    acceleration::AbstractResource = CPU1()

    # trace_test
    # trace_train
end

function FFJORD{T}(
        nn,
        nvars::Integer,
        ;
        basedist::Distribution=MvNormal(zeros(T, nvars), Diagonal(ones(T, nvars))),
        tspan::Tuple{T, T}=convert.(T, (0, 1)),

        solver_test::OrdinaryDiffEqAlgorithm=default_solver_test,
        solver_train::OrdinaryDiffEqAlgorithm=default_solver_train,

        sensealg_test::SciMLBase.AbstractSensitivityAlgorithm=default_sensealg,
        sensealg_train::SciMLBase.AbstractSensitivityAlgorithm=sensealg_test,

        acceleration::AbstractResource=CPU1(),
        ) where {T <: AbstractFloat}
    move = MLJFlux.Mover(acceleration)
    if T <: Float64
        nn = f64(nn)
    elseif T <: Float32
        nn = f32(nn)
    else
        nn = Flux.paramtype(T, nn)
    end
    nn = move(nn)
    p, re = Flux.destructure(nn)
    FFJORD{T}(;
        re, p, nvars, basedist, tspan,
        solver_test, solver_train,
        sensealg_test, sensealg_train,
        acceleration,
    )
end

function augmented_f(icnf::FFJORD{T}, mode::TestMode)::Function where {T <: AbstractFloat}
    move = MLJFlux.Mover(icnf.acceleration)

    function f_aug(u, p, t)
        m = icnf.re(p)
        z = u[1:end - 1, :]
        mz = m(z)
        J = jacobian_batched(m, z, move)
        trace_J = transpose(tr.(eachslice(J, dims=3)))
        vcat(mz, -trace_J)
    end
    f_aug
end

function augmented_f(icnf::FFJORD{T}, mode::TrainMode, sz::Tuple{Int64, Int64})::Function where {T <: AbstractFloat}
    move = MLJFlux.Mover(icnf.acceleration)
    ϵ = randn(T, sz) |> move

    function f_aug(u, p, t)
        m = icnf.re(p)
        z = u[1:end - 1, :]
        mz, back = Zygote.pullback(m, z)
        ϵJ = only(back(ϵ))
        trace_J = sum(ϵJ .* ϵ, dims=1)
        vcat(mz, -trace_J)
    end
    f_aug
end

function inference(icnf::FFJORD{T}, mode::TestMode, xs::AbstractMatrix{T})::AbstractVector{T} where {T <: AbstractFloat}
    move = MLJFlux.Mover(icnf.acceleration)
    xs = xs |> move
    zrs = zeros(T, 1, size(xs, 2)) |> move
    f_aug = augmented_f(icnf, mode)
    prob = ODEProblem{false}(f_aug, vcat(xs, zrs), icnf.tspan, icnf.p)
    sol = solve(prob, icnf.solver_test; sensealg=icnf.sensealg_test)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    Δlogp = fsol[end, :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    logp̂x
end

function inference(icnf::FFJORD{T}, mode::TrainMode, xs::AbstractMatrix{T})::AbstractVector{T} where {T <: AbstractFloat}
    move = MLJFlux.Mover(icnf.acceleration)
    xs = xs |> move
    zrs = zeros(T, 1, size(xs, 2)) |> move
    f_aug = augmented_f(icnf, mode, size(xs))
    prob = ODEProblem{false}(f_aug, vcat(xs, zrs), icnf.tspan, icnf.p)
    sol = solve(prob, icnf.solver_train; sensealg=icnf.sensealg_train)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    Δlogp = fsol[end, :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    logp̂x
end

function generate(icnf::FFJORD{T}, mode::TestMode, n::Integer; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat}
    move = MLJFlux.Mover(icnf.acceleration)
    new_xs = isnothing(rng) ? rand(icnf.basedist, n) : rand(rng, icnf.basedist, n)
    new_xs = new_xs |> move
    zrs = zeros(T, 1, size(new_xs, 2)) |> move
    f_aug = augmented_f(icnf, mode)
    prob = ODEProblem{false}(f_aug, vcat(new_xs, zrs), reverse(icnf.tspan), icnf.p)
    sol = solve(prob, icnf.solver_test; sensealg=icnf.sensealg_test)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    z
end

function generate(icnf::FFJORD{T}, mode::TrainMode, n::Integer; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat}
    move = MLJFlux.Mover(icnf.acceleration)
    new_xs = isnothing(rng) ? rand(icnf.basedist, n) : rand(rng, icnf.basedist, n)
    new_xs = new_xs |> move
    zrs = zeros(T, 1, size(new_xs, 2)) |> move
    f_aug = augmented_f(icnf, mode, size(new_xs))
    prob = ODEProblem{false}(f_aug, vcat(new_xs, zrs), reverse(icnf.tspan), icnf.p)
    sol = solve(prob, icnf.solver_train; sensealg=icnf.sensealg_train)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    z
end

Flux.@functor FFJORD (p,)

function loss_f(icnf::FFJORD{T}; agg::Function=mean)::Function where {T <: AbstractFloat}
    function f(x::AbstractMatrix{T})::T
        logp̂x = inference(icnf, TrainMode(), x)
        agg(-logp̂x)
    end
    f
end