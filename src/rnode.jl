export RNODE

"""
Implementation of RNODE from

[Finlay, Chris, Jörn-Henrik Jacobsen, Levon Nurbekyan, and Adam M. Oberman. "How to train your neural ODE: the world of Jacobian and kinetic regularization." arXiv preprint arXiv:2002.02798 (2020).](https://arxiv.org/abs/2002.02798)
"""
struct RNODE{T <: AbstractFloat, AT <: AbstractArray} <: AbstractICNF{T, AT}
    re::Optimisers.Restructure
    p::AbstractVector

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    solvealg_test::SciMLBase.AbstractODEAlgorithm
    solvealg_train::SciMLBase.AbstractODEAlgorithm

    sensealg_test::SciMLBase.AbstractSensitivityAlgorithm
    sensealg_train::SciMLBase.AbstractSensitivityAlgorithm

    ϵ::AbstractVector

    # trace_test
    # trace_train
end

function RNODE{T, AT}(
        nn,
        nvars::Integer,
        ;
        basedist::Distribution=MvNormal(Zeros{T}(nvars), one(T)*I),
        tspan::Tuple{T, T}=convert(Tuple{T, T}, (0, 1)),

        solvealg_test::SciMLBase.AbstractODEAlgorithm=default_solvealg,
        solvealg_train::SciMLBase.AbstractODEAlgorithm=default_solvealg,

        sensealg_test::SciMLBase.AbstractSensitivityAlgorithm=default_sensealg,
        sensealg_train::SciMLBase.AbstractSensitivityAlgorithm=default_sensealg,

        rng::AbstractRNG=Random.default_rng(),
        ) where {T <: AbstractFloat, AT <: AbstractArray}
    nn = fmap(x -> adapt(T, x), nn)
    p, re = destructure(nn)
    RNODE{T, AT}(
        re, convert(AT{T}, p), nvars, basedist, tspan,
        solvealg_test, solvealg_train,
        sensealg_test, sensealg_train,
        convert(AT, randn(rng, T, nvars)),
    )
end

function augmented_f(icnf::RNODE{T, AT}, mode::TestMode)::Function where {T <: AbstractFloat, AT <: AbstractArray}

    function f_aug(u, p, t)
        m = icnf.re(p)
        z = u[1:end - 1, :]
        ż, J = jacobian_batched(m, z, T, AT)
        l̇ = transpose(tr.(eachslice(J; dims=3)))
        vcat(ż, -l̇)
    end
    f_aug
end

function augmented_f(icnf::RNODE{T, AT}, mode::TrainMode)::Function where {T <: AbstractFloat, AT <: AbstractArray}

    function f_aug(u, p, t)
        m = icnf.re(p)
        z = u[1:end - 3, :]
        ż, back = Zygote.pullback(m, z)
        ϵJ = only(back(icnf.ϵ))
        l̇ = transpose(icnf.ϵ) * ϵJ
        Ė = transpose(norm.(eachcol(ż)))
        ṅ = transpose(norm.(eachcol(ϵJ)))
        vcat(ż, -l̇, Ė, ṅ)
    end
    f_aug
end

function inference(icnf::RNODE{T, AT}, mode::TestMode, xs::AbstractMatrix, p::AbstractVector=icnf.p)::AbstractVector where {T <: AbstractFloat, AT <: AbstractArray}
    zrs = convert(AT, zeros(T, 1, size(xs, 2)))
    f_aug = augmented_f(icnf, mode)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(xs, zrs), icnf.tspan, p; alg=icnf.solvealg_test, sensealg=icnf.sensealg_test)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    Δlogp = fsol[end, :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    logp̂x
end

function inference(icnf::RNODE{T, AT}, mode::TrainMode, xs::AbstractMatrix, p::AbstractVector=icnf.p)::Tuple where {T <: AbstractFloat, AT <: AbstractArray}
    zrs = convert(AT, zeros(T, 3, size(xs, 2)))
    f_aug = augmented_f(icnf, mode)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(xs, zrs), icnf.tspan, p; alg=icnf.solvealg_train, sensealg=icnf.sensealg_train)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:end - 3, :]
    Δlogp = fsol[end - 2, :]
    Ė = fsol[end - 1, :]
    ṅ = fsol[end, :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    logp̂x, Ė, ṅ
end

function generate(icnf::RNODE{T, AT}, mode::TestMode, n::Integer, p::AbstractVector=icnf.p; rng::AbstractRNG=Random.default_rng())::AbstractMatrix where {T <: AbstractFloat, AT <: AbstractArray}
    new_xs = convert(AT, rand(rng, icnf.basedist, n))
    zrs = convert(AT, zeros(T, 1, size(new_xs, 2)))
    f_aug = augmented_f(icnf, mode)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(new_xs, zrs), reverse(icnf.tspan), p; alg=icnf.solvealg_test, sensealg=icnf.sensealg_test)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    z
end

function generate(icnf::RNODE{T, AT}, mode::TrainMode, n::Integer, p::AbstractVector=icnf.p; rng::AbstractRNG=Random.default_rng())::AbstractMatrix where {T <: AbstractFloat, AT <: AbstractArray}
    new_xs = convert(AT, rand(rng, icnf.basedist, n))
    zrs = convert(AT, zeros(T, 3, size(new_xs, 2)))
    f_aug = augmented_f(icnf, mode)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(new_xs, zrs), reverse(icnf.tspan), p; alg=icnf.solvealg_train, sensealg=icnf.sensealg_train)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:end - 3, :]
    z
end

Flux.@functor RNODE (p,)

function loss(icnf::RNODE{T, AT}, xs::AbstractMatrix, p::AbstractVector=icnf.p, λ₁::T=convert(T, 1e-2), λ₂::T=convert(T, 1e-2); agg::Function=mean)::Number where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x, Ė, ṅ = inference(icnf, TrainMode(), xs, p)
    agg(-logp̂x + λ₁*Ė + λ₂*ṅ)
end
