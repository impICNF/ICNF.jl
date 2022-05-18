export
    inference, generate,
    loss, loss_pn, loss_pln,
    loss_f, callback_f,
    ICNFModel, ICNFDist

function inference(icnf::AbstractICNF{T}, mode::TestMode, xs::AbstractMatrix{T}, p::AbstractVector=icnf.p; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractVector where {T <: AbstractFloat} end
function inference(icnf::AbstractICNF{T}, mode::TrainMode, xs::AbstractMatrix{T}, p::AbstractVector=icnf.p; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractVector where {T <: AbstractFloat} end

function generate(icnf::AbstractICNF{T}, mode::TestMode, n::Integer, p::AbstractVector=icnf.p; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat} end
function generate(icnf::AbstractICNF{T}, mode::TrainMode, n::Integer, p::AbstractVector=icnf.p; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat} end

function loss(icnf::AbstractICNF{T}, xs::AbstractMatrix{T}, p::AbstractVector=icnf.p; agg::Function=mean) where {T <: AbstractFloat} end

function loss_pn(icnf::AbstractICNF{T}, xs::AbstractMatrix{T}, p::AbstractVector=icnf.p; agg::Function=mean, nλ::T=convert(T, 1e-4)) where {T <: AbstractFloat}
    lv = loss(icnf, xs, p; agg)
    prm_n = norm(p)
    lv + nλ*prm_n
end

function loss_pln(icnf::AbstractICNF{T}, xs::AbstractMatrix{T}, p::AbstractVector=icnf.p; agg::Function=mean, nλ::T=convert(T, 1e-4)) where {T <: AbstractFloat}
    lv = loss(icnf, xs, p; agg)
    prm_ln = log(norm(p))
    lv + nλ*prm_ln
end

# -- Flux interface

function (icnf::AbstractICNF{T})(xs::AbstractMatrix{T})::AbstractVector{T} where {T <: AbstractFloat}
    inference(icnf, TestMode(), xs)
end

function loss_f(icnf::AbstractICNF{T}, opt_app::FluxOptApp)::Function where {T <: AbstractFloat}
    function f(xs::AbstractMatrix{T})::T
        loss(icnf, xs)
    end
    f
end

function callback_f(icnf::AbstractICNF{T}, opt_app::FluxOptApp, loss::Function, data::DataLoader{T3})::Function where {T <: AbstractFloat, T2 <: AbstractMatrix{T}, T3 <: Tuple{T2}}
    xs, = first(data)
    function f()::Nothing
        vl = loss(icnf, xs)
        @info "Training" loss=vl
        nothing
    end
    f
end

# -- Optim interface

function loss_f(icnf::AbstractICNF{T}, opt_app::OptimOptApp, itrtr::AbstractVector)::Function where {T <: AbstractFloat}
    function f(p::AbstractVector{T})::T
        xs, = itrtr[1]
        loss(icnf, xs, p)
    end
    f
end

function callback_f(icnf::AbstractICNF{T}, opt_app::OptimOptApp, loss::Function, data::DataLoader{T3}, itrtr::AbstractVector)::Function where {T <: AbstractFloat, T2 <: AbstractMatrix{T}, T3 <: Tuple{T2}}
    xs, = first(data)
    function f(s::OptimizationState)::Bool
        vl = loss(icnf, xs, s.metadata["x"])
        @info "Training" loss=vl
        nxitr = iterate(data, itrtr[2])
        if isnothing(nxitr)
            true
        else
            itrtr .= nxitr
            false
        end
    end
    f
end

# -- SciML interface

function loss_f(icnf::AbstractICNF{T}, opt_app::SciMLOptApp)::Function where {T <: AbstractFloat}
    function f(p::AbstractVector, θ::SciMLBase.NullParameters, xs::AbstractMatrix{T})
        loss(icnf, xs, p)
    end
    f
end

function callback_f(icnf::AbstractICNF{T}, opt_app::SciMLOptApp, loss::Function, data::DataLoader{T3})::Function where {T <: AbstractFloat, T2 <: AbstractMatrix{T}, T3 <: Tuple{T2}}
    xs, = first(data)
    function f(p::AbstractVector{T}, l::T)::Bool
        vl = loss(icnf, xs, p)
        @info "Training" loss=vl
        false
    end
    f
end

# MLJ interface

mutable struct ICNFModel{T, T2} <: MLJICNF where {T <: AbstractFloat, T2 <: AbstractICNF{T}}
    m::T2
    loss::Function

    opt_app::OptApp
    optimizer::Any
    n_epochs::Integer
    adtype::SciMLBase.AbstractADType

    batch_size::Integer
end

function ICNFModel(
        m::T2,
        loss::Function=loss,
        ;
        opt_app::OptApp=FluxOptApp(),
        optimizer::Any=default_optimizer[typeof(opt_app)],
        n_epochs::Integer=128,
        adtype::SciMLBase.AbstractADType=GalacticOptim.AutoZygote(),

        batch_size::Integer=128,
        ) where {T <: AbstractFloat, T2 <: AbstractICNF{T}}
    ICNFModel{T, T2}(m, loss, opt_app, optimizer, n_epochs, adtype, batch_size)
end

function MLJModelInterface.fit(model::ICNFModel, verbosity, X)
    x = collect(MLJModelInterface.matrix(X)')
    data = DataLoader((x,); batchsize=model.batch_size, shuffle=true, partial=true)
    ncdata = ncycle(data, model.n_epochs)
    initial_loss_value = model.loss(model.m, x)

    if model.opt_app isa FluxOptApp
        @assert model.optimizer isa Flux.Optimise.AbstractOptimiser
        _loss = loss_f(model.m, model.opt_app)
        _callback = callback_f(model.m, model.opt_app, model.loss, data)
        _p = Flux.params(model.m)
        tst = @timed Flux.Optimise.train!(_loss, _p, ncdata, model.optimizer; cb=_callback)
    elseif model.opt_app isa OptimOptApp
        @assert model.optimizer isa Optim.AbstractOptimizer
        itrtr = Any[nothing, nothing]
        itrtr .= iterate(ncdata)
        _loss = loss_f(model.m, model.opt_app, itrtr)
        _callback = callback_f(model.m, model.opt_app, model.loss, data, itrtr)
        ops = Optim.Options(
            x_abstol=-Inf, x_reltol=-Inf,
            f_abstol=-Inf, f_reltol=-Inf,
            g_abstol=-Inf, g_reltol=-Inf,
            outer_x_abstol=-Inf, outer_x_reltol=-Inf,
            outer_f_abstol=-Inf, outer_f_reltol=-Inf,
            outer_g_abstol=-Inf, outer_g_reltol=-Inf,
            f_calls_limit=0, g_calls_limit=0, h_calls_limit=0,
            allow_f_increases=true, allow_outer_f_increases=true,
            successive_f_tol=typemax(Int), iterations=typemax(Int), outer_iterations=typemax(Int),
            store_trace=false, trace_simplex=true, show_trace=false, extended_trace=true,
            show_every=1, callback=_callback, time_limit=Inf,
        )
        tst = @timed res = optimize(_loss, model.m.p, model.optimizer, ops)
        model.m.p .= res.minimizer
    elseif model.opt_app isa SciMLOptApp
        _loss = loss_f(model.m, model.opt_app)
        _callback = callback_f(model.m, model.opt_app, model.loss, data)
        optfunc = OptimizationFunction(_loss, model.adtype)
        optprob = OptimizationProblem(optfunc, model.m.p)
        tst = @timed res = solve(optprob, model.optimizer, ncdata; callback=_callback)
        model.m.p .= res.u
    end
    final_loss_value = model.loss(model.m, x)
    @info("Fitting",
        "elapsed time (seconds)"=tst.time,
        "garbage collection time (seconds)"=tst.gctime,
    )

    fitresult = nothing
    cache = nothing
    report = (
        stats=tst,
        initial_loss_value=initial_loss_value,
        final_loss_value=final_loss_value,
    )
    fitresult, cache, report
end

function MLJModelInterface.transform(model::ICNFModel, fitresult, Xnew)
    xnew = collect(MLJModelInterface.matrix(Xnew)')

    tst = @timed logp̂x = inference(model.m, TestMode(), xnew)
    @info("Transforming",
        "elapsed time (seconds)"=tst.time,
        "garbage collection time (seconds)"=tst.gctime,
    )

    DataFrame(px=exp.(logp̂x))
end

function MLJModelInterface.fitted_params(model::ICNFModel, fitresult)
    (
        learned_parameters=model.m.p,
    )
end

MLJBase.metadata_pkg.(
    ICNFModel,
    package_name="ICNF",
    package_uuid="9bd0f7d2-bd29-441d-bcde-0d11364d2762",
    package_url="https://github.com/impICNF/ICNF.jl",
    is_pure_julia=true,
    package_license="MIT",
    is_wrapper=false,
)
MLJBase.metadata_model(
    ICNFModel,
    input_scitype=Table{AbstractVector{ScientificTypes.Continuous}},
    target_scitype=Table{AbstractVector{ScientificTypes.Continuous}},
    output_scitype=Table{AbstractVector{ScientificTypes.Continuous}},
    supports_weights=false,
    docstring="ICNFModel",
    load_path="ICNF.ICNFModel",
)

# Distributions interface

struct ICNFDist{T, T2} <: ICNFDistribution where {T <: AbstractFloat, T2 <: AbstractICNF{T}}
    m::T2
end

function ICNFDist(m::T2) where {T <: AbstractFloat, T2 <: AbstractICNF{T}}
    ICNFDist{T, T2}(m)
end

Base.length(d::ICNFDist) = d.m.nvars
Base.eltype(d::ICNFDist) = eltype(d.m.p)
Distributions._logpdf(d::ICNFDist, x::AbstractVector) = first(Distributions._logpdf(d, hcat(x)))
Distributions._logpdf(d::ICNFDist, A::AbstractMatrix) = inference(d.m, TestMode(), A)
Distributions._rand!(rng::AbstractRNG, d::ICNFDist, x::AbstractVector) = (x[:] = Distributions._rand!(rng, d, hcat(x)))
Distributions._rand!(rng::AbstractRNG, d::ICNFDist, A::AbstractMatrix) = (A[:] = generate(d.m, TestMode(), size(A, 2); rng))