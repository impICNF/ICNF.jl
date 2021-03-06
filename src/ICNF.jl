module ICNF

    using
        Adapt,
        CUDA,
        DataFrames,
        DifferentialEquations,
        Distributions,
        DistributionsAD,
        FillArrays,
        Flux,
        IterTools,
        MLJBase,
        MLJModelInterface,
        MLUtils,
        NNlib,
        Optim,
        Optimisers,
        Optimization,
        OptimizationFlux,
        OptimizationOptimJL,
        OptimizationOptimisers,
        OrdinaryDiffEq,
        SciMLBase,
        SciMLSensitivity,
        ScientificTypes,
        Zygote,
        LinearAlgebra,
        Random,
        Statistics

    include("types.jl")
    include("defaults.jl")
    include("core_icnf.jl")
    include("core_cond_icnf.jl")

    include("rnode.jl")
    include("ffjord.jl")
    include("planar.jl")

    include("cond_rnode.jl")
    include("cond_ffjord.jl")
    include("cond_planar.jl")

    include("metrics.jl")

    include("utils.jl")

end
