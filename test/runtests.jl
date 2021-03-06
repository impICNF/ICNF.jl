using
    ICNF,
    AbstractDifferentiation,
    CUDA,
    DataFrames,
    Distributions,
    FiniteDiff,
    FiniteDifferences,
    Flux,
    ForwardDiff,
    Optimization,
    ReverseDiff,
    MLJBase,
    SciMLBase,
    Test,
    Tracker,
    Yota,
    Zygote

CUDA.allowscalar() do
    include("core.jl")

    @testset "Overall" begin
        include("smoke_tests.jl")
    end
end
