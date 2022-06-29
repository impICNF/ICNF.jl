default_acceleration = CPU1()
default_solver = Tsit5(; thread=OrdinaryDiffEq.True())
default_sensealg = InterpolatingAdjoint(
    ;
    autodiff=true,
    chunk_size=0,
    autojacvec=ZygoteVJP(),
)
default_optimizer = Dict(
    FluxOptApp => Flux.AMSGrad(),
    OptimOptApp => BFGS(),
    SciMLOptApp => Optimisers.AMSGrad(),
)
default_differentiation_backend() = AbstractDifferentiation.ZygoteBackend()
