@testset "Smoke Tests" begin
    @testset "$mt | $cr | $tp | $nvars Vars" for
            mt in [FFJORD, RNODE],
            cr in [CPU1(), CUDALibs()],
            tp in [Float64, Float32, Float16],
            nvars in 1:3
        icnf = mt{tp}(Dense(nvars, nvars), nvars; acceleration=cr)
        ufd = copy(icnf.p)
        n = 8
        r = rand(tp, nvars, n)

        @test !isnothing(inference(icnf, TestMode(), r))
        @test !isnothing(inference(icnf, TrainMode(), r))

        @test !isnothing(generate(icnf, TestMode(), n))
        @test !isnothing(generate(icnf, TrainMode(), n))

        @test !isnothing(icnf(r))
        @test !isnothing(loss_f(icnf)(r))

        @test !isnothing(agg_loglikelihood(icnf, r))

        d = ICNFDistribution(icnf)

        @test !isnothing(logpdf(d, r))
        @test !isnothing(pdf(d, r))
        @test !isnothing(rand(d))
        @test !isnothing(rand(d, n))

        df = DataFrame(r', :auto)
        model = ICNFModel(icnf; n_epochs=8)
        mach = machine(model, df)
        fit!(mach)
        fd = MLJBase.fitted_params(mach).learned_parameters

        @test !isnothing(MLJBase.transform(mach, df))
        if tp <: Float16
            @test_broken fd != ufd
        else
            @test fd != ufd
        end
    end
    @testset "$mt | $cr | $tp | $nvars Vars" for
            mt in [CondFFJORD, CondRNODE],
            cr in [CPU1(), CUDALibs()],
            tp in [Float64, Float32, Float16],
            nvars in 1:3
        icnf = mt{tp}(Dense(nvars*2, nvars), nvars; acceleration=cr)
        ufd = copy(icnf.p)
        n = 8
        r = rand(tp, nvars, n)
        r2 = rand(tp, nvars, n)

        @test !isnothing(inference(icnf, TestMode(), r, r2))
        @test !isnothing(inference(icnf, TrainMode(), r, r2))

        @test !isnothing(generate(icnf, TestMode(), r2, n))
        @test !isnothing(generate(icnf, TrainMode(), r2, n))

        @test !isnothing(icnf(r, r2))
        @test !isnothing(loss_f(icnf)(r, r2))

        @test !isnothing(agg_loglikelihood(icnf, r, r2))

        df = DataFrame(r', :auto)
        df2 = DataFrame(r2', :auto)
        model = CondICNFModel(icnf; n_epochs=8)
        mach = machine(model, (df, df2))
        fit!(mach)
        fd = MLJBase.fitted_params(mach).learned_parameters

        @test !isnothing(MLJBase.transform(mach, (df, df2)))
        if tp <: Float16
            @test_broken fd != ufd
        else
            @test fd != ufd
        end
    end
end