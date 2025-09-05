using Test
using RDE

@testset "Types and Initialization" begin
    @testset "RDEParam" begin
        params = RDEParam{Float32}(N = 16)
        @test params.N == 16
        @test typeof(params.L) == Float32
        @test typeof(params.ν_1) == Float32
    end

    @testset "Method Types" begin
        # Test PseudospectralMethod
        ps_method = PseudospectralMethod{Float32}()
        @test ps_method.dealias == false
        @test ps_method.cache === nothing
        @test sprint(show, ps_method) == "PseudospectralMethod{Float32} (without dealiasing, cache uninitialized)"

        ps_method_no_dealias = PseudospectralMethod{Float32}(dealias = true)
        @test ps_method_no_dealias.dealias == true
        @test ps_method_no_dealias.cache === nothing
        @test sprint(show, ps_method_no_dealias) == "PseudospectralMethod{Float32} (with dealiasing, cache uninitialized)"

        # Test FiniteDifferenceMethod
        fd_method = FiniteDifferenceMethod{Float32}()
        @test fd_method.cache === nothing
        @test sprint(show, fd_method) == "FiniteDifferenceMethod{Float32} (cache uninitialized)"

        # Test default constructor
        fd_method_default = FiniteDifferenceMethod()
        @test typeof(fd_method_default) == FiniteDifferenceMethod{Float32}
    end

    @testset "Cache Initialization" begin
        params = RDEParam{Float32}(N = 16)
        dx = Float32(2π / 16)

        # Test PseudospectralMethod cache
        ps_method = PseudospectralMethod{Float32}()
        RDE.init_cache!(ps_method, params, dx)
        @test ps_method.cache isa RDE.PseudospectralRDECache{Float32}
        @test length(ps_method.cache.u_x) == params.N
        @test length(ps_method.cache.dealias_filter) == div(params.N, 2) + 1
        @test sprint(show, ps_method) == "PseudospectralMethod{Float32} (without dealiasing, cache initialized)"

        # Test FiniteDifferenceMethod cache
        fd_method = FiniteDifferenceMethod{Float32}()
        RDE.init_cache!(fd_method, params, dx)
        @test fd_method.cache isa RDE.FDRDECache{Float32}
        @test length(fd_method.cache.u_x) == params.N
        @test fd_method.cache.dx == dx
        @test sprint(show, fd_method) == "FiniteDifferenceMethod{Float32} (cache initialized)"
    end

    @testset "Problem Construction" begin
        params = RDEParam{Float32}(N = 16)

        # Test with default method
        prob = RDEProblem(params)
        @test prob.method isa FiniteDifferenceMethod{Float32}
        @test prob.method.cache isa RDE.FDRDECache{Float32}
        @test prob.reset_strategy isa Default
        @test prob.control_shift_strategy isa ZeroControlShift

        # Test with pseudospectral method
        prob_ps = RDEProblem(params, method = PseudospectralMethod{Float32}())
        @test prob_ps.method isa PseudospectralMethod{Float32}
        @test prob_ps.method.cache isa RDE.PseudospectralRDECache{Float32}
    end
end
