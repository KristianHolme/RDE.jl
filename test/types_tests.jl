using Test
using RDE

@testitem "RDEParam" begin
    params = RDEParam{Float32}(N = 16)
    @test params.N == 16
    @test typeof(params.L) == Float32
    @test typeof(params.ν_1) == Float32
end

@testitem "Method Types" begin
    fv_method = FiniteVolumeMethod{Float32}()
    @test fv_method.cache === nothing
    @test fv_method.limiter isa MCLimiter
    @test sprint(show, fv_method) == "FiniteVolumeMethod{Float32,MCLimiter} (cache uninitialized)"

    fv_method_minmod = FiniteVolumeMethod{Float32}(limiter = MinmodLimiter())
    @test fv_method_minmod.cache === nothing
    @test fv_method_minmod.limiter isa MinmodLimiter
    @test sprint(show, fv_method_minmod) == "FiniteVolumeMethod{Float32,MinmodLimiter} (cache uninitialized)"

    fv_method_default = FiniteVolumeMethod()
    @test typeof(fv_method_default) == FiniteVolumeMethod{Float32, MCLimiter}
end

@testitem "Cache Initialization" begin
    params = RDEParam{Float32}(N = 16)
    dx = Float32(2π / 16)

    fv_method = FiniteVolumeMethod{Float32}()
    RDE.init_cache!(fv_method, params, dx)
    @test fv_method.cache isa RDE.FVCache{Float32}
    @test length(fv_method.cache.u_xx) == params.N
    @test fv_method.cache.dx == dx
    @test sprint(show, fv_method) == "FiniteVolumeMethod{Float32,MCLimiter} (cache initialized)"
end

@testitem "Problem Construction" begin
    params = RDEParam{Float32}(N = 16)

    # Test with default method
    prob = RDEProblem(params)
    @test prob.method isa FiniteVolumeMethod{Float32}
    @test prob.method.cache isa RDE.FVCache{Float32}
    @test prob.reset_strategy isa Default
    @test prob.control_shift_strategy isa ZeroControlShift

    prob_minmod = RDEProblem(params, method = FiniteVolumeMethod{Float32}(limiter = MinmodLimiter()))
    @test prob_minmod.method isa FiniteVolumeMethod{Float32, MinmodLimiter}
    @test prob_minmod.method.cache isa RDE.FVCache{Float32}
end
