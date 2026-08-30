using Test
using RDE

@testitem "Cinfty n=1 matches smooth_control!" begin
    T = Float32
    τ = T(0.1)
    t0 = zero(T)
    prev = T(0.5)
    next = T(1.0)
    profile = UniformMultiStepPressureProfile(
        T[t0, t0 + one(T)],
        T[prev, next],
        T(3.5),
        CinftySmoother(τ),
    )
    u_p = zeros(T, 8)
    s = zeros(T, 8)
    target = zeros(T, 8)
    for t in T(0):T(0.02):T(0.3)
        update_injection!(u_p, s, profile, t)
        RDE.smooth_control!(target, t, t0, fill(next, 8), fill(prev, 8), τ)
        @test u_p ≈ target
        @test all(s .== T(3.5))
    end
end

@testitem "commit_schedule! previous basis is eval at t0" begin
    T = Float32
    profile = UniformMultiStepPressureProfile(
        T[0, 1],
        T[0.4, 0.8],
        T(3.5),
        CinftySmoother(T(0.1)),
    )
    live = RDE._eval_uniform_u_p(profile, T(0.05))
    commit_schedule!(profile, T(0.05), T(1.05), T(1.1))
    @test previous_u_p(profile) ≈ live
    @test current_u_p(profile) ≈ T(1.1)
    @test profile.times[1] ≈ T(0.05)
end

@testitem "spatial commit smooths new knots only" begin
    T = Float32
    N = 32
    raw = vcat(fill(T(0), N ÷ 2), fill(T(1), N ÷ 2))
    profile = SpatialMultiStepPressureProfile(
        T[0, 1],
        [copy(raw), copy(raw)],
        T(3.5),
        CinftySmoother(T(0.1)),
    )
    set_spatial_kernel!(profile, 9)
    prev_before = copy(profile.u_p_values[1])
    new_target = reverse(raw)
    commit_schedule!(profile, T(0), T(1), new_target)
    @test previous_u_p(profile) ≈ prev_before
    @test current_u_p(profile) ≉ new_target
    boundary = N ÷ 2 + 1
    @test abs(current_u_p(profile)[boundary] - current_u_p(profile)[boundary - 1]) <
        abs(new_target[boundary] - new_target[boundary - 1])
end

@testitem "multistep segment indexing" begin
    T = Float32
    times = T[0, 0.25, 0.5, 1]
    values = T[0.2, 0.5, 0.8, 1.0]
    profile = UniformMultiStepPressureProfile(times, values, T(3.5), LinearSmoother())
    @test RDE._multistep_segment(times, T(-1)) == 1
    @test RDE._multistep_segment(times, T(0.3)) == 2
    @test RDE._multistep_segment(times, T(2)) == 3
    u_p = zeros(T, 4)
    s = zeros(T, 4)
    update_injection!(u_p, s, profile, T(0.375))
    @test u_p[1] ≈ T(0.65)
end

@testitem "MovingFrame dest/src shift" begin
    T = Float32
    N = 16
    inner = UniformMultiStepPressureProfile(
        T[0, 1],
        T[0.7, 0.7],
        T(3.5),
        CinftySmoother(T(0.1)),
    )
    wrapped = MovingFrameInjectionProfile(inner, LinearControlShift(T(1)), T(2π / N), N)
    u_p = zeros(T, N)
    s = zeros(T, N)
    update_injection!(u_p, s, wrapped, T(0.5))
    @test all(u_p .≈ T(0.7))
    @test RDE.control_shift(wrapped) isa LinearControlShift
    @test is_uniform_injection(wrapped)
end
