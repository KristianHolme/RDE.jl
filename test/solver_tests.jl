@test begin
    for T in [Float32, Float64]
        prob = RDEProblem(RDEParam{T}(;N=32, tmax = 0.01));
        solve_pde!(prob);
    end
    true
end

@test begin
    success = true
    for T in [Float32, Float64]
        rde = RDEProblem(RDEParam{T}(;N=128, tmax = 5.0), method = :pseudospectral);
        fft_plan = rde.cache.fft_plan
        ifft_plan = rde.cache.ifft_plan
        u0 = rde.u0
        u0_hat = fft_plan*u0
        u0_hat_hat = ifft_plan*u0_hat
        success = success && (u0_hat_hat ≈ u0)
    end
    success
end

@test RDE.ω(1.0, 0.0, 1.0) ≈ exp(1.0)
@test RDE.ω(1.0, 1.0, 1.0) ≈ 1.0
@test RDE.ω(0.0, 1.0, 0.5) ≈ exp(-2.0)

@test RDE.ξ(1.0, 0.0, 1.0) ≈ -1.0
@test RDE.ξ(0.0, 1.0, 1.0) ≈ 0.0
@test RDE.ξ(2.0, 3.0, 2) ≈ 4.0


@test RDE.β(1.0, 1.0, 0.0, 1.0) ≈ 0.0
@test RDE.β(1.0, 1.0, 1.0, 1.0) ≈ 0.5
@test RDE.β(2.0, 3.5, 0.56, 5.0) ≈ 0.0014622165143

