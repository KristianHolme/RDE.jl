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
        rde = RDEProblem(RDEParam{T}(;N=128, tmax = 5.0));
        fft_plan = rde.fft_plan
        ifft_plan = rde.ifft_plan
        u0 = rde.u0
        u0_hat = fft_plan*u0
        u0_hat_hat = ifft_plan*u0_hat
        success = success && (u0_hat_hat ≈ u0)
    end
    success
end

