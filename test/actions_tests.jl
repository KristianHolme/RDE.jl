
@test begin
    at = VectorPressureAction(;N=16, number_of_sections=2)
    action = [1.0, 2.0]
    all(get_standard_normalized_actions(at, action) .≈ [zeros(16), vcat(ones(8)*1.0, ones(8)*2.0)])
end

@test begin
    at = ScalarPressureAction(;N=16)
    action = 1.0
    all(get_standard_normalized_actions(at, action) .≈ [zeros(16), ones(16)*1.0])
end

@test begin
    at = ScalarAreaScalarPressureAction(;N=16)
    action = [1.0, 2.0]
    all(get_standard_normalized_actions(at, action) .≈ [fill(1.0, 16), fill(2.0, 16)])
end
