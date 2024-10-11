@test RDE.periodic_simpsons_rule(zeros(16), 0.5) ≈ 0.0

@test RDE.energy_balance(vcat(zeros(128), ones(128)), RDEParam()) ≈ 0.0