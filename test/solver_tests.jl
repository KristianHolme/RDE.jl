@test begin
    prob = RDEProblem(RDEParam(;N=32, tmax = 0.01));
    solve_pde!(prob);
    true
end

@test begin
    ConstPolicy = ConstantRDEPolicy();
    data = run_policy(ConstPolicy, RDEEnv();tmax = 1.0);
    data isa PolicyRunData
end