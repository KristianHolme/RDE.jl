using POMDPs
@test begin
    params = RDEParam(;N=16, tmax = 0.01);
    prob = RDEProblem(params);
    sum(abs.(prob.u0)) > 0.01
end

@test begin
    mdp = convert(MDP, RDEEnv())
    pomdp = convert(POMDP, RDEEnv())
    true
end

@test begin
    ConstPolicy = ConstantRDEPolicy();
    data = run_policy(ConstPolicy, RDEEnv(RDEParam(;N=16, tmax = 0.1)));
    data isa PolicyRunData
end
