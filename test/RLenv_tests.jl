@test begin
    params = RDEParam(;N=16, tmax = 0.01);
    prob = RDEProblem(params);
    sum(abs.(prob.u0)) > 0.01
end

@test begin
    import POMDPs: MDP
    mdp = convert(MDP, RDEEnv());
    true
end