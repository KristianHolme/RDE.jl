
π_const = ConstantRDEPolicy(RDEEnv());

constPolRunData = run_policy(π_const, RDEEnv();tmax=26.0);

animate_policy_data(constPolRunData, RDEEnv();fname="constant_policy")




