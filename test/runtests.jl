using RDE, Test, TestItems, TestItemRunner


@testitem "solver" begin
    include("solver_tests.jl")
end

@testitem "env" begin
    include("RLenv_tests.jl")
end

@testitem "utils" begin
    include("utils_tests.jl")
end

@testitem "Types" begin
    include("types_tests.jl")
end

@run_package_tests