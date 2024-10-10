using RDE, Test, TestItems, TestItemRunner

@testitem "basic functionality" begin
    include("RLenv_tests.jl")
    include("solver_tests.jl")
end

@testitem "utils" begin
    include("utils_tests.jl")
end

@run_package_tests