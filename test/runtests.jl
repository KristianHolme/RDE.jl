using RDE
using Test
using TestItems
using TestItemRunner

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

@testitem "Actions" begin
    include("actions_tests.jl")
end

@run_package_tests
