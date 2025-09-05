using RDE
using Test
using TestItems
using TestItemRunner

@testitem "solver" begin
    include("solver_tests.jl")
end

@testitem "utils" begin
    include("utils_tests.jl")
end

@testitem "types" begin
    include("types_tests.jl")
end

@testitem "reset" begin
    include("reset_tests.jl")
end

@run_package_tests
