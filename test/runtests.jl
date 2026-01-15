using RDE
using Test
using TestItems
using TestItemRunner

@testitem "Code quality (Aqua.jl)" tags = [:quality] begin
    using Aqua, RDE
    Aqua.test_all(RDE)
end

@testitem "Code linting (JET.jl)" tags = [:quality] begin
    using JET, RDE
    report = JET.report_package(RDE; target_modules = (RDE,), toplevel_logger = nothing)
    @test isempty(JET.get_reports(report))
end

@run_package_tests
