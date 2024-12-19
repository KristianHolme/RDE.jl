abstract type AbstractActionType end

mutable struct ScalarPressureAction <: AbstractActionType 
    N::Int #number of grid points
end
ScalarPressureAction(;N::Int=0) = ScalarPressureAction(N)

mutable struct ScalarAreaScalarPressureAction <: AbstractActionType 
    N::Int #number of grid points
end
# ScalarAreaScalarPressureAction() = ScalarAreaScalarPressureAction(0) #TODO remove
ScalarAreaScalarPressureAction(;N::Int=0) = ScalarAreaScalarPressureAction(N)

mutable struct VectorPressureAction <: AbstractActionType 
    number_of_sections::Int #number of sections 
    N::Int #number of grid points
end
VectorPressureAction(;number_of_sections=1, N=0) = VectorPressureAction(number_of_sections, N)

function n_actions(at::ScalarPressureAction) 
    return 1
end

function n_actions(at::ScalarAreaScalarPressureAction)
    return 2
end

function n_actions(at::VectorPressureAction)
    return at.number_of_sections
end
# Helper function to set N for any action type
function set_N!(action_type::AbstractActionType, N::Int)
    if action_type isa VectorPressureAction
        @assert N % action_type.number_of_sections == 0 "N ($N) must be divisible by number_of_sections ($(action_type.number_of_sections))"
    end
    setfield!(action_type, :N, N)
    return action_type
end

function get_standard_normalized_actions(action_type::AbstractActionType, action)
    @assert action_type.N > 0 "Action type N not set"
    @warn "get_standard_normalized_actions is not implemented for action_type $action_type"
end

function get_standard_normalized_actions(action_type::ScalarAreaScalarPressureAction, action)
    @assert action_type.N > 0 "Action type N not set"
    @assert length(action) == 2
    return [fill(action[1], action_type.N), fill(action[2], action_type.N)]
end

function get_standard_normalized_actions(action_type::ScalarPressureAction, action)
    @assert action_type.N > 0 "Action type N not set"
    @assert length(action) == 1
    return [zeros(action_type.N), ones(action_type.N)*action]
end

function get_standard_normalized_actions(action_type::VectorPressureAction, action)
    @assert action_type.N > 0 "Action type N not set"
    @assert length(action) == action_type.number_of_sections
    @assert action_type.N % action_type.number_of_sections == 0
    
    # Calculate how many points per section
    points_per_section = action_type.N ÷ action_type.number_of_sections
    
    # Initialize pressure actions array
    pressure_actions = zeros(action_type.N)
    
    # Fill each section with its corresponding action value
    for i in 1:action_type.number_of_sections
        start_idx = (i-1) * points_per_section + 1
        end_idx = i * points_per_section
        pressure_actions[start_idx:end_idx] .= action[i]
    end
    return [zeros(action_type.N), pressure_actions]
end