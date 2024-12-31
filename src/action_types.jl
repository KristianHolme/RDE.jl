"""
    AbstractActionType

Abstract supertype for all RDE action types. Subtypes define different ways to control
the RDE system through pressure and area parameters.
"""
abstract type AbstractActionType end

"""
    ScalarPressureAction <: AbstractActionType

Action type for controlling the RDE with a single scalar pressure value.

# Fields
- `N::Int`: Number of grid points (set via `set_N!`)
"""
mutable struct ScalarPressureAction <: AbstractActionType 
    N::Int #number of grid points
end

"""
    ScalarPressureAction(; N::Int=0)

Construct a scalar pressure action type.

# Keywords
- `N::Int=0`: Number of grid points (can be set later via `set_N!`)
"""
ScalarPressureAction(;N::Int=0) = ScalarPressureAction(N)

"""
    ScalarAreaScalarPressureAction <: AbstractActionType

Action type for controlling both area and pressure with scalar values.

# Fields
- `N::Int`: Number of grid points (set via `set_N!`)
"""
mutable struct ScalarAreaScalarPressureAction <: AbstractActionType 
    N::Int #number of grid points
end

"""
    ScalarAreaScalarPressureAction(; N::Int=0)

Construct a scalar area and pressure action type.

# Keywords
- `N::Int=0`: Number of grid points (can be set later via `set_N!`)
"""
ScalarAreaScalarPressureAction(;N::Int=0) = ScalarAreaScalarPressureAction(N)

"""
    VectorPressureAction <: AbstractActionType

Action type for controlling pressure with multiple values across different sections.

# Fields
- `number_of_sections::Int`: Number of control sections
- `N::Int`: Number of grid points (must be divisible by `number_of_sections`)
"""
mutable struct VectorPressureAction <: AbstractActionType 
    number_of_sections::Int #number of sections 
    N::Int #number of grid points
end

"""
    VectorPressureAction(; number_of_sections=1, N=0)

Construct a vector pressure action type.

# Keywords
- `number_of_sections::Int=1`: Number of control sections
- `N::Int=0`: Number of grid points (can be set later via `set_N!`)
"""
VectorPressureAction(;number_of_sections=1, N=0) = VectorPressureAction(number_of_sections, N)

"""
    n_actions(at::AbstractActionType) -> Int

Get the number of action dimensions for a given action type.

# Arguments
- `at::AbstractActionType`: Action type to query

# Returns
- `Int`: Number of action dimensions

# Examples
```julia
n_actions(ScalarPressureAction()) # Returns 1
n_actions(ScalarAreaScalarPressureAction()) # Returns 2
n_actions(VectorPressureAction(number_of_sections=4)) # Returns 4
```
"""
function n_actions(at::ScalarPressureAction) 
    return 1
end

function n_actions(at::ScalarAreaScalarPressureAction)
    return 2
end

function n_actions(at::VectorPressureAction)
    return at.number_of_sections
end

"""
    set_N!(action_type::AbstractActionType, N::Int) -> AbstractActionType

Set the number of grid points for an action type.

# Arguments
- `action_type::AbstractActionType`: Action type to modify
- `N::Int`: Number of grid points

# Returns
- Modified action type

# Throws
- `AssertionError`: For `VectorPressureAction` if N is not divisible by number_of_sections

# Example
```julia
action_type = ScalarPressureAction()
set_N!(action_type, 100)
```
"""
function set_N!(action_type::AbstractActionType, N::Int)
    if action_type isa VectorPressureAction
        @assert N % action_type.number_of_sections == 0 "N ($N) must be divisible by number_of_sections ($(action_type.number_of_sections))"
    end
    setfield!(action_type, :N, N)
    return action_type
end

"""
    get_standard_normalized_actions(action_type::AbstractActionType, action) -> Vector{Vector{T}}

Convert normalized actions to standard form [zeros(N), pressure_actions].

# Arguments
- `action_type::AbstractActionType`: Type of action
- `action`: Raw action values

# Returns
- `Vector{Vector{T}}`: Standard form [zeros(N), pressure_actions]

# Throws
- `AssertionError`: If action_type.N is not set or action dimensions don't match

# Note
This is a fallback method that issues a warning. Each action type should implement
its own specific version.
"""
function get_standard_normalized_actions(action_type::AbstractActionType, action)
    @assert action_type.N > 0 "Action type N not set"
    @warn "get_standard_normalized_actions is not implemented for action_type $action_type"
end

"""
    get_standard_normalized_actions(action_type::ScalarAreaScalarPressureAction, action)

Convert scalar area and pressure actions to standard form.

# Arguments
- `action_type::ScalarAreaScalarPressureAction`: Action type
- `action`: 2-element vector [area, pressure]

# Returns
- `Vector{Vector{T}}`: Standard form [area_actions, pressure_actions]
"""
function get_standard_normalized_actions(action_type::ScalarAreaScalarPressureAction, action)
    @assert action_type.N > 0 "Action type N not set"
    @assert length(action) == 2
    return [fill(action[1], action_type.N), fill(action[2], action_type.N)]
end

"""
    get_standard_normalized_actions(action_type::ScalarPressureAction, action)

Convert scalar pressure action to standard form.

# Arguments
- `action_type::ScalarPressureAction`: Action type
- `action`: Single pressure value

# Returns
- `Vector{Vector{T}}`: Standard form [zeros(N), pressure_actions]
"""
function get_standard_normalized_actions(action_type::ScalarPressureAction, action)
    @assert action_type.N > 0 "Action type N not set"
    @assert length(action) == 1
    if isa(action, AbstractArray)
        return [zeros(action_type.N), ones(action_type.N) .* action[1]]
    else
        return [zeros(action_type.N), ones(action_type.N) .* action]
    end
end

"""
    get_standard_normalized_actions(action_type::VectorPressureAction, action)

Convert vector pressure actions to standard form.

# Arguments
- `action_type::VectorPressureAction`: Action type
- `action`: Vector of pressure values for each section

# Returns
- `Vector{Vector{T}}`: Standard form [zeros(N), pressure_actions]

# Note
Pressure values are repeated for each point within their respective sections.
"""
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