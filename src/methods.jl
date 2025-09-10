"""
    init_cache!(method::AbstractMethod, params::RDEParam{T}, dx::T) where {T<:AbstractFloat}

Initialize the cache for a given method with the problem parameters.
"""
function init_cache!(method::PseudospectralMethod{T}, params::RDEParam{T}, dx::T) where {T <: AbstractFloat}
    return method.cache = PseudospectralRDECache{T}(params, dealias = method.dealias)
end

function init_cache!(method::FiniteDifferenceMethod{T}, params::RDEParam{T}, dx::T) where {T <: AbstractFloat}
    return method.cache = FDRDECache{T}(params, dx)
end

"""
    create_dealiasing_vector(params::RDEParam{T}) where {T<:AbstractFloat}

Create a 2/3 rule dealiasing filter for pseudospectral computations.
"""
function create_dealiasing_vector(params::RDEParam{T}) where {T <: AbstractFloat}
    N = params.N
    N_complex = div(N, 2) + 1
    k = collect(0:(N_complex - 1))
    k_cutoff = div(N, 3)
    dealiasing = @. ifelse(k <= k_cutoff, one(T), zero(T))
    return dealiasing
end

"""
    create_spectral_derivative_arrays(params::RDEParam{T}) where {T<:AbstractFloat}

Create arrays for spectral derivatives: wavenumbers for first (ik) and second (k2) derivatives.
"""
function create_spectral_derivative_arrays(params::RDEParam{T}) where {T <: AbstractFloat}
    N = params.N
    L = params.L
    N_complex = div(N, 2) + 1
    k = T(2π / L) .* Vector(0:(N_complex - 1))
    ik = 1im .* k
    ik[end] = zero(T)
    k2 = k .^ 2
    return ik, k2
end

"""
    calc_derivatives!(u::AbstractArray{T}, λ::AbstractArray{T}, method::PseudospectralMethod{T}) where {T <: AbstractFloat}

Calculate spatial derivatives using pseudospectral method with FFT.

# Arguments
- `u`: Velocity field
- `λ`: Reaction progress
- `method`: Pseudospectral method containing FFT plans and workspace arrays

# Implementation Notes
- Uses in-place FFT operations
- Applies dealiasing filter in spectral space
- Computes first and second derivatives for u
- Computes second derivative for λ
- Handles periodic boundary conditions automatically
"""
function calc_derivatives!(u::AbstractArray{T}, λ::AbstractArray{T}, method::PseudospectralMethod{T}) where {T <: AbstractFloat}
    cache = method.cache
    ik = cache.ik             # Complex array of size N÷2+1
    k2 = cache.k2             # Real array of size N÷2+1
    dealias_filter = cache.dealias_filter  # Real array of size N÷2+1
    # Preallocated arrays
    u_hat = cache.u_hat            # Complex array of size N÷2+1
    u_x_hat = cache.u_x_hat        # Complex array of size N÷2+1
    u_xx_hat = cache.u_xx_hat      # Complex array of size N÷2+1
    u_x = cache.u_x                # Real array of size N
    u_xx = cache.u_xx              # Real array of size N
    λ_hat = cache.λ_hat            # Complex array of size N÷2+1
    λ_xx_hat = cache.λ_xx_hat      # Complex array of size N÷2+1
    λ_xx = cache.λ_xx

    # Transform to spectral space (in-place)
    mul!(u_hat, cache.fft_plan, u)  # Apply fft_plan to u, store in u_hat
    mul!(λ_hat, cache.fft_plan, λ)

    # Compute derivatives in spectral space (with dealiasing)
    @. u_x_hat = ik * u_hat * dealias_filter
    @. u_xx_hat = -k2 * u_hat * dealias_filter
    @. λ_xx_hat = -k2 * λ_hat * dealias_filter

    # Inverse FFT to get derivatives in physical space (in-place)
    mul!(u_x, cache.ifft_plan, u_x_hat)
    mul!(u_xx, cache.ifft_plan, u_xx_hat)
    mul!(λ_xx, cache.ifft_plan, λ_xx_hat)
    return nothing
end

"""
    calc_derivatives!(u::AbstractArray{T}, λ::AbstractArray{T}, method::FiniteDifferenceMethod{T}) where {T <: AbstractFloat}

Calculate spatial derivatives using finite difference method.

# Arguments
- `u`: Velocity field
- `λ`: Reaction progress
- `method`: Finite difference method containing grid parameters and workspace arrays

# Implementation Notes
- Uses second-order central differences
- Handles periodic boundary conditions explicitly
- Computes first and second derivatives for u
- Computes second derivative for λ
- Optimized with @turbo macro for performance
"""
function calc_derivatives!(u::AbstractArray{T}, λ::AbstractArray{T}, method::FiniteDifferenceMethod{T}) where {T <: AbstractFloat}
    cache = method.cache
    dx = cache.dx
    N = cache.N
    inv_2dx = T(0.5) / dx
    inv_dx2 = one(T) / (dx^2)

    # Preallocated arrays
    u_x = cache.u_x
    u_xx = cache.u_xx
    λ_xx = cache.λ_xx

    # Compute u_x using central differences with periodic boundary conditions
    u_x[1] = (u[2] - u[N]) * inv_2dx
    @turbo for i in 2:(N - 1)
        u_x[i] = (u[i + 1] - u[i - 1]) * inv_2dx
    end
    u_x[N] = (u[1] - u[N - 1]) * inv_2dx

    # Compute u_xx using central differences with periodic boundary conditions
    u_xx[1] = (u[2] - 2 * u[1] + u[N]) * inv_dx2
    @turbo for i in 2:(N - 1)
        u_xx[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) * inv_dx2
    end
    u_xx[N] = (u[1] - 2 * u[N] + u[N - 1]) * inv_dx2

    # Compute λ_xx using central differences with periodic boundary conditions
    λ_xx[1] = (λ[2] - 2 * λ[1] + λ[N]) * inv_dx2
    @turbo for i in 2:(N - 1)
        λ_xx[i] = (λ[i + 1] - 2 * λ[i] + λ[i - 1]) * inv_dx2
    end
    λ_xx[N] = (λ[1] - 2 * λ[N] + λ[N - 1]) * inv_dx2

    return nothing
end

function reset_cache!(cache::AbstractRDECache; τ_smooth::AbstractFloat, params::RDEParam)
    cache.τ_smooth = τ_smooth
    cache.u_p_previous = fill(params.u_p, params.N)
    cache.u_p_current = fill(params.u_p, params.N)
    cache.s_previous = fill(params.s, params.N)
    cache.s_current = fill(params.s, params.N)
    return nothing
end

"""
    init_cache!(method::UpwindMethod{T}, params::RDEParam{T}, dx::T) where {T<:AbstractFloat}

Initialize the cache for the upwind method with the problem parameters.
"""
function init_cache!(method::UpwindMethod{T}, params::RDEParam{T}, dx::T) where {T <: AbstractFloat}
    N = params.N
    return method.cache = UpwindRDECache{T}(
        Vector{T}(undef, N),  # u_x
        Vector{T}(undef, N),  # u_xx
        Vector{T}(undef, N),  # λ_xx
        Vector{T}(undef, N),  # ωu
        Vector{T}(undef, N),  # ξu
        Vector{T}(undef, N),  # βu
        dx,                   # dx
        N,                    # N
        fill(params.u_p, N),  # u_p_current
        fill(params.u_p, N),  # u_p_previous
        T(1),                 # τ_smooth
        fill(params.s, N),    # s_previous
        fill(params.s, N),    # s_current
        T(0),                 # control_time
        fill(params.u_p, N),  # u_p_t
        fill(params.s, N),    # s_t
        fill(params.u_p, N),  # u_p_t_shifted
        fill(params.s, N),    # s_t_shifted
    )
end

"""
    init_cache!(method::FiniteVolumeMethod{T}, params::RDEParam{T}, dx::T) where {T<:AbstractFloat}

Initialize the cache for the finite-volume method with the problem parameters.
"""
function init_cache!(method::FiniteVolumeMethod{T, L}, params::RDEParam{T}, dx::T) where {T <: AbstractFloat, L <: AbstractLimiter}
    return method.cache = FVCache{T}(params, dx)
end

"""
    calc_derivatives!(u::AbstractArray{T}, λ::AbstractArray{T}, method::FiniteVolumeMethod{T, L}) where {T <: AbstractFloat, L <: AbstractLimiter}

Compute conservative advective residual using MUSCL reconstruction with a slope
limiter and Rusanov flux, plus second-order central diffusion terms for `u` and `λ`.
Stores:
- `cache.adv[i] = -(F̂_{i+1/2} - F̂_{i-1/2})/dx`
- `cache.u_xx`, `cache.λ_xx`
"""
@inline function _limited_slope!(σ::AbstractVector{T}, u::AbstractVector{T}, limiter::MinmodLimiter) where {T <: AbstractFloat}
    N = length(u)
    @inbounds begin
        # boundary i=1
        Δp = u[2] - u[1]
        Δm = u[1] - u[N]
        s = (sign(Δp) + sign(Δm)) * T(0.5)
        σ[1] = s * min(abs(Δm), abs(Δp))
    end
    @inbounds for i in 2:(N - 1)
        Δp = u[i + 1] - u[i]
        Δm = u[i] - u[i - 1]
        s = (sign(Δp) + sign(Δm)) * T(0.5)
        σ[i] = s * min(abs(Δm), abs(Δp))
    end
    @inbounds begin
        # boundary i=N
        Δp = u[1] - u[N]
        Δm = u[N] - u[N - 1]
        s = (sign(Δp) + sign(Δm)) * T(0.5)
        σ[N] = s * min(abs(Δm), abs(Δp))
    end
    return nothing
end

@inline function _limited_slope!(σ::AbstractVector{T}, u::AbstractVector{T}, ::MCLimiter) where {T <: AbstractFloat}
    N = length(u)
    @inbounds begin
        # boundary i=1
        Δp = u[2] - u[1]
        Δm = u[1] - u[N]
        s = (sign(Δp) + sign(Δm)) * T(0.5)
        σc = T(0.5) * (Δm + Δp)
        σ2m = T(2.0) * Δm
        σ2p = T(2.0) * Δp
        σ[1] = s * min(abs(σc), abs(σ2m), abs(σ2p))
    end
    @inbounds for i in 2:(N - 1)
        Δp = u[i + 1] - u[i]
        Δm = u[i] - u[i - 1]
        s = (sign(Δp) + sign(Δm)) * T(0.5)
        σc = T(0.5) * (Δm + Δp)
        σ2m = T(2.0) * Δm
        σ2p = T(2.0) * Δp
        σ[i] = s * min(abs(σc), abs(σ2m), abs(σ2p))
    end
    @inbounds begin
        # boundary i=N
        Δp = u[1] - u[N]
        Δm = u[N] - u[N - 1]
        s = (sign(Δp) + sign(Δm)) * T(0.5)
        σc = T(0.5) * (Δm + Δp)
        σ2m = T(2.0) * Δm
        σ2p = T(2.0) * Δp
        σ[N] = s * min(abs(σc), abs(σ2m), abs(σ2p))
    end
    return nothing
end

function calc_derivatives!(u::AbstractArray{T}, λ::AbstractArray{T}, method::FiniteVolumeMethod{T, L}) where {T <: AbstractFloat, L <: AbstractLimiter}
    cache = method.cache
    dx = cache.dx
    N = cache.N

    σ = cache.σ
    UL = cache.UL
    UR = cache.UR
    F̂ = cache.F̂
    adv = cache.adv

    # ----------------------------
    # Diffusion (second derivatives)
    # ----------------------------
    inv_dx2 = one(T) / (dx^2)
    u_xx = cache.u_xx
    λ_xx = cache.λ_xx
    u_xx[1] = (u[2] - 2 * u[1] + u[N]) * inv_dx2
    @turbo for i in 2:(N - 1)
        u_xx[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) * inv_dx2
    end
    u_xx[N] = (u[1] - 2 * u[N] + u[N - 1]) * inv_dx2

    λ_xx[1] = (λ[2] - 2 * λ[1] + λ[N]) * inv_dx2
    @inbounds for i in 2:(N - 1)
        λ_xx[i] = (λ[i + 1] - 2 * λ[i] + λ[i - 1]) * inv_dx2
    end
    λ_xx[N] = (λ[1] - 2 * λ[N] + λ[N - 1]) * inv_dx2

    # ----------------------------
    # MUSCL reconstruction (limited slopes)
    # ----------------------------
    _limited_slope!(σ, u, method.limiter)

    # reconstruct interface states at i+1/2 stored at index i (1..N)
    @inbounds for i in 1:(N - 1)
        UL[i] = u[i] + T(0.5) * σ[i]
        UR[i] = u[i + 1] - T(0.5) * σ[i + 1]
    end
    @inbounds begin
        # periodic wrap for the last interface N+1/2 -> index N
        UL[N] = u[N] + T(0.5) * σ[N]
        UR[N] = u[1] - T(0.5) * σ[1]
    end

    # ----------------------------
    # Rusanov (Lax–Friedrichs) numerical flux for f(u) = 0.5*u^2
    # ----------------------------
    @inbounds for i in 1:N
        a = max(abs(UL[i]), abs(UR[i]))
        Fl = T(0.5) * UL[i] * UL[i]
        Fr = T(0.5) * UR[i] * UR[i]
        F̂[i] = T(0.5) * (Fl + Fr) - T(0.5) * a * (UR[i] - UL[i])
    end

    inv_dx = one(T) / dx
    # conservative divergence: -(F_{i+1/2} - F_{i-1/2})/dx
    adv[1] = -(F̂[1] - F̂[N]) * inv_dx
    @inbounds for i in 2:N
        adv[i] = -(F̂[i] - F̂[i - 1]) * inv_dx
    end

    return nothing
end

"""
    calc_derivatives!(u::AbstractArray{T}, λ::AbstractArray{T}, method::UpwindMethod{T}) where {T <: AbstractFloat}

Calculate spatial derivatives using upwinding scheme appropriate for the RDE system.

# Arguments
- `u`: Energy/pressure-like field (always positive)
- `λ`: Reaction progress
- `method`: Upwind method containing grid parameters and workspace arrays

# Implementation Notes
- Uses backward differences for upwinding since the advection term is `-u*u_x` with `u > 0`
- This creates waves propagating from left to right (positive x-direction)
- Supports both first-order and second-order upwinding based on the `order` parameter
- Uses second-order central differences for second derivatives
- Handles periodic boundary conditions explicitly
- Computes first and second derivatives for u
- Computes second derivative for λ
- Optimized with @turbo macro for performance
"""
function calc_derivatives!(u::AbstractArray{T}, λ::AbstractArray{T}, method::UpwindMethod{T}) where {T <: AbstractFloat}
    cache = method.cache
    dx = cache.dx
    N = cache.N
    inv_dx = one(T) / dx
    inv_dx2 = one(T) / (dx^2)
    order = method.order

    # Preallocated arrays
    u_x = cache.u_x
    u_xx = cache.u_xx
    λ_xx = cache.λ_xx

    # Compute u_x using upwind differences appropriate for -u*u_x term
    # with periodic boundary conditions
    if order == 1
        # First-order upwind (backward differences)
        @inbounds u_x[1] = (u[1] - u[N]) * inv_dx
        @turbo for i in 2:N
            u_x[i] = (u[i] - u[i - 1]) * inv_dx
        end
    else
        # Second-order upwind (3-point stencil)
        # boundaries
        @inbounds u_x[1] = (3 * u[1] - 4 * u[N] + u[N - 1]) * (T(0.5) * inv_dx)
        @inbounds u_x[2] = (3 * u[2] - 4 * u[1] + u[N]) * (T(0.5) * inv_dx)
        @turbo for i in 3:N
            u_x[i] = (3 * u[i] - 4 * u[i - 1] + u[i - 2]) * (T(0.5) * inv_dx)
        end
    end

    # Compute u_xx using central differences with periodic boundary conditions
    u_xx[1] = (u[2] - 2 * u[1] + u[N]) * inv_dx2
    @turbo for i in 2:(N - 1)
        u_xx[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) * inv_dx2
    end
    u_xx[N] = (u[1] - 2 * u[N] + u[N - 1]) * inv_dx2

    # Compute λ_xx using central differences with periodic boundary conditions
    λ_xx[1] = (λ[2] - 2 * λ[1] + λ[N]) * inv_dx2
    @turbo for i in 2:(N - 1)
        λ_xx[i] = (λ[i + 1] - 2 * λ[i] + λ[i - 1]) * inv_dx2
    end
    λ_xx[N] = (λ[1] - 2 * λ[N] + λ[N - 1]) * inv_dx2

    return nothing
end
