"""
    init_cache!(method::AbstractMethod, params::RDEParam{T}, dx::T) where {T<:AbstractFloat}

Initialize the cache for a given method with the problem parameters.
"""
function init_cache!(method::FiniteVolumeMethod{T, L}, params::RDEParam{T}, dx::T) where {T <: AbstractFloat, L <: AbstractLimiter}
    return method.cache = FVCache{T}(params)
end

function _reset_cache!(cache::AbstractRDECache; τ_smooth::AbstractFloat, params::RDEParam)
    cache.τ_smooth = τ_smooth
    fill!(cache.u_p_previous, params.u_p)
    fill!(cache.u_p_current, params.u_p)
    fill!(cache.s_previous, params.s)
    fill!(cache.s_current, params.s)
    return nothing
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
    @turbo for i in 2:(N - 1)
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
    @turbo for i in 2:(N - 1)
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
    @turbo for i in 2:(N - 1)
        λ_xx[i] = (λ[i + 1] - 2 * λ[i] + λ[i - 1]) * inv_dx2
    end
    λ_xx[N] = (λ[1] - 2 * λ[N] + λ[N - 1]) * inv_dx2

    # ----------------------------
    # MUSCL reconstruction (limited slopes)
    # ----------------------------
    _limited_slope!(σ, u, method.limiter)

    # reconstruct interface states at i+1/2 stored at index i (1..N)
    @turbo for i in 1:(N - 1)
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
    @turbo for i in 1:N
        a = max(abs(UL[i]), abs(UR[i]))
        Fl = T(0.5) * UL[i] * UL[i]
        Fr = T(0.5) * UR[i] * UR[i]
        F̂[i] = T(0.5) * (Fl + Fr) - T(0.5) * a * (UR[i] - UL[i])
    end

    inv_dx = one(T) / dx
    # conservative divergence: -(F_{i+1/2} - F_{i-1/2})/dx
    adv[1] = -(F̂[1] - F̂[N]) * inv_dx
    @turbo for i in 2:N
        adv[i] = -(F̂[i] - F̂[i - 1]) * inv_dx
    end

    return nothing
end
