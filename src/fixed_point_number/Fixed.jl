# 32-bit fixed point; parameter `f` is the number of fraction bits

import Base: ==, <, <=, -, +, *, /, ~, isapprox,
             convert, promote_rule, show, showcompact, isinteger, abs, decompose,
             isnan, isinf, isfinite,
             zero, oneunit, one, typemin, typemax, realmin, realmax, eps, sizeof, reinterpret,
             float, trunc, round, floor, ceil, bswap,
             div, fld, rem, mod, mod1, fld1, min, max, minmax,
             start, next, done, rand

struct Fixed{T <: Signed,δ,f}
    i::T

    # constructor for manipulating the representation;
    # selected by passing an extra dummy argument
    Fixed{T, δ, f}(i::Integer, _) where {T,δ,f} = new{T,δ,f}(i % T)
    Fixed{T, δ, f}(x) where {T,δ,f} = convert(Fixed{T,δ,f}, x)
    Fixed{T, δ, f}(x::Fixed{T,δ,f}) where {T,δ,f} = x
end

reinterpret(::Type{Fixed{T,δ,f}}, x::T) where {T <: Signed,δ,f} = Fixed{T,δ,f}(x, 0)

# helper for type
widen1(::Type{Int8})   = Int16
widen1(::Type{UInt8})  = UInt16
widen1(::Type{Int16})  = Int32
widen1(::Type{UInt16}) = UInt32
widen1(::Type{Int32})  = Int64
widen1(::Type{UInt32}) = UInt64
widen1(::Type{Int64})  = Int128
widen1(::Type{UInt64}) = UInt128
widen1(::Type{UInt128}) = UInt128
widen1(x::Integer) = x % widen1(typeof(x))
bits_diff(::Type{Int8}, ::Type{Int16}) = 8
bits_diff(::Type{Int16}, ::Type{Int32}) = 16
bits_diff(::Type{Int32}, ::Type{Int64}) = 32

# basic
function zero(::Type{Fixed{T,σ,f}}) where {T<:Integer,σ,f}
    Fixed{T,σ,f}(0,0)
end
# multiplication
function qround(T::Type{<:Integer}, x::Integer)
    if x > typemax(T)
        return typemax(T)
    elseif x < typemin(T)
        return typemin(T)
    else
        d = bits_diff(T, typeof(x))
        x_floor = (x>>d)<<d
        r = x - x_floor
        if rand() > float(r)^2^(-d)
            return T(x_floor>>d)
        else return T(x_floor+1>>d)
        end
    end
end

# convert
function convert(::Type{Fixed{T, σ, f}}, x::AbstractFloat) where {T<:Integer, σ, f}
    # x /= σ
    if x > typemax(T)
        return typemax(T)
    elseif x < typemin(T)
        return typemin(T)
    else
        x_floor = floor(T, trunc(widen1(T),x)<<f + rem(x,1)*(one(widen1(T))<<f))
        r = x - x_floor
        if rand() > r*2.0^(-f)
            return Fixed{T,σ,f}(x_floor+1, 0)
        else return Fixed{T,σ,f}(x_floor, 0)
        end
    end
end

function array_cast(fix::Type{Fixed{T,σ,f}}, A::Array{<:AbstractFloat}) where {T<:Integer,σ,f}
    B = zeros(A, Fixed{T,σ,f})
    for i=1:length(A)
        B[i] = Fixed{T,σ,f}(A[i])
    end
    return B
end
# convert(::Type{Fixed{T,δ,f}}, x::Integer) where {T,σ,f} = Fixed{T,f}(round(T, convert(widen1(T),x/δ)<<f),0)
