# 32-bit fixed point; parameter `f` is the number of fraction bits

import Base: ==, <, <=, -, +, *, /, ~, isapprox,
             convert, promote_rule, show, showcompact, isinteger, abs, decompose,
             isnan, isinf, isfinite,
             zero, oneunit, one, typemin, typemax, realmin, realmax, eps, sizeof, reinterpret,
             float, trunc, round, floor, ceil, bswap,
             div, fld, rem, mod, mod1, fld1, min, max, minmax,
             start, next, done, rand

struct Fixed{T <: Signed,f}
    i::T

    # constructor for manipulating the representation;
    # selected by passing an extra dummy argument
    function Fixed{T, f}(i::Integer, _) where {T,f}
        if i > typemax(T) new{T,f}(typemax(T))
        elseif i < typemin(T) new{T,f}(typemin(T))
        else new{T,f}(i)
        end
    end
    Fixed{T,f}(x::Int) where {T,f} = convert(Fixed{T,f}, x)
    Fixed{T,f}(x::AbstractFloat) where {T,f} = quantize(Fixed{T,f}, x)
    Fixed{T,f}(x::Fixed{T,f}) where {T,f} = x
end

struct BlockFixed{T<:Signed, σ}
    i::T
    function BlockFixed{T,σ}(i::Integer, _) where {T, σ}
        if i > typemax(T) new{T,f}(typemax(T))
        elseif i < typemin(T) new{T,f}(typemin(T))
        else new{T,f}(i)
        end
    end
end

type BlockFixedArray{T<:Signed, σ}
    A::Array{T}
    function BlockFixedArray{T, σ}(A::Array{N}) where {T<:Signed, σ, N<:Integer}
        new(A)
    end
end

reinterpret(::Type{Fixed{T,f}}, x::T) where {T <: Signed,f} = Fixed{T,f}(x, 0)

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
function zero(::Type{Fixed{T,f}}) where {T<:Integer,f}
    Fixed{T,f}(0,0)
end

# quantize
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

function quantize(::Type{Fixed{T, f}}, x::AbstractFloat) where {T<:Integer,f}
    # x /= σ
    if x >= (typemax(T)+1)>>f
        println("x is $x")
        return Fixed{T,f}(typemax(T),0)
    elseif x < typemin(T)>>f
        return Fixed{T,f}(typemin(T),0)
    else
        x_floor = floor(T, trunc(widen1(T),x)<<f + rem(x,1)*(one(widen1(T))<<f))
        r = x*2^f - x_floor
        p = rand()
        if p > r
            return Fixed{T,f}(x_floor+1, 0)
        else return Fixed{T,f}(x_floor, 0)
        end
    end
end

function array_quantize(fix::Type{Fixed{T,f}}, A::Array{<:AbstractFloat}) where {T<:Integer,f}
    B = zeros(A, Fixed{T,f})
    for i=1:length(A)
        B[i] = Fixed{T,f}(A[i])
    end
    return B
end

# convert(::Type{Fixed{T,δ,f}}, x::Integer) where {T,σ,f} = Fixed{T,f}(round(T, convert(widen1(T),x/δ)<<f),0)
