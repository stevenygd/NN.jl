import Base: +, -, *, float


type BlockFixedArray{T<:Signed}
    arr::Array{T}
    σ::Float64
    function BlockFixedArray{T}(arr::Array{N}, σ::Float64) where {T<:Signed, N<:Integer}
        new(arr, σ)
    end
end

function *(A::BlockFixedArray{T}, B::BlockFixedArray{T}) where {T<:Signed}
    nT = widen(T)
    arr = Array{nT}(A.arr) * Array{nT}(B.arr)
    σ = A.σ*B.σ
    BlockFixedArray{nT}(arr, σ)
end

function -(A::BlockFixedArray{T}, B::BlockFixedArray{T}) where {T<:Signed}
    @assert A.σ == B.σ # only allow same scale factor for now
    BlockFixedArray{nT}(A.arr-B.arr, A.σ)
end

function +(A::BlockFixedArray{T}, B::BlockFixedArray{T}) where {T<:Signed}
    @assert A.σ == B.σ # only allow same scale factor for now
    BlockFixedArray{nT}(A.arr+B.arr, A.σ)
end

function float(A::BlockFixedArray)
    float(A.arr)*A.σ
end

function quantize(T::Type{<:Integer}, σ::Float64, x::AbstractFloat)
    x /= σ
    if x >= (typemax(T)+1)
        return typemax(T)
    elseif x < typemin(T)
        return typemin(T)
    else
        x_floor = floor(T, x)
        r = x - x_floor
        p = rand()
        if p <= r
            return x_floor+1
        else return x_floor
        end
    end
end

function quantize(::Type{BlockFixedArray{T}}, σ::Float64, A::Array{N}) where {T<:Signed, N<:AbstractFloat}
    map(a->quantize(T, σ, a), A)
end

# helper for type
widen1(::Type{Int8})   = Int16
widen1(::Type{Int16})  = Int32
widen1(::Type{Int32})  = Int64
widen1(::Type{Int64})  = Int128
widen1(x::Integer) = x % widen1(typeof(x))
bits_diff(::Type{Int8}, ::Type{Int16}) = 8
bits_diff(::Type{Int16}, ::Type{Int32}) = 16
bits_diff(::Type{Int32}, ::Type{Int64}) = 32
