import Base: +, -, *, float, rand, randn, size, transpose

type BlockFixedArray{T<:Signed}
    arr::Array{T}
    σ::Real

    function BlockFixedArray{T}(arr::Array{T}, σ::Real) where {T<:Signed}
        new(arr, σ)
    end

    function BlockFixedArray{T}(arr::Array{nT}, σ::Real) where {T<:Signed, nT<:Signed}
        quantize(T,σ,arr)
    end

    function BlockFixedArray{T}(σ::Real) where {T<:Signed}
        new(T[], σ)
    end

    function BlockFixedArray{T}(σ::Real, dims::Integer...) where {T<:Signed}
        new(Array{T}(dims), σ)
    end

    function BlockFixedArray{T}(arr::Array{N}, σ::Real) where {T<:Signed, N<:AbstractFloat}
        quantize(T,σ,arr)
    end
end

# function setindex!(A::BlockFixedArray{T}, X, inds) where {T<:Signed}
#     setindex!(A.arr, X, inds)
# end

# block array muls return a float array
transpose(A::BlockFixedArray) = BlockFixedArray(transpose(A.arr), A.σ)

function *(A::BlockFixedArray{T}, B::BlockFixedArray{T}) where {T<:Signed}
    nT = widen(T)
    arr = Array{Int32}(A.arr) * Array{Int32}(B.arr)
    arr * A.σ * B.σ
end

# block array muls float
function *(A::BlockFixedArray{T}, f::Real) where {T<:Signed}
    arr = map(x->quantize(T, 1, x), A.arr * f)
    BlockFixedArray{T}(arr, A.σ)
end

*(f::Real, A::BlockFixedArray)= *(A, f)

*(M::Array{Bool}, A::BlockFixedArray{T}) = BlockFixedArray{T}(M*A.arr, A.σ)
*(A::BlockFixedArray{T}, M::Array{Bool}) = BlockFixedArray{T}(A.arr*M, A.σ)

function -(A::BlockFixedArray{T}, B::BlockFixedArray{T}) where {T<:Signed}
    @assert A.σ == B.σ # only allow same scale factor for now
    BlockFixedArray{T}(A.arr-B.arr, A.σ)
end

function +(A::BlockFixedArray{T}, B::BlockFixedArray{T}) where {T<:Signed}
    @assert A.σ == B.σ # only allow same scale factor for now
    BlockFixedArray{T}(A.arr+B.arr, A.σ)
end

function float(A::BlockFixedArray)
    float(A.arr)*A.σ
end

function quantize(T::Type{<:Signed}, σ::Real, x::Real)
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
            return T(x_floor+1)
        else return T(x_floor)
        end
    end
    T(x)
end

function quantize(T::Type{<:Signed}, σ::Real, A::Array)
    arr = Array{T}(map(x->quantize(T, σ, x), A))
    BlockFixedArray{T}(arr, σ)
end

function rescale!(A::BlockFixedArray{T}, σ::Real) where {T<:Signed}
    arr = quantize(T, σ, float(A))
    A.arr = arr
    A.σ = σ
    A
end

function rand_blocked(T::Type{<:Signed}, σ, dims::Dims)
    quantize(T, σ, rand(dims))
end

function rand_blocked(T::Type{<:Signed}, σ, dims::Integer...)
    quantize(T, σ, rand(convert(Dims, dims)))
end

function randn_blocked(T::Type{<:Signed}, σ::Real, dims::Dims)
    quantize(T, σ, randn(dims))
end

function randn_blocked(T::Type{<:Signed}, σ::Real, dims::Integer...)
    quantize(T, σ, randn(convert(Dims, dims)))
end

function size(A::BlockFixedArray)
    size(A.arr)
end

function size(A::BlockFixedArray,i::Int)
    size(A.arr, i)
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
