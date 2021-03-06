function jacobian_batched(f, xs::AbstractMatrix, T::DataType, AT::UnionAll)::Tuple
    y, back = Zygote.pullback(f, xs)
    z = convert(AT, zeros(T, size(xs)))
    res = convert(AT, zeros(T, size(xs, 1), size(xs, 1), size(xs, 2)))
    for i in 1:size(y, 1)
        z[i, :] .= one(eltype(xs))
        res[i, :, :] .= only(back(z))
        z[i, :] .= zero(eltype(xs))
    end
    y, res
end
