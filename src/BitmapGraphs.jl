module BitmapGraphs

using LightGraphs, SparseArrays

export AbstractBMGraph, BMGraph, invert!, set_diag!, BitRow, SRow, nchunks, mk_msk, SBMGraph
import Base: setindex
#utils for iterating over bit-chunks
@inline _blsr(x) = x & (x-1)


struct LBitRow{chunk_T}
    len::Int
    chunks::chunk_T
end

#future feature
struct BitRow{chunk_T}
    chunks::chunk_T
end
struct SRow{N}
    chunks::NTuple{N, UInt64}
end
const _BitRow = Union{LBitRow, BitRow, SRow}# BitRow

#opt out of simd. Fixme once github.com/JuliaLang/julia/pull/31113 has landed.
Base.SimdLoop.simd_index(v::_BitRow, j::Int64, i) = j
Base.SimdLoop.simd_inner_length(v::_BitRow, j::Int64) = 1
Base.SimdLoop.simd_outer_range(v::_BitRow) = v


nchunks(br::_BitRow) = length(br.chunks)

Base.@propagate_inbounds function Base.in(br::_BitRow, idx)
    i1,i2 = Base.get_chunks_id(idx)
    @boundscheck ((0<i1<len(br)) || return false)
    @inbounds u = br.chunks[i1]
    return iszero(u & (1<< (i2 & 63)))
end


function Base.iterate(br::_BitRow)
    nchunks(br)==0 && return nothing
    return iterate(br, (1, @inbounds br.chunks[1]))
end

#@inline Base.length(br::BitRow) = sum(count_ones, br.chunks)
@inline Base.length(br::LBitRow) = br.len


@inline function Base.iterate(br::_BitRow, s)
    chunks = br.chunks
    i1, c = s
    while c==0
        i1 % UInt >= length(chunks) % UInt && return nothing
        i1 += 1
        @inbounds c = chunks[i1]
    end
    tz = trailing_zeros(c) + 1
    c = _blsr(c)
    return ((i1-1)<<6 + tz, (i1, c))
end

struct BMEdgeIter{chunkT, T}
    ne::Int
    chunks::chunkT
end
@inline Base.eltype(::BMEdgeIter{chunkT, T}) where {chunkT, T} = T
@inline Base.length(ei::BMEdgeIter) = ei.ne


@inline function Base.iterate(br::BMEdgeIter)
    length(br.chunks)>0 || return nothing
    return iterate(br, (1, 1, @inbounds br.chunks[1]))
end

@inline function Base.iterate(br::BMEdgeIter{chunkT, T}, s) where {chunkT, T}
    chunks = br.chunks
    nchunks = size(chunks,1)
    row, col, c = s
    while c==0
        if ((row % UInt) >= nchunks)
            col >= size(chunks, 2) && return nothing
            col += 1
            row,i2 = Base.get_chunks_id(col)
            @inbounds c = chunks[row, col] & (~UInt(0) << (i2 & 63) )
            continue
        end
        row += 1
        @inbounds c = chunks[row, col]
    end
    tz = trailing_zeros(c) + 1
    c = _blsr(c)
    return T((row-1)<<6 + tz, col) ,  (row, col, c)
end

#TODO utils for bitrows:
#intersection, union, xor, setminus, complement, count.
#need to come in broadcast and broadcast! variants. 


abstract type AbstractBMGraph <:LightGraphs.AbstractGraph{Int} end

mutable struct BMGraph <: AbstractBMGraph
    adj_chunks::Matrix{UInt64}
    adj_mat::BitMatrix
    degrees::Vector{Int}
    ne::Int
end

function BMGraph(n::Int)
    nc = Base.num_bit_chunks(n)
    adj_mat = falses(nc*64, n)
    adj_chunks = reshape(adj_mat.chunks, nc, n)
    degs = zeros(Int, n)
    return BMGraph(adj_chunks, adj_mat, degs, 0)
end

function BMGraph(g::AbstractGraph)
    vertices(g) == Base.OneTo(nv(g))||throw(ArgumentError("BMGraphs only support `vertices isa Base.OneTo`"))
    res = BMGraph(nv(g))
    for ed in edges(g)
        add_edge!(res, src(ed), dst(ed))
    end
    return res
end
nchunks(g::BMGraph) = size(g.adj_chunks, 1)

function Base.copy(bm::BMGraph)
    adj_mat = copy(bm.adj_mat)
    adj_chunks = reshape(adj_mat.chunks, size(bm.adj_chunks))
    return BMGraph(adj_chunks, adj_mat, copy(bm.degs), bm.ne)
end


#interface:
LightGraphs.edgetype(::AbstractBMGraph) = LightGraphs.SimpleGraphs.SimpleEdge{Int}
Base.eltype(::AbstractBMGraph) = Int
LightGraphs.nv(g::AbstractBMGraph) = size(g.adj_chunks, 2)
LightGraphs.ne(g::BMGraph) = g.ne
LightGraphs.vertices(g::AbstractBMGraph) = Base.OneTo(nv(g))
LightGraphs.edges(g::AbstractBMGraph) = BMEdgeIter{typeof(g.adj_chunks), LightGraphs.SimpleGraphs.SimpleEdge{Int}}(ne(g), g.adj_chunks)
LightGraphs.is_directed(g::AbstractBMGraph) = false
LightGraphs.is_directed(::Type{<:AbstractBMGraph}) = false

LightGraphs.has_vertex(g::BMGraph, v) =  g in vertices(g)


Base.@propagate_inbounds LightGraphs.has_edge(g::BMGraph, e::AbstractEdge) = has_edge(g, src(e), dst(e)) 
Base.@propagate_inbounds LightGraphs.has_edge(g::BMGraph, e::Tuple{<:Integer, <:Integer}) = has_edge(g, e...) 
Base.@propagate_inbounds LightGraphs.has_edge(g::BMGraph, s, d) = g.adj_mat[s, d]
Base.@propagate_inbounds function LightGraphs.outneighbors(g::BMGraph, v) 
    vvv = view(g.adj_chunks, :, v)
    return LBitRow(g.degrees[v], vvv)
end
Base.@propagate_inbounds LightGraphs.inneighbors(g::AbstractBMGraph, v...) = outneighbors(g, v...)
Base.@propagate_inbounds LightGraphs.indegree(g::BMGraph, v::Integer) = g.degrees[v]
Base.@propagate_inbounds LightGraphs.outdegree(g::BMGraph, v) = g.degrees[v]
#Base.@propagate_inbounds LightGraphs.degree(g::BMGraph, v) = g.degrees[v] #causes method ambiguity errors



LightGraphs.add_edge!(g::BMGraph, e::AbstractEdge) = add_edge!(g, src(e), dst(e)) 
LightGraphs.add_edge!(g::BMGraph, e::Tuple{<:Integer, <:Integer}) = add_edge!(g, e...) 
Base.@propagate_inbounds function LightGraphs.add_edge!(g::BMGraph, s, d)
    @boundscheck begin 
        0 < d <= nv(g) || throw(BoundsError(g, d))
        0 < s <= nv(g) || throw(BoundsError(g, s))
    end
    @inbounds oldv = g.adj_mat[s,d]
    @inbounds g.adj_mat[s,d] = true
    @inbounds g.adj_mat[d,s] = true
    if !oldv
        g.ne += 1
        @inbounds g.degrees[s] += 1
        s != d && @inbounds g.degrees[d] += 1
    end
    return !oldv
end



Base.@propagate_inbounds function LightGraphs.rem_edge!(g::BMGraph, s, d)
    @boundscheck begin 
        0 < d <= nv(g) || throw(BoundsError(g, d))
        0 < s <= nv(g) || throw(BoundsError(g, s))
    end
    @inbounds oldv = g.adj_mat[s,d]
    @inbounds g.adj_mat[s,d] = false
    @inbounds g.adj_mat[d,s] = false
    if oldv
        g.ne -= 1
        @inbounds g.degrees[s] -= 1
        s != d && @inbounds g.degrees[d] -= 1
    end
    return oldv
end

"""
invert!(bm::BMGraph,  no_selfloops = true)

Inverts the graph: All formerly present edges are now missing, and vice versa. This includes self-loops.
If `no_selfloops` is set, then all self-loops are removed afterwards.
"""
function invert!(bm::BMGraph, no_selfloops = true)
    nr, nc = size(bm)
    @inbounds for i=1:nc
        bm.degrees[i] = nc - bm.degrees[i]
    end
    bm.ne = sum(bm.degrees)
    @simd for i=1:length(bm.adj_chunks)
        @inbounds bm.adj_chunks[i] = ~bm.adj_chunks[i]
    end
    msk_e = Base._msk_end(nc)
    for j = 1:nr 
        @inbounds bm.adj_chunks[j, nr] &= msk_e
    end
    if no_selfloops
        set_diag!(bm, false)
    end

    return bm
end

"""
set_diag!(bm::BMGraph, b::Bool)

Sets the diagonal to b, i.e. adds or removes all self-loops.
"""
function set_diag!(bm::BMGraph, b::Bool)
    if b
        for i=1:nv(bm)
            add_edge!(bm, i, i)
        end
    else
        for i=1:nv(bm)
            rem_edge!(bm, i, i)
        end
    end
    bm 
end



"""
induced_subgraph(BMGraph, g::Lightgraphs.AbstractGraph, vertices)

Returns the induced subgraph as a BMGraph. Assumes that input vertices and `neighbors` are sorted.
"""
function LightGraphs.induced_subgraph(::Type{BMGraph}, g::LightGraphs.AbstractGraph, vertices)
    n = length(vertices)
    bm = BMGraph(n)
    for (i,v) in enumerate(vertices)
        vi = 1
        #vv = vertices[1]
        @inbounds for u in neighbors(g, v)
            u <  vertices[vi] && continue
            while vertices[vi] < u
                vi < n || @goto done
                vi += 1
            end
            u < vertices[vi] && continue
            add_edge!(bm, i, vi)
            vi += 1
        end
        @label done
    end
    return bm
end


_true(args...) = true
_false(args...) = false

"""
induced(::Type{BMGraph}, mat::SparseMatrixCSC, vertices, edgepred = _true, bailout = _false)

Returns the induced subgraph, of the graph with adjacency matrix `mat` for which `edgepred(i, j, mat[i,j])` evaluates to `true`.
The matrix is assumed symmetric and `vertices` is assumed sorted.

The predicate `bailout(bm, i)` is called after each constructed row `i`. If it returns `true`, then the partial result is returned.
"""
function LightGraphs.induced_subgraph(::Type{BMGraph}, mat::SparseMatrixCSC, vertices, edgepred = _true, bailout = _false)
    n = length(vertices)
    bm = BMGraph(n)
    n > 1 || return bm
    for (i,v) in enumerate(vertices)
        vi = 1
        ran =  mat.colptr[v]:mat.colptr[v+1]-1
        for ui in ran #neighbors(g, v)
            @inbounds u = mat.rowval[ui]
            @inbounds u <  vertices[vi] && continue

            while @inbounds vertices[vi] < u
                vi < n || @goto done
                vi += 1
            end
            @inbounds u <  vertices[vi] && continue
            edgepred(v, u, @inbounds mat.nzval[ui]) && (@inbounds bm.adj_mat[vi, i] = true)
            vi += 1
            vi < n || @goto done
        end
        @label done
        deg = sum(count_ones, view(bm.adj_chunks, :, i))
        bm.degrees[i] = deg
        bm.ne += deg
        bailout(bm, i) && return bm
    end
    return bm
end

#example algorithm
function LightGraphs.gdistances(g::BMGraph, s)
    res = fill(-1, nv(g))
    nchunks = size(g.adj_chunks, 1)
    visited = zeros(UInt, nchunks)
    todo = zeros(UInt, nchunks)
    nxt = zeros(UInt, nchunks)
    done = false
    dist = 1
    todo .= outneighbors(g, s).chunks
    visited .= outneighbors(g, s).chunks
    while !done
        done = true
        for i in BitRow(todo)
            @simd for j = 1:nchunks
                @inbounds nxt[j] |= g.adj_chunks[j, i] & ~visited[j]
            end
            res[i] = dist
            done = false
        end
        done && break
        visited .|= nxt
        todo .= nxt
        fill!(nxt, 0)
        dist += 1
    end
    res[s] = 0
    res
end

###############
#Static graphs

#Base.in, Base.iterate, are defined above
@inline Base.length(a::SRow) = count_ones(a)
Base.eltype(::SRow) = Int

(Base.:&)(a::SRow{N}, b::SRow{N}) where N = SRow(ntuple(i->(@inbounds a.chunks[i] & b.chunks[i]), N))
(Base.:|)(a::SRow{N}, b::SRow{N}) where N = SRow(ntuple(i->(@inbounds a.chunks[i] | b.chunks[i]), N))
(Base.xor)(a::SRow{N}, b::SRow{N}) where N = SRow(ntuple(i->(@inbounds xor(a.chunks[i], b.chunks[i])), N))
(Base.:~)(a::SRow{N}) where N = SRow(ntuple(i->(@inbounds ~a.chunks[i]), N))
Base.zero(::Type{SRow{N}}) where N = SRow(ntuple(i->UInt64(0), N))
Base.iszero(a::SRow{N}) where N = all(iszero, a.chunks)
Base.isempty(a::SRow{N}) where N = all(iszero, a.chunks)


Base.@propagate_inbounds function Base.setindex!(a::Base.RefValue{SRow{N}}, b, idx) where N
    i1,i2 = Base.get_chunks_id(idx)
    @boundscheck ((0<i1<=N) || throw(BoundsError(a, idx)))
    @inbounds u = a[].chunks[i1]
    nu = ifelse(b, u | (1<< (i2 & 63)), u & ~(1<< (i2 & 63)) )
    ptr = convert(Ptr{UInt64}, pointer_from_objref(a))
    GC.@preserve a unsafe_store!(ptr, nu, i1)
    b
end

Base.@propagate_inbounds function mk_msk(::Type{SRow{N}}, idx) where N
    i1,i2 = Base.get_chunks_id(idx)
    @boundscheck ((0<i1<=N) || throw(BoundsError(SRow{N}, idx)))
    r = Ref(zero(SRow{N}))
    ptr = convert(Ptr{UInt64}, pointer_from_objref(r))
    GC.@preserve r begin
    for i=1:i1-1
        unsafe_store!(ptr, -1%UInt, i)
    end
    unsafe_store!(ptr, Base._msk_end(idx), i1)
    end
    return r[]
end

#=
function pop_(a::SRow{N}) where N
    i = 1
    @inbounds while i <= N && iszero(a.chunks[i])
        i += 1
    end
    i == N+1 && return -1, a
    @inbounds idx = trailing_zeros(a.chunks[i]) + 1 + (i-1)<<6
    r = Ref(a)
    ptr = convert(Ptr{UInt64}, pointer_from_objref(a))
    unsafe_store!(ptr, _blsr(a.chunks[i]), i)
    return idx, a[]
end
=#

Base.@propagate_inbounds Base.setindex(a::SRow{N}, b, idx) where N = begin r=Ref(a); setindex!(r, b, idx); r[] end

@inline function Base.count_ones(a::SRow{N}) where N
    res = 0
    for i=1:N
        @inbounds res+= count_ones(a.chunks[i])
    end
    res
end
Base.@propagate_inbounds function Base.count_ones(a::SRow{N}, k) where N
    res = 0
    i1, i2 = Base.get_chunks_id(k)
    for i = 1:i1-1
        res += count_ones(a.chunks[i])
    end
    res += count_ones(a.chunks[i1] & Base.msk_end(i1))
    res
end


##SBMGraph
struct SBMGraph{N} <:AbstractBMGraph
    adj_chunks::Matrix{UInt64}
end
@inline SBMGraph(g::BMGraph) = SBMGraph(g, Val(nchunks(g)))
SBMGraph(g::BMGraph, ::Val{N}) where N = SBMGraph{N}(g.adj_chunks)



nchunks(g::SBMGraph{N}) where N = N
Base.@propagate_inbounds function Base.getindex(g::SBMGraph{N}, idx) where N
    @boundscheck checkbounds(g.adj_chunks, 1, idx)
    off = N*(idx-1)
    return SRow(ntuple(i -> (@inbounds g.adj_chunks[i + off]), N))
end
Base.@propagate_inbounds LightGraphs.outneighbors(g::SBMGraph, idx) = g[idx]
function LightGraphs.ne(g::SBMGraph)
    s = sum(count_ones, g.adj_chunks)
    for i=1:nv(g)
        i in g[i] && (s+=1)
    end
    return s>>1
end
Base.@propagate_inbounds LightGraphs.has_edge(g::SBMGraph, s, d) = d in g[s]
Base.@propagate_inbounds LightGraphs.indegree(g::SBMGraph, v::Integer) = count_ones(g[v])
Base.@propagate_inbounds LightGraphs.outdegree(g::SBMGraph, v::Integer) = count_ones(g[v])


LightGraphs.gdistances(g::SBMGraph{N}, s) where N = gdistances!(g, s, fill(0, nv(g)))
function LightGraphs.gdistances!(g::SBMGraph{N}, s, res) where N
    resize!(res, nv(g))
    fill!(res, 0)
    T = SRow{N}
    @inbounds todo = visited = g[s]
    nxt = zero(T)
    dist = 1
    done = false
    while !done
        done = true
        for i in todo
            @inbounds nxt |= g[i] & ~visited
            @inbounds res[i] = dist
            done = false
        end
        done && break
        visited |= nxt
        todo = nxt
        nxt = zero(T)
        dist += 1
    end
    res[s] = 0
    res
end

end
