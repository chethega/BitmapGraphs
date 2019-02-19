module BitmapGraphs

using LightGraphs, SparseArrays

export AbstractBMGraph, BMGraph, invert!, set_diag!

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
const _BitRow = Union{LBitRow, BitRow}# BitRow

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
    nv::Int
end

function BMGraph(n::Int)
    nc = Base.num_bit_chunks(n)
    adj_mat = falses(nc*64, n)
    adj_chunks = reshape(adj_mat.chunks, nc, n)
    degs = zeros(Int, n)
    return BMGraph(adj_chunks, adj_mat, degs, 0, n)
end

function BMGraph(g::AbstractGraph)
    vertices(g) == Base.OneTo(nv(g))||throw(ArgumentError("BMGraphs only support `vertices isa Base.OneTo`"))
    res = BMGraph(nv(g))
    for ed in edges(g)
        add_edge!(res, src(ed), dst(ed))
    end
    return res
end

function Base.copy(bm::BMGraph)
    adj_mat = copy(bm.adj_mat)
    adj_chunks = reshape(adj_mat.chunks, size(bm.adj_chunks))
    return BMGraph(adj_chunks, adj_mat, copy(bm.degs), bm.ne, bm.nv)
end


#interface:
LightGraphs.edgetype(::AbstractBMGraph) = LightGraphs.SimpleGraphs.SimpleEdge{Int}
Base.eltype(::BMGraph) = Int
LightGraphs.nv(g::BMGraph) = g.nv
LightGraphs.ne(g::BMGraph) = g.ne
LightGraphs.vertices(g::BMGraph) = Base.OneTo(g.nv)
LightGraphs.edges(g::AbstractBMGraph) = BMEdgeIter{typeof(g.adj_chunks), LightGraphs.SimpleGraphs.SimpleEdge{Int}}(g.ne, g.adj_chunks)
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
Base.@propagate_inbounds LightGraphs.inneighbors(g::BMGraph, v...) = outneighbors(g, v...)
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
        for i=1:bm.nv
            add_edge!(bm, i, i)
        end
    else
        for i=1:bm.nv
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
        for u in neighbors(g, v)
            u < v && continue
            while @inbounds vertices[vi] < u
                vi < n || @goto done
                vi += 1
            end
            u < v && continue
            @assert (u == @inbounds vertices[i])
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
            edgepred(v, u, @inbounds mat.nzval[ui]) && add_edge!(bm, i, vi)
            vi += 1
            vi < n || @goto done
        end
        @label done
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


    #TODO: SBMGraph, where nchunks is a type parameter.
#This permits us to return an isbits row, backed by tuple or StaticVector.
#idea is to be super fast and allocation-free for tiny graphs with <=512 vertices.

end
