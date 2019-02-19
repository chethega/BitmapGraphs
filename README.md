# BitmapGraphs

Experimental Bitmap-backed undirected graphs for [LightGraphs.jl](https://github.com/JuliaGraphs/LightGraphs.jl).

Examples:
```
using LightGraphs, BitmapGraphs, SparseArrays, BenchmarkTools
r=sprand(1000, 1000, 0.05); r=r+r';
g = induced_subgraph(BMGraph, r, 1:3:1000);
gs = SimpleGraph(nv(g)); for ed in edges(g) add_edge!(gs, src(ed), dst(ed)); end;

g2 = BMGraph(gs); #direct constructor
g2p = induced_subgraph(BMGraph, gs, 1:7:nv(gs)) #induced subgraph of AbstractGraph

#compute some things
@time LightGraphs.clique_percolation(g);
  1.499100 seconds (35.78 M allocations: 1.610 GiB, 11.54% gc time)
@time LightGraphs.clique_percolation(gs);
  1.580663 seconds (35.78 M allocations: 1.610 GiB, 11.19% gc time)

#some algorithms in LightGraphs use @simd on neighbors, and this fails
LightGraphs.vertex_cover(g, DegreeVertexCover());
ERROR: MethodError: no method matching firstindex(::BitmapGraphs.LBitRow{SubArray{UInt64,1,Array{UInt64,2},Tuple{Base.Slice{Base.OneTo{Int64}},Int64},true}})

#Mutation works as expected: We can add and remove edges, but not vertices
add_edge!(g, 17, 19);
rem_edge!(g, 17, 19);
add_vertex!(g, 19)
ERROR: MethodError: no method matching add_vertex!(::BMGraph, ::Int64)
```
The backing storage is a `BitMatrix`. However, its size is rounded up to the next multiple of `64`:
```
julia> g = induced_subgraph(BMGraph, r, 1:10); gs = SimpleGraph(nv(g)); for ed in edges(g) add_edge!(gs, src(ed), dst(ed)); end;

julia> gs
{10, 4} undirected simple Int64 graph

julia> g.adj_mat
64×10 BitArray{2}:
...
julia> g.adj_chunks
1×10 Array{UInt64,2}:
 0x0000000000000002  0x0000000000000001  …  0x0000000000000000
```
The chunks are conveniently accessible in a `Matrix{UInt64}`. Iteration over `outneighbors(g, v)` uses the same algorithm as `Base.LogicalIndex` and is therefore significantly faster than the naive `for i=1:nv(g) if (g.adj_mat[i, j]) ... end end`. 

This is pretty experimental. `LightGraphs` has some `@simd` issues in some algorithms, and `BitmapGraphs` can only really shine if algorihms are optimized for the layout.

Note that `BMGraph` has a dense storage format: It takes `nv(g)^2` bits. This is impractical for large sparse graphs. It is unlikely that `BMGraphs` will ever be useful with more than `1<<16` vertices.

The `BMGraph` is originally intended for `induced_subgraph`: Use a two-tiered algorithm that combines a large sparse graph with many small dense induced subgraphs.


Bitmaped graphs require a very different style of coding. As an example:
```
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
```
gives
```
julia> r=sprand(10_000, 10_000, 0.025); r=r+r'; g = induced_subgraph(BMGraph, r, 1:10_000); gs=SimpleGraph(nv(g)); for ed in edges(g) add_edge!(gs, src(ed), dst(ed)) end;
julia> @btime gdistances(g, 17); @btime gdistances(gs, 17); gdistances(g, 17)==gdistances(gs, 17)
  1.145 ms (7 allocations: 82.28 KiB)
  9.139 ms (8 allocations: 236.09 KiB)
true
```

For tiny graphs, we provide `SBMGraph{N}`: A bitmapped graph that carries `size(g.adj_chunks, 1)` in its type. At the price of a likely type-instability (that needs to be caught by a function barrier), this provides a very convenient and fast graph type, since `neighbors(g, i)` (shorthand: `g[i]`)  can often be a single vector register. This is only useful for tiny graphs, where we can amortize the type-instability by running expensive analyses. As an example implementation:

```
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
        @inbounds for i in todo
            nxt |= g[i] & ~visited
            res[i] = dist
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
```
yields a significant speed-up over LightGraphs (depending on density):
```
julia> n=350; r=sprand(n, n, 0.025); r=r+r'; g = induced_subgraph(BMGraph, r, 1:n); set_diag!(g, false); gsimple=SimpleGraph(nv(g)); for ed in edges(g) add_edge!(gsimple, src(ed), dst(ed)) end; gstatic = SBMGraph(g);

julia> s=17; gdistances(g, s) == gdistances(gsimple, s) == gdistances(gstatic, s)
true

julia> @btime gdistances(gsimple, s); @btime gdistances(g, s); @btime gdistances(gstatic, s);
  23.768 μs (7 allocations: 8.81 KiB)
  4.476 μs (6 allocations: 3.34 KiB)
  1.752 μs (1 allocation: 2.88 KiB)

julia> n=350; r=sprand(n, n, 0.1); r=r+r'; g = induced_subgraph(BMGraph, r, 1:n); set_diag!(g, false); gsimple=SimpleGraph(nv(g)); for ed in edges(g) add_edge!(gsimple, src(ed), dst(ed)) end; gstatic = SBMGraph(g);

julia> @btime gdistances(gsimple, s); @btime gdistances(g, s); @btime gdistances(gstatic, s);
  49.087 μs (7 allocations: 8.81 KiB)
  4.400 μs (6 allocations: 3.34 KiB)
  1.786 μs (1 allocation: 2.88 KiB)

julia> n=128; r=sprand(n, n, 0.1); r=r+r'; g = induced_subgraph(BMGraph, r, 1:n); set_diag!(g, false); gsimple=SimpleGraph(nv(g)); for ed in edges(g) add_edge!(gsimple, src(ed), dst(ed)) end; gstatic = SBMGraph(g);

julia> @btime gdistances(gsimple, s); @btime gdistances(g, s); @btime gdistances(gstatic, s);
  8.845 μs (7 allocations: 3.55 KiB)
  1.153 μs (6 allocations: 1.52 KiB)
  486.164 ns (1 allocation: 1.14 KiB)
```
