using LightGraphs, BitmapGraphs, SparseArrays, BenchmarkTools
using Test, Random

Random.seed!(2)
r = sprand(1000, 1000, 0.05)
r = r + r'
g = induced_subgraph(BMGraph, r, 1:3:1000)
gs = SimpleGraph(nv(g))
for ed in edges(g)
    add_edge!(gs, src(ed), dst(ed))
end

g2 = BMGraph(gs) # direct constructor
@test g2 isa BMGraph
@test g2.ne == 5427

g2p = induced_subgraph(BMGraph, gs, 1:7:nv(gs)) # induced subgraph of AbstractGraph
@test g2p isa BMGraph
@test g2p.ne == 105

# compute some things
LightGraphs.clique_percolation(g)
LightGraphs.clique_percolation(gs)

g = induced_subgraph(BMGraph, r, 1:10)
@test g.ne == 7
gs = SimpleGraph(nv(g))
for ed in edges(g)
    add_edge!(gs, src(ed), dst(ed))
end

@test gs isa SimpleGraph{Int}
@test gs.ne == 4

@test size(g.adj_mat) == (64, 10)
@test g.adj_mat isa BitArray{2}

@test size(g.adj_chunks) == (1, 10)
@test g.adj_chunks isa Matrix{UInt64}


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

r = sprand(10_000, 10_000, 0.025)
r = r + r'
g = induced_subgraph(BMGraph, r, 1:10_000)
gs = SimpleGraph(nv(g))
for ed in edges(g)
    add_edge!(gs, src(ed), dst(ed))
end
gdistances(g, 17)
gdistances(gs, 17)
@test gdistances(g, 17) == gdistances(gs, 17)


# Static BitmapGraphs

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

n = 350
r = sprand(n, n, 0.025)
r = r+r'
g = induced_subgraph(BMGraph, r, 1:n)
set_diag!(g, false)
gsimple = SimpleGraph(nv(g))
for ed in edges(g)
    add_edge!(gsimple, src(ed), dst(ed))
end
gstatic = SBMGraph(g)

s = 17
@test gdistances(g, s) == gdistances(gsimple, s) == gdistances(gstatic, s)

gdistances(gsimple, s)
gdistances(g, s)
gdistances(gstatic, s)

n = 350
r = sprand(n, n, 0.1)
r = r + r'
g = induced_subgraph(BMGraph, r, 1:n)
set_diag!(g, false)
gsimple = SimpleGraph(nv(g))
for ed in edges(g)
    add_edge!(gsimple, src(ed), dst(ed))
end
gstatic = SBMGraph(g)

gdistances(gsimple, s)
gdistances(g, s)
gdistances(gstatic, s)

n = 128
r = sprand(n, n, 0.1)
r = r+r'
g = induced_subgraph(BMGraph, r, 1:n)
set_diag!(g, false)
gsimple = SimpleGraph(nv(g))
for ed in edges(g)
    add_edge!(gsimple, src(ed), dst(ed))
end
gstatic = SBMGraph(g)

gdistances(gsimple, s)
gdistances(g, s)
gdistances(gstatic, s)
