using YaStan
using Test

q = Dict(:m => 0.1, :s => [0.3 0.5; 0.4 0.6], :b => [0.2])
u = Dict(:m => 0.1, :s => [0.3 0.5; 0.4 0.6], :b => [0.2])

q = YaStan.prepareq(q)

@test q[:length] == 6
@test haskey(q, :vec) == true
@test q[:vec][:m][1] == 1
@test q[:vec][:b][1] == 2

x = zeros(6)
YaStan.assignq!(x, q)

@test length(x) == 6
@test isapprox(x, [0.1; 0.2; 0.3; 0.4; 0.5; 0.6])

v = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0]
YaStan.updateq!(q, v; addself = true)

@test isapprox(q[:m], 1.1)
@test isapprox(q[:s], [3.3 5.5; 4.4 6.6])
@test isapprox(q[:b], [2.2])

#updateq: inverse of assign
YaStan.updateq!(q, x)

@test all(isapprox(q[k], u[k]) for k in keys(q[:vec]))
