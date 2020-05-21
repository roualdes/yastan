#!/usr/bin/env julia

using YaStan
using Test
using SafeTestsets

@safetestset "Welford Accumulator" begin include("welford.jl") end
@safetestset "Utilities" begin include("utilities.jl") end
