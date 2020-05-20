#!/usr/bin/env julia

using yastan
using Test
using SafeTestsets

@safetestset "Welford Accumulator" begin include("welford.jl") end

# @testset "yastan.jl" begin
#     # Write your own tests here.
# end
