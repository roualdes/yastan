# yastan

Yet another implementation of the version of no U-turn Hamiltonian
Monte Carlo algorithm behind Stan.

## Introduction

Use the file play.jl play around with this version of Stan in Julia.
I'm using Julia 1.3.0 and the following packages

* Revise
* Plots
* Statistics
* LinearAlgebra
* RandomNumbers
* Zygote
* Distributed
* FFTW

There's a file named play.R which I use to compare my implementation of Stan
to the RStan (2.19.2) version of Stan.

## Examples

```
q = Dict{Symbol, Any}(:xraw => 1.0) # paramters
d = Dict{Symbol, Any}(:mu => 14.5, :sigma => 3.14) # data

function model(q, d)
    d[:x] = d[:mu] .+ q[:xraw] * d[:sigma] # transformed parameter
    return YaStan.normal_.(q[:xraw], 0.0, 1.0) # target
end

samples, info = YaStan.hmc(model, q, d)
```


## Limitations

* Can't mutate an array with differentiable variables until I figure
  out what to do with [Zygote's issue
  377](https://github.com/FluxML/Zygote.jl/issues/377).  This pretty
  much rules out anything to do with a matrix.
