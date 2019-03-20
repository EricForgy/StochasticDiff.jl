# StochasticForwardDiff.jl

This is a simple package to demonstrate the extension of automatic differentiation to stochastic processes.

## Newtonian Processes

Consider the Netwonian process

$$df = \left(\partial_x f\right) dx + \left(\partial_t f\right) dt$$

The process can be coded into a struct that contains information about both the value and the derivatives, i.e.

```julia
struct Newtonian{F,dFdx,dFdt}
    f::F
    dfdx::dFdx
    dfdt::dFdt
end
```



Consider the stochastic process

$$df = \left(\partial_x f\right) dx + \left(\partial_t f + \frac {1}{2}\partial_x^2 f\right) dt$$

