module StochasticDiff

export Stochastic, Newtonian

include("stochastic.jl"); using .Stochastics

include("newtonian.jl"); using .Newtonians

end # module
