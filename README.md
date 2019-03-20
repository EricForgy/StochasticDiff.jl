_Note: I wrote this note and this package out of curiosity and I did not perform any literature search whatsoever. I would be suprised if this is new, but you never know_ :) _If this material is well known, please let me know and I will add a reference._

## StochasticDiff.jl

This is a simple package to demonstrate the extension of automatic differentiation to stochastic processes.

### Background

Give any two functions `f` and `g`, a derivation `d` is a nilpotent map satisfying the product rule

```julia
d(fg) = (df)g + f(dg).
```

Collecting terms on the right-hand side, we observe that `d(fg)` can be determined if we know, not just `f` and `g`, but the pairs `(f,df)` and `(g,dg)`.

Since `f` and `g` are functions and `df` and `dg` are covector fields on some space, then the pairs `(f,df)` and `(g,dg)` represent covector fields on a larger total space.

Let `Pi` denote a projection map given by

```julia
Pi(f,df) = f.
```

For a given derivation `d`, we have the inverse map

```
Pi^(-1) f = (f,df).
```

We introduce a product of `(f,df)` and `(g,dg)` by insisting that `Pi^(-1)` be an algebra homomorphism, i.e.

```julia
Pi^(-1)(f*g) = Pi^(-1)(f)*Pi^(-1)(g)
```

so that 

```julia
(f,df)(g,dg) = (fg,d(fg)).
```

#### Example: One Dimension

Consider the case of a smooth one-dimensional space parameterized by a smooth  coordinate function `x` and two covectors

```julia
(f,df) = (f,(@_x f) dx)
```

and

```julia
(g,dg) = (g,(@_x g) dx),
```

where `@_x` denotes partial derivative with respect to `x`.

The data for a covector `(f,df)` can be encoded in a struct

```julia
struct Newton1D{F,dFdx}
    f::F
    dfdx::dFdx
end
```

and the product is given by

```julia
Base.:*(a::Newton1D,b::Newton1D) = Newton1D(a.f*b.f, a.dfdx*b.x+a.f*b.dfdx)
```

As we will see below, it is convenient to define

```julia
Base.exp(a::Newton1D) = exp(a.f)*Newton1D(1.0,a.dfdx)
```

and

```julia
function Base.log(a::Newton1D)
    invx = inv(a.f)
    return Newtonian(log(a.f),invx*a.dfdx)
end
```

so that

```julia
Base.:^(x,y::Newton1D) = exp(y*log(x))
```

### Newtonian Processes

Next consider a (1+1)-dimensional Newtonian process

```julia
df = (@_x f) dx + (@_t f) dt.
```

The corresponding covector `(f,df)` can be expanded to

```julia
(f,df) = (f,(@_x f)dx) + (f,(@_t f)dt)
```

indicating the process can be encoded into a struct

```julia
struct Newtonian{F,dFdx,dFdt}
    f::F
    dfdx::dFdx
    dfdt::dFdt
end
```

Similar to the one-dimensional example above, the product of two covectors is given by

```julia
Base.:*(a::Newtonian,b::Newtonian) = Newtonian(
    a.f*b.f,
    a.f*b.dfdx + a.dfdx*b.f,
    a.f*b.dfdt + a.dfdt*b.f)
```

and we have

```julia
Base.exp(x::Newtonian) = exp(x.f)*Newtonian(1,x.dfdx,x.dfdt)

function Base.log(x::Newtonian)
    invx = inv(x.f)
    return Newtonian(log(x.f),invx*x.dfdx,invx*x.dfdt)
end

Base.:^(x,y::Newtonian) = exp(y*log(x))
```

### Stochastic Processes

Finally, consider the (1+1)-dimensional stochastic process

``df = (@_x f) dx + (@_t f + 1/2 @_x^2 f) dt.``

Like the Newtonian process above, the stochastic process can be encoded into a struct

```julia
struct Stochastic{F,dFdx,dFdt}
    f::F
    dfdx::dFdx
    dfdt::dFdt
end
```

Stochastic processes are also amenable to automatic differentiate with some minor revisions.

To see this, first rewrite the above process as

``df = (@_x f) dx + (~@_t f) dt,``

where

``~@_t = @_t + 1/2 @_x^2.``

The operator `~@_t` does not satisfy the usual product rule of partial derivatives. Instead, it satisfies

``~@_t(fg) = (~@_t f)g + f(~@_t g) + (@_x f)(@_x g).``

Therefore, we have

```julia
Base.:*(x::Stochastic,y::Stochastic) =
    Stochastic(
        x.f*y.f,
        x.f*y.dfdx + x.dfdx*y.f,
        x.f*y.dfdt + x.dfdt*y.f + x.dfdx*y.dfdx
    )
```

Then, with a bit of tedious, but straightforward algebra, we have

```julia
function Base.inv(y::Stochastic)
    invy = inv(y.f)
    return invy*Stochastic(
        1.0,
        -invy*y.dfdx,
        -invy*y.dfdt + invy^2*y.dfdx^2)
end

Base.:^(x::Stochastic,k::R) where R <:Number = 
    Stochastic(
        x.f^k,
        k*x.f^(k-1)*x.dfdx,
        k*x.f^(k-1)*x.dfdt + .5*k*(k-1)*x.f^(k-2)*x.dfdx^2)

        Base.:/(x::Stochastic,y::Stochastic) = 
    Stochastic(
        x.f/y.f,
        (x.dfdx*y.f - x.f*y.dfdx)/y.f^2,
        (x.dfdt*y.f^2 + x.f*(y.dfdx^2 - y.f*y.dfdt))/y.f^3
    )

Base.exp(x::Stochastic) =
    exp(x.f)*Stochastic(
        1,
        x.dfdx,
        x.dfdt + .5*x.dfdx^2)

function Base.log(x::Stochastic)
    invx = inv(x.f)
    val = invx*x.dfdx
    Stochastic(
        log(x.f),
        val,
        invx*x.dfdt - .5*val^2)
end

Base.:^(x,y::Stochastic) = exp(y*log(x))
```

### Geometric Brownian Motion

Consider geometric Brownian motion in (1+1)-dimensions given by the stochastic differential equation

```julia
dS = mu S dt + sigma S dx
```

with closed-form solution

```julia
S(x,t) = S(0,0) exp[(mu-sigma^2/2)t + sigma x]
```

We'll first write down the closed form solution as a stochastic function:

```julia
julia> mu = 4; sigma = 5; f(x::Stochastic,t::Stochastic) = exp((mu-sigma^2/2)*t + sigma*x)
f (generic function with 2 methods)
```

Next, for convenience, we'll add a method to take number values for `x` and `t`:

```julia
julia> f(x,t) = f(Stochastic(x,1,0),Stochastic(t,0,1))
f (generic function with 2 methods)
```

Finally, we evaluate the function at `x = 0, y = 0`:

```julia
julia> f(0,0)
Stochastic{Float64,Float64,Float64}(1.0, 5.0, 4.0)
```

This is the expected result since

```julia
@_x S = mu S and ~@_t S = mu S.
```

Furthermore, we have

```julia
julia> f(1,1)/f(1,1).f
Stochastic{Float64,Float64,Float64}(1.0, 5.0, 4.0)
```

as expected.

#### Sanity Checks

Check inverses:

```julia
julia> x
Stochastic{Float64,Float64,Float64}(0.8198419485754191, 0.5608953995378232, 0.3496467042194824)

julia> inv(x)
Stochastic{Float64,Float64,Float64}(1.2197472960948492, -0.8344909017733231, 0.050718904750958915)

julia> 1.0/x
Stochastic{Float64,Float64,Float64}(1.2197472960948492, -0.8344909017733231, 0.050718904750958915)

julia> x^(-1.0)
Stochastic{Float64,Float64,Float64}(1.2197472960948492, -0.8344909017733233, 0.05071890475095886)

julia> inv(x)*x
Stochastic{Float64,Float64,Float64}(0.9999999999999999, 1.1102230246251565e-16, 0.0)
```

Check division:

```julia
julia> x/y
Stochastic{Float64,Float64,Float64}(0.8341627751184755, 0.039239594639088035, 0.4932061731418642)

julia> x*y^(-1)
Stochastic{Float64,Float64,Float64}(0.8341627751184755, 0.039239594639088105, 0.12961198046378852)
```

Check `log(exp(x)) == exp(log(x)) == x`:

```julia
julia> exp(x)
Stochastic{Float64,Float64,Float64}(2.270141010155909, 1.2733116488985963, 1.1508446453370713)

julia> log(x)
Stochastic{Float64,Float64,Float64}(-0.19864370294139905, 0.6841506469783, 0.19244956817977596)

julia> log(exp(x))
Stochastic{Float64,Float64,Float64}(0.8198419485754191, 0.5608953995378232, 0.3496467042194824)

julia> exp(log(x))
Stochastic{Float64,Float64,Float64}(0.8198419485754191, 0.5608953995378232, 0.34964670421948235)

julia> x
Stochastic{Float64,Float64,Float64}(0.8198419485754191, 0.5608953995378232, 0.3496467042194824)
```

Check `exp(-log(x)) = inv(x)`

```julia
julia> exp(-log(x))
Stochastic{Float64,Float64,Float64}(1.2197472960948492, -0.8344909017733231, 0.050718904750958915)

julia> inv(x)
Stochastic{Float64,Float64,Float64}(1.2197472960948492, -0.8344909017733231, 0.050718904750958915)
```