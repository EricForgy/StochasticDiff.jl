# StochasticDiff.jl

This is a simple package to demonstrate the extension of automatic differentiation to stochastic processes.

## Background

Give any two functions $f$ and $g$, a derivation $d$ is a nilpotent map satisfying the product rule

$$d(fg) = (df)g + f(dg).$$

We observe that $d(fg)$ can be determined if we know, not just $f$ and $g$, but the pairs $(f,df)$ and $(g,dg)$.

Since $f$ and $g$ are functions and $df$ and $dg$ are covector fields on some space, then the pairs $(f,df)$ and $(g,dg)$ represent covector fields on a larger total space.

Let $\pi$ denote a projection map given by

$$\pi(f,df) = f.$$

For a given derivation $d$, we have the inverse map

$$\pi^{-1} f = (f,df).$$

We introduce a product of $(f,df)$ and $(g,dg)$ by insisting that $\pi^{-1}$ be an algebra homomorphism, i.e.

so that

$$\begin{aligned}(f,df)(g,dg) :&= (fg,d(fg)) \\ &= (fg,(df)g+f(dg)).\end{aligned}$$

## Newtonian Processes

Consider a (1+1)-dimensional Newtonian process

$$df = (\partial_x f) dx + (\partial_t f) dt.$$

The corresponding covector $(f,df)$ can be expanded to

$$(f,df) = (f,(\partial_x f)dx) + (f,(\partial_t f)dt)$$

indicating the process can be encoded into a struct

```julia
struct Newtonian{F,dFdx,dFdt}
    f::F
    dfdx::dFdx
    dfdt::dFdt
end
```

### Example: Geometric Linear Motion

Consider a simple (1+1)-dimensional differential equation

$$df = \alpha f dx + \beta f dt + $$

with closed-form solution

$$f(x,t) = f_0 \exp{\left(\alpha x + \beta t\right)}.$$

The partial derivatives of $f(x,t)$ are

$$\partial_x f(x,t) = \alpha f(x,t)$$

and

$$\partial_t f(x,t) = \beta f(x,y).$$

For concreteness, let $\alpha = 3$, $\beta = 4$ and $f(0,0) = 1$ so we have:

```julia
julia> f(x,t) = exp(3*x+4*t)
```

with

$$\partial_x f = 3\quad\text{and}\quad\partial_t f = 4.$$

Next, define:

$$\pi^{-1}[f(x,t)] := f\left[\pi^{-1}(x),\pi^{-1}(t)\right]$$

or

```julia
julia> fnewt(x,t) = f(Newtonian(x,1,0),Newtonian(t,0,1))
```

which comes from

$$\pi^{-1}(x) = (x,dx) = Newtonian(x,1,0)$$

and

$$\pi^{-1}(t) = (t,dt) = Newtonian(t,0,1).$$

Evaluating

```julia
julia> fnewt(0,0)
Newtonian:
  f: 1.0
  dfdx: 3.0 # alpha
  dfdt: 4.0 # beta
```

as expected.

Furthermore,

```julia
julia> x = rand(); t = rand(); fnewt(x,t)/fnewt(x,t).f
Newtonian:
  f: 1.0
  dfdx: 3.0 # alpha
  dfdt: 4.0 # beta
```

also as expected.

## Stochastic Processes

Finally, consider the (1+1)-dimensional stochastic process

$$df = (\partial_x f) dx + (\partial_t f + \frac{1}{2} \partial_x^2 f) dt.$$

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

$$df = (\partial_x f) dx + (\widetilde{\partial_t} f) dt,$$

where

$$\widetilde{\partial}_t = \partial_t + \frac{1}{2} \partial_x^2.$$

The operator $\widetilde{\partial_t}$ does not satisfy the usual product rule of partial derivatives. Instead, it satisfies

$$\widetilde{\partial_t}(fg) = (\widetilde{\partial_t} f)g + f(\widetilde{\partial_t} g) + (\partial_x f)(\partial_x g).$$

Therefore, we have

```julia
Base.:*(x::Stochastic,y::Stochastic) =
    Stochastic(
        x.f*y.f,
        x.f*y.dfdx + x.dfdx*y.f,
        x.f*y.dfdt + x.dfdt*y.f + x.dfdx*y.dfdx
    )
```

### Example: Geometric Brownian Motion

Consider geometric Brownian motion in (1+1)-dimensions given by the stochastic differential equation

$$dS = \mu S dt + \sigma S dx$$

with closed-form solution

$$S(x,t) = S_0 \exp{\left[\sigma x + (\mu-\frac{\sigma^2}{2})t\right]}$$

The partial derivatives are

$$\partial_x S(x,t) = \sigma S(x,t)$$

and

$$\widetilde{\partial_t} S(x,t) = \left(\partial_t + \frac{1}{2}\partial^2_x \right) S(x,t) = \mu S(x,t).$$

Note the form of the solution to the stochastic differential equation coincides with the form of previous differential equation if

$$\sigma = \alpha = 3\quad\text{and}\quad \mu = \beta + \frac{\alpha^2}{2} = 8.5.$$


Therefore, we'll similarly define

```julia
julia> fstoch(x,t) = f(Stochastic(x,1,0),Stochastic(t,0,1))
```

Finally, we evaluate the function at $x = 0, y = 0$:

```julia
julia> fstoch(0,0)
julia> fstoch(0,0)
Stochastic:
  f: 1.0
  dfdx: 3.0 # mu
  dfdt: 8.5 # sigma
```

and

```julia
julia> x = rand(); t = rand(); fstoch(x,t)/fstoch(x,t).f
Stochastic:
  f: 1.0
  dfdx: 3.0 # mu
  dfdt: 8.5 # sigma
```

as expected.

### Sanity Checks

Check inverses:

```julia
julia> x = Stochastic(rand(),rand(),rand())
Stochastic:
  f: 0.6646540596826009
  dfdx: 0.9674190744363276
  dfdt: 0.1327748608375081

julia> inv(x)
Stochastic:
  f: 1.5045420778405239
  dfdx: -2.189895154014495
  dfdt: 2.886886719560768

julia> 1.0/x
Stochastic:
  f: 1.5045420778405239
  dfdx: -2.189895154014495
  dfdt: 2.886886719560768

julia> x^(-1.0)
Stochastic:
  f: 1.5045420778405239
  dfdx: -2.189895154014495
  dfdt: 2.886886719560768

julia> inv(x)*x
Stochastic:
  f: 1.0
  dfdx: 2.220446049250313e-16
  dfdt: -4.440892098500626e-16
```

Check division:

```julia
julia> y = Stochastic(rand(),rand(),rand())
Stochastic:
  f: 0.24628623901245295
  dfdx: 0.9421598146810133
  dfdt: 0.4889415699838624

julia> x/y
Stochastic:
  f: 2.6987056294647225
  dfdx: -6.39578129746468
  dfdt: 34.67486911938278

julia> x*y^(-1)
Stochastic:
  f: 2.6987056294647225
  dfdx: -6.395781297464681
  dfdt: 19.648331286416997
```

Check $\log\left[\exp(x)\right] == \exp\left[\log(x)\right] == x$:

```julia
julia> exp(x)
Stochastic:
  f: 1.9438179600084473
  dfdx: 1.8804865717440824
  dfdt: 1.167699448496872

julia> log(x)
Stochastic:
  f: -0.4084885846865794
  dfdx: 1.4555227043949888
  dfdt: -0.8595078064952

julia> log(exp(x))
Stochastic:
  f: 0.6646540596826009
  dfdx: 0.9674190744363275
  dfdt: 0.13277486083750817

julia> log(exp(x))-x
Stochastic:
  f: 0.0
  dfdx: -1.1102230246251565e-16
  dfdt: 5.551115123125783e-17

julia> exp(log(x))-x
Stochastic:
  f: 0.0
  dfdx: 0.0
  dfdt: 0.0
```

Check $\exp\left[-\log(x)\right] = x^{-1}$

```julia
julia> exp(-log(x))
Stochastic:
  f: 1.5045420778405239
  dfdx: -2.189895154014495
  dfdt: 2.886886719560768

julia> inv(x)
Stochastic:
  f: 1.5045420778405239
  dfdx: -2.189895154014495
  dfdt: 2.886886719560768
```
