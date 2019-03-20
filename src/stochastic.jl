module Stochastics

struct Stochastic{F,dFdx,dFdt}
    f::F
    dfdx::dFdx
    dfdt::dFdt
end

Base.:+(x::Stochastic,y::Stochastic) = 
    Stochastic(x.f+y.f,x.dfdx+y.dfdx,x.dfdt+y.dfdt)

Base.:-(x::Stochastic) = Stochastic(-x.f,-x.dfdx,-x.dfdt)
Base.:-(x::Stochastic,y::Stochastic) = +(x,-y)

Base.:*(x::Stochastic,k::R) where R <:Real = Stochastic(k*x.f,k*x.dfdx,k*x.dfdt)
Base.:*(k::R,x::Stochastic) where R <:Real = x*k

function Base.inv(y::Stochastic)
    Stochastic(
        1/y.f,
        -y.dfdx/y.f^2,
        -y.f^(-2)*y.dfdt + y.f^(-3)*y.dfdx^2)

Base.:^(x::Stochastic,k::R) where R <:Number = 
    Stochastic(
        x.f^k,
        k*x.f^(k-1)*x.dfdx,
        k*x.f^(k-1)*x.dfdt + .5*k*(k-1)*x.f^(k-2)*x.dfdx^2)

Base.:/(x::Stochastic,k::R) where R <:Real = Stochastic(x.f/k,x.dfdx/k,x.dfdt/k)
Base.:/(k::R,y::Stochastic) where R <:Number = k*inv(y)

# Check 1/x == x^(-1) == x^(-1.0)

Base.:*(x::Stochastic,y::Stochastic) =
    Stochastic(
        x.f*y.f,
        x.f*y.dfdx + x.dfdx*y.f,
        x.f*y.dfdt + x.dfdt*y.f + x.dfdx*y.dfdx
    )

Base.:/(x::Stochastic,y::Stochastic) = 
    Stochastic(
        x.f/y.f,
        (x.dfdx*y.f - x.f*y.dfdx)/y.f^2,
        (x.dfdt*y.f^2 + x.f*(y.dfdx^2 - y.f*y.dfdt))/y.f^3
    )

# Check x/y == x*y^(-1) 

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

Base.:^(x::Stochastic,y::Stochastic) = exp(y*log(x))

# Check log(exp(x)) == x

# Check exp(y*log(x)) == x^y (for y = Real and y = Stochastic)

end