module Newtonians

export Newtonian

struct Newtonian{F,dFdx,dFdt}
    f::F
    dfdx::dFdx
    dfdt::dFdt
end

Base.:+(x::Newtonian,y::Newtonian) = 
    Newtonian(x.f+y.f,x.dfdx+y.dfdx,x.dfdt+y.dfdt)

Base.:-(x::Newtonian) = Newtonian(-x.f,-x.dfdx,-x.dfdt)
Base.:-(x::Newtonian,y::Newtonian) = +(x,-y)

Base.:*(x::Newtonian,k::R) where R <:Real = Newtonian(k*x.f,k*x.dfdx,k*x.dfdt)
Base.:*(k::R,x::Newtonian) where R <:Real = x*k

function Base.inv(y::Newtonian)
    invy = inv(y.f)
    return invy*Newtonian(1.0,-invy*y.dfdx,-invy*y.dfdt)
end

# Check inv(y)*y == 1

Base.:^(x::Newtonian,k::R) where R <:Number = 
    Newtonian(x.f^k,k*x.f^(k-1)*x.dfdx,k*x.f^(k-1)*x.dfdt)

Base.:/(x::Newtonian,k::R) where R <:Real = Newtonian(x.f/k,x.dfdx/k,x.dfdt/k)
Base.:/(k::R,y::Newtonian) where R <:Number = k*inv(y)

# Check 1/x == x^(-1) == x^(-1.0)

Base.:*(x::Newtonian,y::Newtonian) =
    Newtonian(x.f*y.f,x.f*y.dfdx + x.dfdx*y.f,x.f*y.dfdt + x.dfdt*y.f)

Base.:/(x::Newtonian,y::Newtonian) = 
    Newtonian(
        x.f/y.f,
        (x.dfdx*y.f - x.f*y.dfdx)/y.f^2,
        (x.dfdt*y.f^2 + x.f*(y.dfdx^2 - y.f*y.dfdt))/y.f^3
    )

# Check x/y == x*y^(-1) 

Base.exp(x::Newtonian) = exp(x.f)*Newtonian(1,x.dfdx,x.dfdt)

function Base.log(x::Newtonian)
    invx = inv(x.f)
    return Newtonian(log(x.f),invx*x.dfdx,invx*x.dfdt)
end

Base.:^(x,y::Newtonian) = exp(y*log(x))

# Check log(exp(x)) == x

# Check exp(y*log(x)) == x^y (for x=Real|Newtonian and y= Real|Newtonian)

end