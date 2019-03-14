export Gamma

"""
Description:

    A gamma node with shape-rate parameterization:

    f(out,a,b) = Gam(out|a,b) = 1/Γ(a) b^a out^{a - 1} exp(-b out)

Interfaces:

    1. out
    2. a (shape)
    3. b (rate)

Construction:

    Gamma(out, a, b, id=:some_id)
"""
mutable struct Gamma <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function Gamma(out, a, b; id=generateId(Gamma))
        @ensureVariables(out, a, b)
        self = new(id, Array{Interface}(undef, 3), Dict{Symbol,Interface}())
        addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = associate!(Interface(self), out)
        self.i[:a] = self.interfaces[2] = associate!(Interface(self), a)
        self.i[:b] = self.interfaces[3] = associate!(Interface(self), b)

        return self
    end
end

slug(::Type{Gamma}) = "Gam"

format(dist::ProbabilityDistribution{V, Gamma}) where V<:VariateType = "$(slug(Gamma))(a=$(format(dist.params[:a])), b=$(format(dist.params[:b])))"

ProbabilityDistribution(::Type{Univariate}, ::Type{Gamma}; a=1.0, b=1.0) = ProbabilityDistribution{Univariate, Gamma}(Dict(:a=>a, :b=>b))
ProbabilityDistribution(::Type{Gamma}; a=1.0, b=1.0) = ProbabilityDistribution{Univariate, Gamma}(Dict(:a=>a, :b=>b))
ProbabilityDistribution(::Type{Multivariate}, ::Type{Gamma}; a=[1.0], b=[1.0]) = ProbabilityDistribution{Multivariate, Gamma}(Dict(:a=>a, :b=>b))

dims(dist::ProbabilityDistribution{V, Gamma}) where V<:VariateType = length(dist.params[:a])

vague(::Type{Gamma}) = ProbabilityDistribution(Univariate, Gamma, a=1.0, b=tiny) # Flat prior leads to more stable behaviour than Jeffrey's prior
vague(::Type{Gamma}, dims::Int64) = ProbabilityDistribution(Multivariate, Gamma, a=ones(dims), b=tiny*ones(dims))

unsafeMean(dist::ProbabilityDistribution{V, Gamma}) where V<:VariateType = dist.params[:a] ./ dist.params[:b] # unsafe mean

unsafeLogMean(dist::ProbabilityDistribution{V, Gamma}) where V<:VariateType = digamma.(dist.params[:a]) - log.(dist.params[:b])

unsafeVar(dist::ProbabilityDistribution{V, Gamma}) where V<:VariateType = dist.params[:a] ./ dist.params[:b].^2 # unsafe variance

isProper(dist::ProbabilityDistribution{V, Gamma}) where V<:VariateType = all(dist.params[:a] .>= tiny) && all(dist.params[:b] .>= tiny)

function prod!( x::ProbabilityDistribution{V, Gamma},
                y::ProbabilityDistribution{V, Gamma},
                z::ProbabilityDistribution{V, Gamma}=ProbabilityDistribution(V, Gamma, a=x.params[:a], b=x.params[:b])) where V<:VariateType

    z.params[:a] = x.params[:a] .+ y.params[:a] .- 1.0
    z.params[:b] = x.params[:b] .+ y.params[:b]

    return z
end

@symmetrical function prod!(x::ProbabilityDistribution{V, Gamma},
                            y::ProbabilityDistribution{V, PointMass},
                            z::ProbabilityDistribution{V, PointMass}=ProbabilityDistribution(V, PointMass)) where V<:VariateType

    all(y.params[:m] .> 0.0) || error("PointMass location $(y.params[:m]) should be positive")
    z.params[:m] = y.params[:m]

    return z
end

# Entropy functional
function differentialEntropy(dist::ProbabilityDistribution{V, Gamma}) where V<:VariateType
    sum(lgamma.(dist.params[:a]) -
        (dist.params[:a] .- 1.0) .* digamma.(dist.params[:a]) -
        log.(dist.params[:b]) +
        dist.params[:a])
end

# Average energy functional
function averageEnergy(::Type{Gamma},
                       marg_out::ProbabilityDistribution{V},
                       marg_a::ProbabilityDistribution{V, PointMass},
                       marg_b::ProbabilityDistribution{V}) where V<:VariateType
    sum(lgamma.(marg_a.params[:m]) -
        marg_a.params[:m] .* unsafeLogMean(marg_b) -
        (marg_a.params[:m] .- 1.0) .* unsafeLogMean(marg_out) +
        unsafeMean(marg_b) .* unsafeMean(marg_out))
end
