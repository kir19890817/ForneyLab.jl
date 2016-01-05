############################################
# NormalGammaDistribution
############################################
# Description:
#   Encodes a normal-gamma distribution.
#   Pamameters: m (location), beta (precision) scalars a (shape) and b (rate).
############################################
export NormalGammaDistribution

type NormalGammaDistribution <: MultivariateProbabilityDistribution
    # All univariate, so parameters are floats
    m::Float64    # location
    beta::Float64 # precision
    a::Float64    # shape
    b::Float64    # rate
end

NormalGammaDistribution(; m=0.0, beta=1.0, a=1.0, b=1.0) = NormalGammaDistribution(m, beta, a, b)

# TODO: verify
function vague!(dist::NormalGammaDistribution)
    dist.m = 0.0
    dist.beta = 1.0
    dist.a = 1.0-tiny
    dist.b = tiny
    return dist
end

isProper(dist::NormalGammaDistribution) = ((dist.beta >= tiny) && (dist.a >= tiny)  && (dist.b >= tiny))

format(dist::NormalGammaDistribution) = "Ng(m=$(format(dist.m)), β=$(format(dist.beta)), a=$(format(dist.a)), b=$(format(dist.b)))"

show(io::IO, dist::NormalGammaDistribution) = println(io, format(dist))

==(x::NormalGammaDistribution, y::NormalGammaDistribution) = (x.m==y.m && x.beta==y.beta && x.a==y.a && x.b==y.b)