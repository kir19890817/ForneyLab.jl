module GammaTest

using Test
using ForneyLab
using ForneyLab: prod!, unsafeMean, unsafeVar, outboundType, isApplicable, dims, isProper
using ForneyLab: SPGammaOutVPP, VBGammaOut

@testset "dims" begin
    @test dims(ProbabilityDistribution(Univariate, Gamma, a=1.0, b=1.0)) == 1
    @test dims(ProbabilityDistribution(Multivariate, Gamma, a=ones(2), b=ones(2))) == 2
end

@testset "vague" begin
    @test vague(Gamma) == ProbabilityDistribution(Univariate, Gamma, a=1.0, b=tiny)
    @test vague(Gamma, 2) == ProbabilityDistribution(Multivariate, Gamma, a=ones(2), b=tiny*ones(2))
end

@testset "prod!" begin
    @test ProbabilityDistribution(Univariate, Gamma, a=1.0, b=2.0) * ProbabilityDistribution(Univariate, Gamma, a=3.0, b=4.0) == ProbabilityDistribution(Univariate, Gamma, a=3.0, b=6.0)
    @test ProbabilityDistribution(Univariate, Gamma, a=1.0, b=2.0) * ProbabilityDistribution(Univariate, PointMass, m=1.0) == ProbabilityDistribution(Univariate, PointMass, m=1.0)
    @test ProbabilityDistribution(Univariate, PointMass, m=1.0) * ProbabilityDistribution(Univariate, Gamma, a=1.0, b=2.0) == ProbabilityDistribution(Univariate, PointMass, m=1.0)
    @test_throws Exception ProbabilityDistribution(Univariate, PointMass, m=-1.0) * ProbabilityDistribution(Univariate, Gamma, a=1.0, b=2.0)

    @test ProbabilityDistribution(Multivariate, Gamma, a=ones(2), b=fill(2.0, 2)) * ProbabilityDistribution(Multivariate, Gamma, a=fill(3.0, 2), b=fill(4.0, 2)) == ProbabilityDistribution(Multivariate, Gamma, a=fill(3.0, 2), b=fill(6.0, 2))
    @test ProbabilityDistribution(Multivariate, Gamma, a=ones(2), b=fill(2.0, 2)) * ProbabilityDistribution(Multivariate, PointMass, m=ones(2)) == ProbabilityDistribution(Multivariate, PointMass, m=ones(2))
    @test ProbabilityDistribution(Multivariate, PointMass, m=ones(2)) * ProbabilityDistribution(Multivariate, Gamma, a=ones(2), b=fill(2.0, 2)) == ProbabilityDistribution(Multivariate, PointMass, m=ones(2))
    @test_throws Exception ProbabilityDistribution(Multivariate, PointMass, m=-ones(2)) * ProbabilityDistribution(Multivariate, Gamma, a=ones(2), b=fill(2.0, 2))
end

@testset "unsafe mean and variance" begin
    @test unsafeMean(ProbabilityDistribution(Univariate, Gamma, a=1.0, b=2.0)) == 0.5
    @test unsafeMean(ProbabilityDistribution(Multivariate, Gamma, a=fill(1.0, 2), b=fill(2.0, 2))) == fill(0.5, 2)
    @test unsafeVar(ProbabilityDistribution(Univariate, Gamma, a=1.0, b=2.0)) == 0.25
    @test unsafeVar(ProbabilityDistribution(Multivariate, Gamma, a=fill(1.0, 2), b=fill(2.0, 2))) == fill(0.25, 2)
end

@testset "isProper" begin
    @test isProper(ProbabilityDistribution(Univariate, Gamma, a=1.0, b=1.0)) == true
    @test isProper(ProbabilityDistribution(Multivariate, Gamma, a=ones(2), b=ones(2))) == true
    @test isProper(ProbabilityDistribution(Univariate, Gamma, a=0.0, b=0.0)) == false
    @test isProper(ProbabilityDistribution(Multivariate, Gamma, a=zeros(2), b=zeros(2))) == false
end

#-------------
# Update rules
#-------------

@testset "SPGammaOutVPP" begin
    @test SPGammaOutVPP <: SumProductRule{Gamma}
    @test outboundType(SPGammaOutVPP) == Message{Gamma}
    @test isApplicable(SPGammaOutVPP, [Nothing, Message{PointMass}, Message{PointMass}]) 

    @test ruleSPGammaOutVPP(nothing, Message(Univariate, PointMass, m=1.0), Message(Univariate, PointMass, m=2.0)) == Message(Univariate, Gamma, a=1.0, b=2.0)
end

@testset "VBGammaOut" begin
    @test VBGammaOut <: NaiveVariationalRule{Gamma}
    @test outboundType(VBGammaOut) == Message{Gamma}
    @test isApplicable(VBGammaOut, [Nothing, ProbabilityDistribution, ProbabilityDistribution]) 
    @test !isApplicable(VBGammaOut, [ProbabilityDistribution, ProbabilityDistribution, Nothing]) 

    @test ruleVBGammaOut(nothing, ProbabilityDistribution(Univariate, PointMass, m=1.5), ProbabilityDistribution(Univariate, PointMass, m=3.0)) == Message(Univariate, Gamma, a=1.5, b=3.0)
end

@testset "averageEnergy and differentialEntropy" begin
    @test differentialEntropy(ProbabilityDistribution(Univariate, Gamma, a=1.0, b=2.0)) == averageEnergy(Gamma, ProbabilityDistribution(Univariate, Gamma, a=1.0, b=2.0), ProbabilityDistribution(Univariate, PointMass, m=1.0), ProbabilityDistribution(Univariate, PointMass, m=2.0))
end

end #module
