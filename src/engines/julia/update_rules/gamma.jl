export 
ruleVBGammaOut,
ruleSPGammaOutVPP

ruleSPGammaOutVPP(  msg_out::Nothing, 
                    msg_a::Message{PointMass, V},
                    msg_b::Message{PointMass, V}) where V<:VariateType =
    Message(V, Gamma, a=deepcopy(msg_a.dist.params[:m]), b=deepcopy(msg_b.dist.params[:m]))

ruleVBGammaOut( dist_out::Any,
                dist_a::ProbabilityDistribution{V},
                dist_b::ProbabilityDistribution{V}) where V<:VariateType =
    Message(V, Gamma, a=unsafeMean(dist_a), b=unsafeMean(dist_b))
