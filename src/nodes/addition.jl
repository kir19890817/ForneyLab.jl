############################################
# AdditionNode
############################################
# Description:
#   Addition of two messages of same type.
#
#          in2
#          |
#    in1   v  out
#   ----->[+]----->
#
#   out = in1 + in2
#
#   Example:
#       AdditionNode(; name="my_node")
#
# Interface ids, (names) and supported message types:
#   1. (in1):
#       Message{GaussianDistribution}
#       Message{Float64}
#       Message{Array{Float64}}
#   2. (in2):
#       Message{GaussianDistribution}
#       Message{Float64}
#       Message{Array{Float64}}
#   3. (out):
#       Message{GaussianDistribution}
#       Message{Float64}
#       Message{Array{Float64}}
############################################

export AdditionNode

type AdditionNode <: Node
    name::ASCIIString
    interfaces::Array{Interface,1}
    in1::Interface
    in2::Interface
    out::Interface

    function AdditionNode(; name="unnamed")
        self = new(name, Array(Interface, 3))
        # Create interfaces
        self.interfaces[1] = Interface(self)
        self.interfaces[2] = Interface(self)
        self.interfaces[3] = Interface(self)
        # Init named interface handles
        self.in1 = self.interfaces[1]
        self.in2 = self.interfaces[2]
        self.out = self.interfaces[3]
        return self
    end
end

############################################
# GaussianDistribution methods
############################################

# Rule set for forward propagation, from: Korl (2005), "A Factor graph approach to signal modelling, system identification and filtering", Table 4.1
forwardAdditionMRule{T<:Number}(m_x::Array{T, 1}, m_y::Array{T, 1}) = m_x + m_y
forwardAdditionVRule{T<:Number}(V_x::Array{T, 2}, V_y::Array{T, 2}) = V_x + V_y
forwardAdditionWRule{T<:Number}(W_x::Array{T, 2}, W_y::Array{T, 2}) = W_x * pinv(W_x + W_y) * W_y
forwardAdditionXiRule{T<:Number}(V_x::Array{T, 2}, xi_x::Array{T, 1}, V_y::Array{T, 2}, xi_y::Array{T, 1}) = pinv(V_x + V_y) * (V_x*xi_x + V_y*xi_y)

# Rule set for backward propagation, from: Korl (2005), "A Factor graph approach to signal modelling, system identification and filtering", Table 4.1
# The backward propagation merely negates the mean of the present input message (edge X) and uses the same rules to determine the missing input (edge Y)
# For the sake of clarity there is some redundancy between forward and backward rules.
backwardAdditionMRule{T<:Number}(m_x::Array{T, 1}, m_z::Array{T, 1}) = m_z - m_x
backwardAdditionVRule{T<:Number}(V_x::Array{T, 2}, V_z::Array{T, 2}) = V_x + V_z
backwardAdditionWRule{T<:Number}(W_x::Array{T, 2}, W_z::Array{T, 2}) = W_x * pinv(W_x + W_z) * W_z
backwardAdditionXiRule{T<:Number}(V_x::Array{T, 2}, xi_x::Array{T, 1}, V_z::Array{T, 2}, xi_z::Array{T, 1}) = pinv(V_x + V_z) * (V_z*xi_z - V_x*xi_x)

# Message towards OUT
function updateNodeMessage!(node::AdditionNode,
                            ::Int,
                            ::Type{GaussianDistribution},
                            msg_in1::Message{GaussianDistribution},
                            msg_in2::Message{GaussianDistribution},
                            msg_out::Nothing)
    dist_out = getOrCreateMessage(node.out, GaussianDistribution).value
    dist_1 = msg_in1.value
    dist_2 = msg_in2.value

    # Select parameterization
    # Order is from least to most computationally intensive
    if dist_1.m != nothing && dist_1.V != nothing && dist_2.m != nothing && dist_2.V != nothing
        dist_out.m = forwardAdditionMRule(dist_1.m, dist_2.m)
        dist_out.V = forwardAdditionVRule(dist_1.V, dist_2.V)
        dist_out.W = nothing
        dist_out.xi= nothing
    elseif dist_1.m != nothing && dist_1.W != nothing && dist_2.m != nothing && dist_2.W != nothing
        dist_out.m = forwardAdditionMRule(dist_1.m, dist_2.m)
        dist_out.V = nothing
        dist_out.W = forwardAdditionWRule(dist_1.W, dist_2.W)
        dist_out.xi= nothing
    elseif dist_1.xi != nothing && dist_1.V != nothing && dist_2.xi != nothing && dist_2.V != nothing
        dist_out.m = nothing
        dist_out.V = forwardAdditionVRule(dist_1.V, dist_2.V)
        dist_out.W = nothing
        dist_out.xi= forwardAdditionXiRule(dist_1.V, dist_1.xi, dist_2.V, dist_2.xi)
    else
        # Last resort: calculate (m,V) parametrization for both inbound messages
        ensureMVParametrization!(dist_1)
        ensureMVParametrization!(dist_2)
        dist_out.m = forwardAdditionMRule(dist_1.m, dist_2.m)
        dist_out.V = forwardAdditionVRule(dist_1.V, dist_2.V)
        dist_out.W = nothing
        dist_out.xi= nothing
    end

    return node.out.message
end

# Message towards IN1 or IN2
function updateNodeMessage!(node::AdditionNode,
                            outbound_interface_id::Int,
                            outbound_message_value_type::Type{GaussianDistribution},
                            msg_in1::Union(Message{GaussianDistribution},Nothing),
                            msg_in2::Union(Message{GaussianDistribution},Nothing),
                            msg_out::Message{GaussianDistribution})
    (outbound_interface_id<3) || error("Invalid interface id ", outbound_interface_id, " for calculating message on ", typeof(node), " ", node.name)
    dist_out = getOrCreateMessage(node.interfaces[outbound_interface_id], outbound_message_value_type).value

    # Calculations for the GaussianDistribution type; Korl (2005), table 4.1
    # Backward message, one message on the incoming edge and one on the outgoing edge.
    dist_1or2 = (outbound_interface_id==1) ? node.interfaces[2].partner.message.value : node.interfaces[1].partner.message.value
    dist_3 = node.interfaces[3].partner.message.value

    # Select parameterization
    # Order is from least to most computationally intensive
    if dist_1or2.m != nothing && dist_1or2.V != nothing && dist_3.m != nothing && dist_3.V != nothing
        dist_out.m = backwardAdditionMRule(dist_1or2.m, dist_3.m)
        dist_out.V = backwardAdditionVRule(dist_1or2.V, dist_3.V)
        dist_out.W = nothing
        dist_out.xi = nothing
    elseif dist_1or2.m != nothing && dist_1or2.W != nothing && dist_3.m != nothing && dist_3.W != nothing
        dist_out.m = backwardAdditionMRule(dist_1or2.m, dist_3.m)
        dist_out.V = nothing
        dist_out.W = backwardAdditionWRule(dist_1or2.W, dist_3.W)
        dist_out.xi = nothing
    elseif dist_1or2.xi != nothing && dist_1or2.V != nothing && dist_3.xi != nothing && dist_3.V != nothing
        dist_out.m = nothing
        dist_out.V = backwardAdditionVRule(dist_1or2.V, dist_3.V)
        dist_out.W = nothing
        dist_out.xi = backwardAdditionXiRule(dist_1or2.V, dist_1or2.xi, dist_3.V, dist_3.xi)
    else
        # Last resort: calculate (m,V) parametrization for both inbound messages
        ensureMVParametrization!(dist_1or2)
        ensureMVParametrization!(dist_3)
        dist_out.m = backwardAdditionMRule(dist_1or2.m, dist_3.m)
        dist_out.V = backwardAdditionVRule(dist_1or2.V, dist_3.V)
        dist_out.W = nothing
        dist_out.xi = nothing
    end

    return node.interfaces[outbound_interface_id].message
end

#############################################
# Float64 and Array{Float64} methods
#############################################

# Message towards OUT
function updateNodeMessage!(node::AdditionNode,
                            ::Int,
                            outbound_message_value_type::Union(Type{Float64}, Type{Vector{Float64}}, Type{Matrix{Float64}}),
                            msg_in1::Union(Message{Float64}, Message{Vector{Float64}}, Message{Matrix{Float64}}),
                            msg_in2::Union(Message{Float64}, Message{Vector{Float64}}, Message{Matrix{Float64}}),
                            msg_out::Nothing)
    ans = msg_in1.value + msg_in2.value
    msg_result = getOrCreateMessage(node.out, outbound_message_value_type, size(ans))
    msg_result.value = ans
    (typeof(msg_result.value) == outbound_message_value_type) || error("Output type $(typeof(msg_result)) of $(typeof(node)) node $(node.name) does not match expected output type $(outbound_message_value_type)")

    return node.out.message
end

# Message towards IN1 or IN2
# TODO: HIER GEBLEVEN!!!!
function updateNodeMessage!(node::AdditionNode,
                            outbound_interface_id::Int,
                            outbound_message_value_type::Union(Type{Float64}, Type{Vector{Float64}}, Type{Matrix{Float64}}),
                            msg_in1::Union(Message{Float64}, Message{Vector{Float64}}, Message{Matrix{Float64}}, Nothing),
                            msg_in2::Union(Message{Float64}, Message{Vector{Float64}}, Message{Matrix{Float64}}, Nothing),
                            msg_out::Union(Message{Float64}, Message{Vector{Float64}}, Message{Matrix{Float64}}))
    if outbound_interface_id == 1
        ans = msg_out.value - msg_in2.value
    elseif outbound_interface_id == 2
        ans = msg_out.value - msg_in1.value
    else
        error("Invalid interface id ", outbound_interface_id, " for calculating message on ", typeof(node), " ", node.name)
    end

    msg_result = getOrCreateMessage(node.interfaces[outbound_interface_id], outbound_message_value_type, size(ans))
    msg_result.value = ans
    (typeof(msg_result.value) == outbound_message_value_type) || error("Output type $(typeof(msg_result)) of $(typeof(node)) node $(node.name) does not match expected output type $(outbound_message_value_type)")

    return node.interfaces[outbound_interface_id].message
end