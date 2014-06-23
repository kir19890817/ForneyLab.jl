module ForneyLab

export  Message, Node, CompositeNode, Interface, Edge
export  calculateMessage!, calculateMessages!, calculateForwardMessage!, calculateBackwardMessage!,
        calculateMarginal, calculateMarginal!,
        getMessage, getForwardMessage, getBackwardMessage, setMessage!, setForwardMessage!, setBackwardMessage!, clearMessage!, clearMessages!, clearAllMessages!,
        generateSchedule, executeSchedule

# Verbosity
verbose = false
setVerbose(verbose_mode=true) = global verbose=verbose_mode
printVerbose(msg) = if verbose println(msg) end

# Helpers
include("helpers.jl")

# Other includes
import Base.show

# Top-level abstracts
abstract Message
abstract RootEdge # An Interface belongs to an Edge, but Interface is defined before Edge. Because you can not belong to something undefined, Edge will inherit from RootEdge, solving this problem.

abstract Node
show(io::IO, node::Node) = println(io, typeof(node), " with name ", node.name, ".")
abstract CompositeNode <: Node

type Interface
    # An Interface belongs to a node and is used to send/receive messages.
    # An Interface has exactly one partner interface, with wich it forms an edge.
    # An Interface can be seen as a half-edge, that connects to a partner Interface to form a complete edge.
    # A message from node a to node b is stored at the Interface of node a that connects to an Interface of node b.
    node::Node
    edge::Union(RootEdge, Nothing)
    partner::Union(Interface, Nothing) # Partner indicates the interface to which it is connected.
    child::Union(Interface, Nothing)   # An interface that belongs to a composite has a child, which is the corresponding (effectively the same) interface one lever deeper in the node hierarchy.
    message::Union(Message, Nothing)
    message_dependencies::Array{Interface, 1}   # Optional array of interfaces (of the same node) on which the outbound msg on this interface depends.
                                                # If this array is #undef, it means that the outbound msg depends on the inbound msgs on ALL OTHER interfaces of the node.
    internal_schedule::Array{Interface, 1}      # Optional schedule that should be executed to calculate outbound message on this interface.
                                                # The internal_schedule field is used in composite nodes, and holds the schedule for internal message passing.
    # Sanity check for matching message types
    function Interface(node::Node, edge::Union(RootEdge, Nothing)=nothing, partner::Union(Interface, Nothing)=nothing, child::Union(Interface, Nothing)=nothing, message::Union(Message, Nothing)=nothing)
        if typeof(partner) == Nothing || typeof(message) == Nothing # Check if message or partner exist
            return new(node, edge, partner, child, message)
        elseif typeof(message) != typeof(partner.message) # Compare message types
            error("Message type of partner does not match with interface message type")
        else
            return new(node, edge, partner, child, message)
        end
    end
end
Interface(node::Node, message::Message) = Interface(node, nothing, nothing, nothing, message)
Interface(node::Node) = Interface(node, nothing, nothing, nothing, nothing)
show(io::IO, interface::Interface) = println(io, "Interface of $(typeof(interface.node)) with node name $(interface.node.name) holds message of type $(typeof(interface.message)).")
setMessage!(interface::Interface, message::Message) = (interface.message=message)
clearMessage!(interface::Interface) = (interface.message=nothing)
getMessage(interface::Interface) = interface.message

function show(io::IO, schedule::Array{Interface, 1})
    # Show schedules in a specific way
    println(io, "Message passing schedule:")
    for interface in schedule
        println(io, " $(typeof(interface.node)) $(interface.node.name)")
    end
end

type Edge <: RootEdge
    # An Edge joins two interfaces and has a direction (from tail to head).
    # Edges are mostly useful for code readability, they are not used internally.
    # Forward messages flow in the direction of the Edge (tail to head).
    # Edges can contain marginals, which are the product of the forward and backward message.

    tail::Interface
    head::Interface
    marginal::Union(Message, Nothing) # Messages are probability distributions

    function Edge(tail::Interface, head::Interface, marginal::Union(Message, Nothing)=nothing)
        if  typeof(head.message) == Nothing ||
            typeof(tail.message) == Nothing ||
            typeof(head.message) == typeof(tail.message)
            if !is(head.node, tail.node)
                self = new(tail, head, marginal)
                # Assign pointed to edge from interfaces
                tail.edge = self
                head.edge = self
                # Partner head and tail, and merge their families
                tail.partner = head
                head.partner = tail
                # Backreferences for tail's children
                child_interface = tail.child
                while child_interface != nothing
                    child_interface.partner = tail.partner
                    child_interface = child_interface.child
                end
                # Backreferences for head's children
                child_interface = head.child
                while child_interface != nothing
                    child_interface.partner = head.partner
                    child_interface = child_interface.child
                end
                return self
            else
                error("Cannot connect two interfaces of the same node: ", typeof(head.node), " ", head.node.name)
            end
        else
            error("Head and tail message types do not match: ", typeof(head.message), " and ", typeof(tail.message))
        end
    end
end

function Edge(tail_node::Node, head_node::Node, marginal::Union(Message, Nothing)=nothing)
    # Create an Edge from tailNode to headNode.
    # Use the first free interface on each node.
    tail = nothing
    head = nothing
    for interface in tail_node.interfaces
        if interface.partner==nothing
            tail = interface
            break
        end
    end
    @assert(tail!=nothing, "Cannot create edge: no free interface on tail node: ", typeof(tail_node), " ", tail_node.name)

    for interface in head_node.interfaces
        if interface.partner==nothing
            head = interface
            break
        end
    end
    @assert(head!=nothing, "Cannot create edge: no free interface on head node: $(typeof(head_node)) $(head_node.name)")

    return Edge(tail, head, marginal)
end
show(io::IO, edge::Edge) = println(io, "Edge from $(typeof(edge.tail.node)) with node name $(edge.tail.node.name) to $(typeof(edge.head.node)) with node name $(edge.head.node.name) holds marginal of type $(typeof(edge.marginal)). Forward message type: $(typeof(edge.tail.message)). Backward message type: $(typeof(edge.head.message)).")
setForwardMessage!(edge::Edge, message::Message) = setMessage!(edge.tail, message)
setBackwardMessage!(edge::Edge, message::Message) = setMessage!(edge.head, message)
getForwardMessage(edge::Edge) = edge.tail.message
getBackwardMessage(edge::Edge) = edge.head.message

# Messages
include("messages.jl")

# Nodes
include("nodes/addition.jl")
include("nodes/constant.jl")
include("nodes/equality.jl")
include("nodes/fixed_gain.jl")
include("nodes/gaussian.jl")
# Composite nodes
include("nodes/composite/gain_addition.jl")
include("nodes/composite/gain_equality.jl")
include("nodes/composite/general.jl")

#############################
# Generic methods
#############################

function calculateMessage!(outbound_interface::Interface)
    # Calculate the outbound message on a specific interface by generating a schedule and executing it.
    # The resulting message is stored in the specified interface and returned.

    # Generate a message passing schedule
    printVerbose("Auto-generating message passing schedule...")
    schedule = generateSchedule(outbound_interface)
    if verbose show(schedule) end

    # Execute the schedule
    printVerbose("Executing above schedule...")
    executeSchedule(schedule)
    printVerbose("calculateMessage!() done.")

    return outbound_interface.message
end

function updateNodeMessage!(outbound_interface::Interface)
    # Calculate the outbound message based on the inbound messages and the node update function.
    # The resulting message is stored in the specified interface and returned.

    node = outbound_interface.node

    # Determine types of inbound messages
    inbound_message_types = Union() # Union of all inbound message types
    outbound_interface_id = 0
    for node_interface_id = 1:length(node.interfaces)
        node_interface = node.interfaces[node_interface_id]
        if is(node_interface, outbound_interface)
            outbound_interface_id = node_interface_id
        end
        if (!isdefined(outbound_interface, :message_dependencies) && outbound_interface_id==node_interface_id) ||
           (isdefined(outbound_interface, :message_dependencies) && !(node_interface in outbound_interface.message_dependencies))
            continue
        end
        @assert(node_interface.partner!=nothing, "Cannot receive messages on disconnected interface $(node_interface_id) of $(typeof(node)) $(node.name)")
        @assert(node_interface.partner.message!=nothing, "There is no inbound message present on interface $(node_interface_id) of $(typeof(node)) $(node.name)")
        inbound_message_types = Union(inbound_message_types, typeof(node_interface.partner.message))
    end

    # Evaluate node update function
    printVerbose("Calculate outbound message on $(typeof(node)) $(node.name) interface $outbound_interface_id")
    msg = updateNodeMessage!(outbound_interface_id, node, inbound_message_types)

    return msg
end

function calculateMessages!(node::Node)
    # Calculate the outbound messages on all interfaces of node.
    for interface in node.interfaces
        calculateMessage!(interface)
    end
end

# Calculate forward/backward messages on an Edge
calculateForwardMessage!(edge::Edge) = calculateMessage!(edge.tail)
calculateBackwardMessage!(edge::Edge) = calculateMessage!(edge.head)

function executeSchedule(schedule::Array{Interface, 1})
    # Execute a message passing schedule
    for interface in schedule
        updateNodeMessage!(interface)
    end
    # Return the last message in the schedule
    return schedule[end].message
end

function executeSchedule(schedule::Array{Edge, 1})
    # Execute a marginal update schedule
    for edge in schedule
        calculateMarginal!(edge)
    end
    # Return the last message in the schedule
    return schedule[end].marginal
end

function calculateMarginal(forward_msg::Message, backward_msg::Message)
    # Calculate the marginal from a forward/backward message pair.
    # We calculate the marginal by using an EqualityNode.
    # The forward and backward messages are inbound messages to the EqualityNode.
    # The outbound message is the marginal.
    @assert(typeof(forward_msg)==typeof(backward_msg), "Cannot create marginal from forward/backward messages of different types.")
    eq_node = EqualityNode(3)
    c_node1 = ConstantNode(forward_msg)
    c_node2 = ConstantNode(backward_msg)
    Edge(c_node1.out, eq_node.interfaces[1])
    Edge(c_node2.out, eq_node.interfaces[2])
    c_node1.out.message = deepcopy(c_node1.value) # just do it the quick way
    c_node2.out.message = deepcopy(c_node2.value)
    marginal_msg = updateNodeMessage!(3, eq_node, typeof(forward_msg)) # quick direct call
    return marginal_msg
end

function calculateMarginal!(edge::Edge)
    # Calculates and writes the marginal on edge
    @assert(typeof(edge.tail.message)<:Message, "Edge should hold a forward message.")
    @assert(typeof(edge.head.message)<:Message, "Edge should hold a backward message.")
    msg = calculateMarginal(edge.tail.message, edge.head.message)
    edge.marginal = msg
    return(msg)
end

function clearMessages!(node::Node)
    # Clear all outbound messages on the interfaces of node
    for interface in node.interfaces
        interface.message = nothing
    end
end

function clearMessages!(edge::Edge)
    # Clear all messages on an edge.
    edge.head.message = nothing
    edge.tail.message = nothing
end

# Functions to clear ALL MESSAGES in the graph
clearAllMessages!(seed_node::Node) = map(clearMessages!, getAllNodesInGraph(seed_node))
clearAllMessages!(seed_edge::Edge) = map(clearMessages!, getAllNodesInGraph(seed_edge.tail))

function generateSchedule(outbound_interface::Interface)
    # Generate a schedule that can be executed to calculate the outbound message on outbound_interface.
    # IMPORTANT: the resulting schedule depends on the current messages stored in the factor graph.
    # The same graph with different messages being present can (and probably will) result in a different schedule.
    # When a lot of iterations of the same message passing schedule are required, it can be very beneficial
    # to generate the schedule just once using this function, and then execute the same schedule over and over.
    # This prevents having to generate the same schedule in every call to calculateMessage!().
    schedule = generateScheduleByDFS(outbound_interface)
end

function generateSchedule(partial_schedule::Array{Interface, 1})
    # Generate a complete schedule based on partial_schedule.
    # A partial schedule only defines the order of a subset of all required messages.
    # This function will find a valid complete schedule that satisfies the partial schedule.
    # IMPORTANT: the resulting schedule depends on the current messages stored in the factor graph.
    # The same graph with different messages being present can (and probably will) result in a different schedule.
    # When a lot of iterations of the same message passing schedule are required, it can be very beneficial
    # to generate the schedule just once using this function, and then execute the same schedule over and over.
    # This prevents having to generate the same schedule in every call to calculateMessage!().
    schedule = Array(Interface, 0)
    for interface_order_constraint in partial_schedule
        schedule = generateScheduleByDFS(interface_order_constraint, schedule)
    end

    return schedule
end

function generateScheduleByDFS(outbound_interface::Interface, backtrace::Array{Interface, 1}=Array(Interface, 0), call_list::Array{Interface, 1}=Array(Interface, 0))
    # This is a private function that performs a search through the factor graph to generate a schedule.
    # IMPORTANT: the resulting schedule depends on the current messages stored in the factor graph.
    # This is a recursive implementation of DFS. The recursive calls are stored in call_list.
    # backtrace will hold the backtrace.
    node = outbound_interface.node

    # Apply stopping condition for recursion. When the same interface is called twice, this is indicative of an unbroken loop.
    if outbound_interface in call_list
        # Notify the user to break the loop with an initial message
        error("Loop detected around $(outbound_interface) Consider setting an initial message somewhere in this loop.")
    else # Stopping condition not satisfied
        push!(call_list, outbound_interface)
    end

    # Check all inbound messages on the other interfaces of the node
    outbound_interface_id = 0
    for node_interface_id = 1:length(node.interfaces)
        node_interface = node.interfaces[node_interface_id]
        if is(node_interface, outbound_interface)
            outbound_interface_id = node_interface_id
        end
        if (!isdefined(outbound_interface, :message_dependencies) && outbound_interface_id==node_interface_id) ||
           (isdefined(outbound_interface, :message_dependencies) && !(node_interface in outbound_interface.message_dependencies))
            continue
        end
        @assert(node_interface.partner!=nothing, "Disconnected interface should be connected: interface #$(node_interface_id) of $(typeof(node)) $(node.name)")

        if node_interface.partner.message == nothing # Required message missing.
            if !(node_interface.partner in backtrace) # Don't recalculate stuff that's already in the schedule.
                # Recursive call
                printVerbose("Recursive call of generateSchedule! on node $(typeof(node_interface.partner.node)) $(node_interface.partner.node.name)")
                generateScheduleByDFS(node_interface.partner, backtrace, call_list)
            end
        end
    end

    # Update call_list and backtrace
    pop!(call_list)

    return push!(backtrace, outbound_interface)
end

function getAllNodesInGraph(seed_node::Node, node_array::Array{Node,1}=Array(Node,0))
    # Return a list of all nodes in the graph
    if !(seed_node in node_array)
        push!(node_array, seed_node)
        for node_interface in seed_node.interfaces
            if node_interface.partner != nothing
                # Recursion
                getAllNodesInGraph(node_interface.partner.node, node_array)
            end
        end
    end

    return node_array
end

try
    # Try to load user-defined extensions
    include("$(Main.FORNEYLAB_EXTENSION_DIR)/src/forneylab_extensions.jl")
end

end # module ForneyLab