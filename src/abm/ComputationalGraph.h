//
// Created by daniel on 11/09/23.
//

#ifndef MULTIAGENTGOVERNMENT_COMPUTATIONALGRAPH_H
#define MULTIAGENTGOVERNMENT_COMPUTATIONALGRAPH_H

/** The computational graph view of computation:
 * A computational graph is a set of nodes that pass messages to each other. The reception
 * of a message may initiate a node to send other messages, so in this way, the reception of
 * some number of external initiating messages can set up a chain of computations which may
 * cause the internal state of nodes to change and messages to be passed down "output" channels.
 *
 * Compile-time type checking
 * If a node receives a message type that it doesn't know how to handle, we have an error condition.
 * A C++ class can, via templated member functions, function overloading and auto return-type, handle a set of
 * message types and return different message types depending on the message type received.
 * Ideally, we'd like to identify any error conditions at compile-time where
 * this is deducible. Compile-time deduction can proceed until a run-time-set optional
 * message is sent, as we don't know whether a communication chain will stop at this point.
 *
 * If there is no intersection between the output message types of one node and the
 * handled message types of another, we can say there is no way they can be connected.
 * Otherwise, they MAY be able to talk.
 *
 * If the delivery strategy specifies the endpoint of every edge (i.e. class, method, input type) and
 * connects all message-passing methods to a compatible endpoint then no error condition
 * can occur. Once we define the type of the initiating message(s), the types of all edges should be
 * defined (this is a little over-specified as edges that may never be used in a computation also need to
 * be type-correct, but that's acceptable).
 *
 * So, an edge needs to connect an (object,method,in_message_type) to an (object,method). Or perhaps
 * (object,method) via message_type to (object, method). If we know the classes of the objects, then
 * we can check that they are compatible. If we define the message type of the edge, then we can
 * check the compatibility of the end-point of the edge irrespective of the start point. We can define
 * a velid computational graph by saying that an ourput that isn't connected by an edge is classed
 * as an output. In this way there can be no runtime errors, but the output may not be as expected
 * (this is possibly worse than a runtime error!).
 *
 * If we consider a node to have only one input type (and possibly a tuple of output types) then
 * a class possibly implements multiple nodes and (class,method,in_message_type) identifies
 * a node class with well defined input/output typing. An edge can then either connect two nodes
 * (output to input) or, if the output is a tuple, can connect (node) to (nodes...) if the
 * output of the source node is a tuple. A delivery strategy then imports a well defined set of
 * nodes from an object into the computational graph.
 *
 *
 * TODO: The message passing view of Everything:
 *   In this view, an object has a set of incoming channels and a set of outgoing channels.
 *   A method is a function from incoming message to (possibly empty)
 *   set/tuple of outgoing messages (with channel identity markers).
 *   [This could be done by returning a set of Callables which, when called, deliver the message by calling
 *   the recipient's operator() with the message payload. Perhaps a DeliverableMessage with a recipient and a
 *   payload and a virtual deliver() method.].
 *   Messages can be sent to self.
 *   A Mind takes only observation messages from body and sends only act messages to body.
 *   A body has a channel from Mind which receives acts, a channel to mind which sends observations and a channel
 *   from the bodies of each other agent it is in contact with [or if channels are strongly typed we could
 *   have a separate channel for each type of message].
 *   Named Channels:
 *   A node can provide named channels (in addition to operator()), in this way channel delivery is
 *   just calling of methods on objects and classes can be checked for channel provision using C++ concepts.
 *   The difference comes in the interpretation of return value, which becomes a value that should be
 *   delivered to some agent.
 *
 * TODO: Think about delivery strategies:
 *   A delivery network can be compile-time or runtime or some combination of both.
 *   The computational network consists of nodes which are instances of member functions, an edge indicates
 *   that the source returns a message addressed to the destination object/member which is called by
 *   the delivery strategy. An object is a set of nodes that share parameters.
 *   A
 *   Any subgraph of the communication network can have its own message delivery strategy, and so a whole
 *   graph may implement many different strategies. In this way, our current Agent can be seen as an
 *   efficient delivery strategy for the Mind/Body pair, which accounts for the fixed number and type
 *   of each message passed. However, a node should function out of the box in any delivery strategy
 *   so we need to design a flexible interface that works for both delayed and immediate delivery.
 *   An object can expect to have some number of output channels that accept a given type of message.
 *   In order to deal with runtime edges, an object can have a runtime input channel which the delivery
 *   agent uses to establish new connections, and a runtime output channel which the object uses to
 *   send messages down runtime-identified channels.
 *   The return type of a method should uniquely identify which of these channels is the intended target.
 *   The delivery strategy should intelligently interpret the return type of a method
 *     - A void return type means no message.
 *     - If a function returns a ReturnToSender<MESSAGE>, the payload should be delivered to the operator ()
 *       of the sender of the message that invoked the function.
 *     - A message can be sent to a compile-time output channel by returning a Message<CHANNELID,MESSAGE>
 *       where CHANNELID is an integer that identifies a fixed output channel of the node. The delivery
 *       strategy should map this to an input channel (agent, method, argument type) at the connection stage.
 *     - A message can be sent to a runtime named channel (method) on an agent by returning a
 *       RuntimeMessage<(AGENT::*)(MESSAGESIG),MESSAGE> that identifies an agent, a non-static method and a message.
 *       The default method is the operator(). [In more generality, the agent and method should be replaced by
 *       a channelID, which is supplied by the delivery strategy during connection, so the node needs to be
 *       templated on ChannelID type. ]
 *     - An unset optional of any message type should be interpreted as no message.
 *     - A tuple of message types or optional message types should be interpreted as multiple messages to be delivered
 *     - Any other return type should be interpreted as being delivered to the default channel of the
 *       channel that returned the value. The default should be determined by the delivery strategy.
 *   Identifying agents:
 *   If we identify agents by ID rather than address, then we allow execution when there is no shared memory
 *   (e.g. remote execution). A node can have input and output "slots" identified by integers, which the
 *   delivery strategy maps to actual other nodes. Sending a message to an unconnected slot becomes a runtime
 *   error (a node can receive notification from the delivery strategy of such events by receiving DeliveryFailure
 *   messages). Nodes could have compile-time slots which can be identified via template parameters, and runtime
 *   slots which are channel id's supplied by the delivery strategy when a new channel is opened.
 *   .
 * TODO: perhaps objects can intelligently connect themselves to other objects depending on their abilities,
 *   this is effectively what optional callbacks are doing. An object can have an "environment" of other objects
 *   and can negotiate how to connect. Runtime connections can be made by moving objects in and out of environments.
 *   (this induces yet another directed graph where an edge means the destination is in the environment of the source).
 */

#endif //MULTIAGENTGOVERNMENT_COMPUTATIONALGRAPH_H
