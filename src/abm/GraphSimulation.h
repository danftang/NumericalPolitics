// A graph simulation consists of a number of agents (nodes) connected
// by communication channels (directed edges). Agents can send messages
// down their outgoing edges, which become readable by the destination
// agent at time Dt after being sent. On receipt of a message, an agent
// can choose to respond by sending messages down its outgoing edges after
// some response time has elapsed.
// Edges may be loops, allowing agents to perform periodic computation.
//
// Computationally, an agent has a local time until which
// all outgoing edges are guaranteed to be up to date. Each incoming
// channel has a corresponding time which is the senders local time
// plus the transmission time of the channel.
//
// Let an "event" be the receipt of a message at an incoming edge.
// On processing an event, an agent's local time moves forward
// to the earlier of the next of the agent's events or the earliest
// incoming channel time. The response may cause other events
// and the time of outgoing channels to change. This may cause
// other events to be processed, or be scheduled for processing.
// An agent can either be waiting for an event to be processed or
// be waiting for an incoming channel time to moe forward.
// So, in-fact there are two local times relevent to an agent
// (1) the latest time up to which all input channels are up to date (the channel time).
// (2) the latest time up to which all events have been processed (the local time)
//
// We want agent events to be guaranteed to execute sequentially in time
// order, and at the time of processing, all input channels should be up to date
// at least until the time of the event being processed.
//
// So, at the end of an event being processed, the event is removed from the
// agent's channel. If the agent has any other events
// that are before the channel time then the earliest of these is scheduled.
// If the channel time is earlier than any events and moves forward past
// at least one event, then the earliest of these events is scheduled.
//
// [Could have an (unordered) set of scheduled tasks on each agent which are currently waiting for channel
// updates. Each channel has a list of tasks awaiting on that channel, and tasks are updated on channel
// update events. If a channel update makes all read channels sufficiently up to date for a task, the taek
// gets sent to a simulation-wide task list for immediate execution. In this way, tasks on the central task
// list need not be executed in time order. As long as channels have a finite transfer time, and a task can
// only schedule new tasks in the future w.r.t its current time then this can never block.]
//
// Created by daniel on 22/03/23.
//

#ifndef MULTIAGENTGOVERNMENT_GRAPHSIMULATION_H
#define MULTIAGENTGOVERNMENT_GRAPHSIMULATION_H


class GraphSimulation {
public:

};


#endif //MULTIAGENTGOVERNMENT_GRAPHSIMULATION_H
