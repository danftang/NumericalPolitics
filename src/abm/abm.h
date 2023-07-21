// In this view, an ABM consists of a set of "agent" objects that implement a set of "handler" methods
// that deal with incoming messages from other agents. A handler returns a response to the incoming message in
// the form of a schedule of messages to be sent to other agents.
//
// A schedule can contain any executable, but we provide the CommunicationChannel class as a convenient way
// to pass messages between agents via a schedule. A CommunicationChannel object should be owned by an agent
// that wants to send a message. The channel is first connected to another agent's handler using the connectTo
// method. Messages can then be sent down the channel using the send method, which returns a schedule that
// contains the execution of the receiving agent's handler (rather than calling it directly).
//
// An ABM is executed by creating a schedule with some initial messages (by convention, certain agents have
// a start method which returns a schedule of initial messages) then calling exec or execUntil on the schedule.
//
// A reinforcement learning agent has:
//   - a map from <internal-state, received-message> pairs to <new-internal-state>
//   - a reward function from <internal-state, received-message> to reward  (or <old-state, new-state>?)
//   - a map from internal-state to acts (i.e. a decision process - possibly stochastic)
//   - a map from acts to schedules (i.e responses to received messages) (equivalently, but with smaller act space
//     from <state,act> to schedule).
//
// This is equivalent to a state machine with state transitions driven by incoming messages and states
// associated with acts (or probability distributions over acts). So, the ABM is
// a set of linked state machines.
//
// From a more procedural point of view, an agent handler should:
//  - calculate reward and update internal state
//  - train or record for later training on old state, last act, reward and new state.
//  - decide on an act (and remember for next training step)
//  - return the associated schedule
//
// A single training step consists of internal-state, act, reward, new-internal-state.
//
// What to do about internal states that have only one act?
// If they have a reward, they should
// be included in the Q-table. If they have no reward, they can be absorbed into the new-internal-state
// of the next state that either has a reward or a where a decision must be made. The intermediate states
// then just become part of the probability of transition between states in the table.
// (rewards could also be summed and lumped into the next decision point, so that only states that require
// a decision are in the table, and rewards become probabilistic, but this may slow learning?)
//
// Created by daniel on 20/02/23.
//

#ifndef MULTIAGENTGOVERNMENT_ABM_H
#define MULTIAGENTGOVERNMENT_ABM_H

#include "Schedule.h"
#include "CommunicationChannel.h"
#include "QTablePolicy.h"
#include "DQNPolicy.h"
#include "MlPackAction.h"
#include "QTable.h"

#endif //MULTIAGENTGOVERNMENT_ABM_H
