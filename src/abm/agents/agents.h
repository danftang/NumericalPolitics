//
// Agent body/Environment:
//   The body has no decision capability but defines the actions/messages that can be sent/received,
//  their effect on the state of the body and the physical reward (i.e. the physical semantics of the messages)
//
//  class Body {
//    class State {
//      double transition(message_type myMessage, message_type yourResponse)
//                                              performs the transition and returns the reward,
//                                              if myMessage is null (open channel), must be first move of second mover.
//                                              If both messages are null, I am the first mover on a newly opened channel.
//      std::bitset legalIntents()           actions that can legally be performed from the current agent state.
//    }
//    class message_type {
//      static bool isTerminal(message_type act, message_type response) does the act/response pair terminate the episode (assumes episodic comms)?
//                                              N.B. the comms protocol should have an unambiguous end that doesn't depend
//                                              on agent state
//    }
// }
//
//
//
// Created by daniel on 15/05/23.
//

#ifndef MULTIAGENTGOVERNMENT_AGENTS_H
#define MULTIAGENTGOVERNMENT_AGENTS_H

#include "ParallelPairingManager.h"
#include "PrisonersDilemmaAgent.h"
#include "SequentialPairingManager.h"
#include "SimpleSugarSpiceAgent.h"
#include "SugarSpiceAgentWithFriends.h"
#include "SugarSpiceTradingAgent.h"

#endif //MULTIAGENTGOVERNMENT_AGENTS_H
