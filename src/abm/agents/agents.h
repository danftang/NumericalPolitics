//
// Agent body/Environment:
//   The body has no decision capability but defines the actions/messages that can be sent/received,
//  their effect on the state of the body and the physical reward (i.e. the physical semantics of the messages)
//
//  class Body {
//    class State {
//      double transition(Action myMessage, Action yourResponse)
//                                              performs the transition and returns the reward,
//                                              if myMessage is null (open channel), must be first move of second mover.
//                                              If both messages are null, I am the first mover on a newly opened channel.
//      std::bitset legalActions()           actions that can legally be performed from the current agent state.
//    }
//    class Action {
//      static bool isTerminal(Action act, Action response) does the act/response pair terminate the episode (assumes episodic comms)?
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
