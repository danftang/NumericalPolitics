// Implements a Multiagent Partially Observable Markov Decision Process
// An agent in a POMDP must implement two methods:
// observe()
// which is the agent's observation step (the agent should only change its own unobservable state during this phase) and
// step()
// which is the agents timestep where the agent acts. Observation and action occurs through the agent's own communication
// channels or pointers to other agent's observable states, and these should be managed by the agent.
//
//
// Created by daniel on 04/06/23.
//

#ifndef MULTIAGENTGOVERNMENT_POMDPMANAGER_H
#define MULTIAGENTGOVERNMENT_POMDPMANAGER_H


#include <vector>
#include <functional>
#include "TimestepManager.h"

namespace abm {
    namespace agents {

        template<class EXECUTIONPOLICY, class TIMETYPE = uint>
        class POMDPManager {
        public:
            TimestepManager<std::execution::parallel_unsequenced_policy, TIMETYPE> observationFunctions;
            TimestepManager<EXECUTIONPOLICY, TIMETYPE> timestepFunctions;

            POMDPManager(const EXECUTIONPOLICY &timestepExecPolicy):
            observationFunctions(std::execution::par_unseq, 2),
            timestepFunctions(timestepExecPolicy, 2) { }

            Schedule<TIMETYPE> start() {
                return observationFunctions.start(0) + timestepFunctions.start(1);
            }

        };

    }
}


#endif //MULTIAGENTGOVERNMENT_POMDPMANAGER_H
