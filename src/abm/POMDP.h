// Implements a Multiagent Partially Observable Markov Decision Process
// An agent in a POMDP must implement two methods:
// observe(ModelState)
// which takes a model state (which is just the set of agents) and returns the agent's observation (of any type)
// step(Observation, actInterface)
// which performs a timestep given the observation and returns nothing. The timestep consists of changing the agent's
// own state and acting on the environment via the actInterface which implements methods which perform
// operations on the model state. Each action must be commutative with actions of the other agents in the same timestep.
//
// Or, agents have pointers to their (observable) neighbours and we just call observe(), which makes the observation
// and step() which performs any mutable operations. Then the
//
// Created by daniel on 04/06/23.
//

#ifndef MULTIAGENTGOVERNMENT_POMDP_H
#define MULTIAGENTGOVERNMENT_POMDP_H

#include <vector>
#include <functional>

class POMDP {
public:
    std::vector<std::function<void()>> agentObservations;
    std::vector<std::function<void()>> agentSteps;

};


#endif //MULTIAGENTGOVERNMENT_POMDP_H
