//
// Created by daniel on 17/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_RANDOMENCOUNTERSOCIETY_H
#define MULTIAGENTGOVERNMENT_RANDOMENCOUNTERSOCIETY_H

#include <vector>
#include "../../DeselbyStd/random.h"

namespace abm::societies {
    template<class AGENT>
    class RandomEncounterSociety {
    public:
        std::vector<AGENT> agents;

        RandomEncounterSociety(int nAgents): agents(nAgents) {
        }

        void episode() {
            auto &[firstAgent, secondAgent] = chooseAgentPair();
            do {
                firstAgent.reactTo(-1);
            }
        }

        // Randomly choose a pair of agent's without replacement
        std::pair<AGENT &, AGENT &> chooseAgentPair() {
            int firstAgentIndex = deselby::Random::nextInt(0, agents.size());
            int secondAgentIndex;
            do {
                secondAgentIndex = deselby::Random::nextInt(0, agents.size());
            } while(firstAgentIndex == secondAgentIndex);
            return { agents[firstAgentIndex], agents[secondAgentIndex] }
        }
    };

}


#endif //MULTIAGENTGOVERNMENT_RANDOMENCOUNTERSOCIETY_H
