//
// Created by daniel on 17/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_RANDOMENCOUNTERSOCIETY_H
#define MULTIAGENTGOVERNMENT_RANDOMENCOUNTERSOCIETY_H

#include <vector>
#include <functional>
#include "../../DeselbyStd/random.h"

namespace abm::societies {
    template<class AGENT>
    class RandomEncounterSociety {
    public:
        std::vector<AGENT> agents;
        bool verbose;

        RandomEncounterSociety(int nAgents): agents(nAgents), verbose(false) { }

        // execute one episode between randomly chosen agents
        // returns the total number of messages passed (not including close) in the episode
        int episode() {
            return episode(chooseAgentPair());
        }

        int episode(std::array<AGENT *,2> players) {
            if(verbose) {
                std::cout << std::endl << "------- Starting game -------";
                std::cout << std::endl << players[0]->body << players[1]->body;
            }
            int nextPlayerIndex = 1;
            int nMessages = 0;
            typename AGENT::message_type lastMessage = players[0]->startDialogue();
            if(verbose) std::cout << lastMessage << std::endl;
            while(lastMessage != AGENT::message_type::close) {
                lastMessage = players[nextPlayerIndex]->reactTo(lastMessage);
                if(verbose) std::cout << lastMessage << std::endl;
                nextPlayerIndex ^= 1;
                ++nMessages;
            }
            lastMessage = players[nextPlayerIndex]->reactTo(lastMessage); // let the other player know it's end of game
            if(verbose) {
                std::cout << lastMessage << std::endl;
                std::cout << std::endl << players[0]->body << players[1]->body;
            }
            return nMessages;
        }


        // execute n episodes between randomly chosen agents
        // returns the total number of messages passed
        int episodes(int totalEpisodes) {
            int nMessages = 0;
            for(int episodeCount = 0; episodeCount < totalEpisodes; ++episodeCount) {
                nMessages += episode();
            }
            return nMessages;
        }


        // Randomly choose a pair of agent's without replacement
        std::array<AGENT *,2> chooseAgentPair() {
            assert(agents.size() >= 2);
            int firstAgentIndex = deselby::Random::nextInt(0, agents.size());
            int secondAgentIndex = deselby::Random::nextInt(0, agents.size()-1);
            if(secondAgentIndex >= firstAgentIndex) ++secondAgentIndex;
            return { &agents[firstAgentIndex], &agents[secondAgentIndex] };
        }
    };

}


#endif //MULTIAGENTGOVERNMENT_RANDOMENCOUNTERSOCIETY_H
