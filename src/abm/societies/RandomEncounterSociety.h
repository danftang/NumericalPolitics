//
// Created by daniel on 17/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_RANDOMENCOUNTERSOCIETY_H
#define MULTIAGENTGOVERNMENT_RANDOMENCOUNTERSOCIETY_H

#include <vector>
#include <functional>

#include "../../DeselbyStd/random.h"
#include "../../DeselbyStd/tupleutils.h"
#include "../episodes/SimpleEpisode.h"

/** A society consists of a number of agents that can communicate with eachother. The object that represents
 * the society initialises the agents, orchestrates the communication by assigning processor time to methods of
 * the agents and allows the user to inspect the behaviour of the society.
 *
 * Every agent receives information (observations) from its environment and acts on its environment.
 * Observations are sent via events to optional on(event) callbacks if present.
 *
 * Events can be monitored by the user by supplying callbacks to the society object. Callback dispatch is
 * calculated at compile time so dispatch has no runtime overhead.
 *
 */
namespace abm::societies {

    /** In a random encounter society, pairs of agents are repeatedly chosen at random (currently uniformly) and
     * allowed to have an asynchronous, episodic interaction (i.e. a random agent starts the episode by sending
     * a message, the other responds etc. until either agent responds with an empty optional which signifies
     * the end of the episode).
     *
     * @tparam AGENTS
     */
    template<class...AGENTS> class RandomEncounterSociety {
    protected:
        std::tuple<std::vector<AGENTS>...>  agents;
    public:
        template<deselby::IsUniquelyConvertibleToTemplate<std::vector>... VECTORS> requires(sizeof...(VECTORS) > 1)
        explicit RandomEncounterSociety(VECTORS &&...vectors): agents(std::forward<VECTORS>(vectors)...) { }


        template<class AGENT>
        void insert(AGENT &&agent) { deselby::push_back(agents, std::forward<AGENT>(agent)); }

        template<class AGENT>
        std::vector<AGENT> &get_vec() { return std::get<std::vector<AGENT>>(agents); }

        // execute n episodes between randomly chosen agents
        // returns the total number of messages passed
        template<class...CALLBACKS>
        void run(uint nEpisodes, CALLBACKS &&...callbacks) {
            auto callbackTuple = std::tie(callbacks...);
            while(nEpisodes != 0) {
                --nEpisodes;
                deselby::ElementID agent1ID = deselby::randomElementIndex(agents, deselby::Random::gen);
                deselby::ElementID agent2ID = deselby::randomElementIndex(agents, deselby::Random::gen);
                deselby::visit_tuple(agents,
                                            [nEpisodes, &callbackTuple](auto &agent1, auto &agent2) {
                                                episodes::runAsync(agent1, agent2, callbackTuple);
                                            },
                                            agent1ID, agent2ID);
            }
        }
    };
    template<deselby::IsUniquelyConvertibleToTemplate<std::vector>... VECTORS> requires(sizeof...(VECTORS)>1 && deselby::AllDifferentAndNotEmpty<VECTORS...>)
    RandomEncounterSociety(VECTORS &&...vectors) -> RandomEncounterSociety<typename deselby::ConvertsToTemplateType<std::remove_reference_t<VECTORS>,std::vector>::value_type...>;

    // ----- Specialization for just one type of agent
    template<class AGENT>
    class RandomEncounterSociety<AGENT> {
    protected:
        std::vector<AGENT> agents;
    public:

        explicit RandomEncounterSociety(size_t nAgents): agents(nAgents) { }
        template<class ...TS> requires (std::same_as<TS,AGENT> && ...)
        explicit RandomEncounterSociety(TS &&... agents): agents{agents...} { }

        void insert(const AGENT &agent) { agents.push_back(agent); }
        void insert(AGENT &&agent) { agents.push_back(std::move(agent)); }


        // execute n episodes between randomly chosen agents
        // returns the total number of messages passed
        template<class... CALLBACKS>
        void run(uint nEpisodes, CALLBACKS... callbacks) {
            auto callbackTuple = std::tie(callbacks...);
            while(nEpisodes != 0) {
                --nEpisodes;
                auto [agent1, agent2] =  chooseAgentPair();
                episodes::runAsync(agent1, agent2, callbackTuple);
            }
        }


        // Randomly choose a pair of agent's without replacement
        std::pair<AGENT &, AGENT &> chooseAgentPair() {
            assert(agents.size() >= 2);
            int firstAgentIndex = deselby::Random::nextInt(0, agents.size());
            int secondAgentIndex = deselby::Random::nextInt(0, agents.size()-1);
            if(secondAgentIndex >= firstAgentIndex) ++secondAgentIndex;
            return { agents[firstAgentIndex], agents[secondAgentIndex] };
        }
    };

    template<class T, class... TS> requires (std::same_as<std::remove_reference_t<T>,std::remove_reference_t<TS>> && ...)
    RandomEncounterSociety(T &&, TS &&...) -> RandomEncounterSociety<std::remove_reference_t<T>>;
}


#endif //MULTIAGENTGOVERNMENT_RANDOMENCOUNTERSOCIETY_H
