//
// Created by daniel on 14/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_SIMPLEEPISODE_H
#define MULTIAGENTGOVERNMENT_SIMPLEEPISODE_H

#include <functional>
#include "../Agent.h"

namespace abm::episodes {
    /** An episode where the initial states of the agents are drawn at random from
     * separate distributions and each agent knows both distributions.
     *
     * @tparam AGENT1 The first mover in the episode
     * @tparam AGENT2 The second mover in the episode
     */
    template<class BODY1, class BODY2>
    class SimpleEpisode {
    public:
        std::function<BODY1()> firstMoverPriorBodyDistribution;
        std::function<BODY2()> secondMoverPriorBodyDistribution;

        SimpleEpisode(
                std::function<BODY1()> firstMoverPriorBodyDistribution,
                std::function<BODY2()> secondMoverPriorBodyDistribution) :
        firstMoverPriorBodyDistribution(firstMoverPriorBodyDistribution),
        secondMoverPriorBodyDistribution(secondMoverPriorBodyDistribution) {}

        /** Sample the body states from the priors and run an episode
         * @param agent1
         * @param agent2
         * @param verbose if true, debug info will be printed to stdout
         * @return number of timesteps in the episode
         */
        template<class AGENT1, class AGENT2> requires std::same_as<typename AGENT1::body_type, BODY1> && std::same_as<typename AGENT2::body_type, BODY2>
        inline int run(AGENT1 &agent1, AGENT2 &agent2, bool verbose = false) {
            agent1.initEpisode(firstMoverPriorBodyDistribution(), *this);
            agent2.initEpisode(secondMoverPriorBodyDistribution(), *this);
            return episode(agent1, agent2, verbose);
        }

        /** Set the agent's bodies to those supplied and run an episode
         * @param agent1
         * @param agent2
         * @param initBodyState1
         * @param initBodyState2
         * @param verbose
         * @return
         */
        template<class AGENT1, class AGENT2> requires std::same_as<typename AGENT1::body_type, BODY1> && std::same_as<typename AGENT2::body_type, BODY2>
        inline int run(AGENT1 &agent1, AGENT2 &agent2, BODY1 initBodyState1, BODY2 initBodyState2, bool verbose = false) {
            agent1.initEpisode(std::move(initBodyState1), *this);
            agent2.initEpisode(std::move(initBodyState2), *this);
            return episode(agent1, agent2, verbose);
        }

    };
}

#endif //MULTIAGENTGOVERNMENT_SIMPLEEPISODE_H
