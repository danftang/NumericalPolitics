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
    template<class BODY1, class BODY2 = BODY1>
    class SimpleEpisode {
    public:
        std::function<BODY1()> firstMoverPriorBodyDistribution;
        std::function<BODY2()> secondMoverPriorBodyDistribution;

        SimpleEpisode(
                std::function<BODY1()> firstMoverPriorBodyDistribution,
                std::function<BODY2()> secondMoverPriorBodyDistribution) :
        firstMoverPriorBodyDistribution(std::move(firstMoverPriorBodyDistribution)),
        secondMoverPriorBodyDistribution(std::move(secondMoverPriorBodyDistribution)) {}

        /** Sample the body states from the priors and run an episode
         * @param agent1
         * @param agent2
         * @param verbose if true, debug info will be printed to stdout
         * @return number of timesteps in the episode
         */
        template<class AGENT1, class AGENT2> requires std::same_as<typename AGENT1::body_type, BODY1> && std::same_as<typename AGENT2::body_type, BODY2>
        inline int run(AGENT1 &agent1, AGENT2 &agent2, bool verbose = false) const;

        /** Set the agent's bodies to those supplied and run an episode
         * @param agent1
         * @param agent2
         * @param initBodyState1
         * @param initBodyState2
         * @param verbose
         * @return
         */
        template<class AGENT1, class AGENT2> requires std::same_as<typename AGENT1::body_type, BODY1> && std::same_as<typename AGENT2::body_type, BODY2>
        inline int run(AGENT1 &agent1, AGENT2 &agent2, BODY1 initBodyState1, BODY2 initBodyState2, bool verbose = false) const;

        BODY1 sampleFirstMoverBody() const { return firstMoverPriorBodyDistribution(); }
        BODY1 sampleSecondMoverBody() const { return secondMoverPriorBodyDistribution(); }
    };



    /** Execute a single episode between two agents
     *
     * @tparam BODY0
     * @tparam MIND0
     * @tparam BODY1
     * @tparam MIND1
     * @param agent0
     * @param agent1
     * @param verbose
     * @return
     */
    template<class AGENT0, class AGENT1>
    requires std::is_convertible_v<typename AGENT0::out_message_type, typename AGENT1::in_message_type> &&
             std::is_convertible_v<typename AGENT1::out_message_type, typename AGENT0::in_message_type>
    int episode(AGENT0 &agent0, AGENT1 &agent1, bool verbose = false) {
        if(verbose) {
            std::cout << "------- Starting episode -------" << std::endl;
            std::cout  << agent0.body << agent1.body << std::endl;
        }
        std::optional<typename AGENT0::out_message_type> message0 = agent0.startEpisode();
        std::optional<typename AGENT1::out_message_type> message1;
        int nMessages = 0;
        do {
            if (verbose) std::cout << "--> " << message0 << std::endl;
            message1 = agent1.handleMessage(message0.value());
            ++nMessages;
            if(!message1.has_value()) break;
            if (verbose) std::cout << "<-- " << message1 << std::endl;
            message0 = agent0.handleMessage(message1.value());
            ++nMessages;
        } while(message0.has_value());
        agent0.endEpisode();
        agent1.endEpisode();
        if(verbose) {
            std::cout << agent0.body << agent1.body << std::endl;
            std::cout << "------- Ending episode -------" << std::endl;
        }
        return nMessages;
    }


//    template<Body BODY0, Mind MIND0, Body BODY1, Mind MIND1>
//    requires std::is_convertible_v<typename Traits<BODY0>::out_message_type, typename BODY1::in_message_type> &&
//             std::is_convertible_v<typename Traits<BODY1>::out_message_type, typename BODY0::in_message_type>
    template<class AGENT0, class AGENT1>
    requires std::is_convertible_v<typename AGENT0::out_message_type, typename AGENT1::in_message_type> &&
             std::is_convertible_v<typename AGENT1::out_message_type, typename AGENT0::in_message_type>
    int synchronousEpisode(AGENT0 &agent0, AGENT1 &agent1, bool verbose = false) {
        if (verbose) {
            std::cout << "------- Starting episode -------" << std::endl;
            std::cout << agent0.body << agent1.body << std::endl;
        }
        std::optional<typename AGENT0::out_message_type> message0 = agent0.startEpisode();
        std::optional<typename AGENT1::out_message_type> message1 = agent1.startEpisode();
        int nMessages = 0;
        while (message0.has_value() && message1.has_value()) {
            if (verbose) std::cout << message0 << " <--> " << message1 << std::endl;
            auto tmpMessage = agent1.handleMessage(message0.value());
            message0 = agent0.handleMessage(message1.value());
            message1 = tmpMessage;
            ++nMessages;
        }
        agent0.endEpisode();
        agent1.endEpisode();
        if (verbose) {
            std::cout << agent0.body << agent1.body << std::endl;
            std::cout << "------- Ending episode -------" << std::endl;
        }
        return nMessages;

    }

    template<class BODY1, class BODY2>
    template<class AGENT1, class AGENT2>
    requires std::same_as<typename AGENT1::body_type, BODY1> && std::same_as<typename AGENT2::body_type, BODY2>
    int SimpleEpisode<BODY1, BODY2>::run(AGENT1 &agent1, AGENT2 &agent2, bool verbose) const {
        agent1.init(sampleFirstMoverBody(), *this);
        agent2.init(sampleSecondMoverBody(), *this);
        return episode(agent1, agent2, verbose);
    }

    template<class BODY1, class BODY2>
    template<class AGENT1, class AGENT2>
    requires std::same_as<typename AGENT1::body_type, BODY1> && std::same_as<typename AGENT2::body_type, BODY2>
    int SimpleEpisode<BODY1, BODY2>::run(AGENT1 &agent1, AGENT2 &agent2, BODY1 initBodyState1, BODY2 initBodyState2,
                                         bool verbose) const {
        agent1.init(std::move(initBodyState1), *this);
        agent2.init(std::move(initBodyState2), *this);
        return episode(agent1, agent2, verbose);
    }

}

#endif //MULTIAGENTGOVERNMENT_SIMPLEEPISODE_H
