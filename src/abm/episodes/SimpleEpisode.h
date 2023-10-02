//
// Created by daniel on 14/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_SIMPLEEPISODE_H
#define MULTIAGENTGOVERNMENT_SIMPLEEPISODE_H

#include <functional>
#include <concepts>
#include "../Agent.h"
#include "../Events.h"

namespace abm::episodes {

    namespace callbacks {
        constexpr auto print = [](auto &agent) { std::cout << agent << std::endl; };
        /** Callback object for verbose logging of episodes
         */
        class Verbose {
        public:
            void *firstMover;

            template<class AGENT1, class AGENT2>
            void on(const events::StartEpisode<AGENT1,AGENT2> &event) {
                firstMover = &event.agent1;
                std::cout << "------- Starting episode -------" << std::endl;
                deselby::constexpr_if<deselby::IsStreamable<AGENT1>>(print, event.agent1);
                deselby::constexpr_if<deselby::IsStreamable<AGENT2>>(print, event.agent2);
            }

            template<class SOURCE, class DEST, class MESSAGE>
            void on(const events::Message<SOURCE,DEST,MESSAGE> &event) {
                bool isFromFirstMover = (&event.source == firstMover);
                std::cout << (isFromFirstMover?"--> ":"<-- ") << event.message << std::endl;
            }

            template<class AGENT1, class AGENT2>
            void on(const events::EndEpisode<AGENT1,AGENT2> &event) {
                deselby::constexpr_if<deselby::IsStreamable<AGENT1>>(print, event.agent1);
                deselby::constexpr_if<deselby::IsStreamable<AGENT2>>(print, event.agent2);
                std::cout << "------- Ending episode -------" << std::endl;
            }
        };

        class MessageCounter {
        public:
            size_t nMessages;
            template<class SOURCE, class DEST, class MESSAGE>
            void on(const events::Message<SOURCE,DEST,MESSAGE> & /* event */) {
                ++nMessages;
            }
        };

        // ========================= FUNCTION TO CALLBACK WRAPPER =======================

        /** Wraps a lambda, or other function, so that its call operator is moved to the
         * .on(.) method for interception of events
         */
        template<class FUNCTION>
        class CallbackFunction {
        public:
             FUNCTION function;
             CallbackFunction(FUNCTION function): function(std::move(function)) {}

             template<class EVENT> requires requires(EVENT event) { function(event); }
             inline void on(EVENT &&event) { function(std::forward<EVENT>(event)); }
        };

    }

    /** Runs episodic interactions between two agents of any type.
     * Agents should implement startEpisode() and handleMessage(.) methods.
     * which can return a message of any type or an optional of any type.
     * At the end of an episode, an EndEpisode event will be sent
     * to both agents
     * The supplied callbacks will be called on the following events:
     *   - StartEpisode
     *   - Message
     *   - EndEpisode
     * TODO: add synchronous episode running (for e.g. rock/paper/scissors)
     */
    template<class AGENT0, class AGENT1, class... CALLBACKS>
    class Runner {
    public:
        AGENT0                agent0;
        AGENT1                agent1;
        std::tuple<CALLBACKS...> callbacks;


        template<class FIRSTAGENT, class SECONDAGENT>
        Runner(FIRSTAGENT &&firstMover, SECONDAGENT &&secondMover, CALLBACKS &&...callbacks):
                agent0(std::forward<AGENT0>(firstMover)),
                agent1(std::forward<AGENT0>(secondMover)),
                callbacks(std::forward<CALLBACKS>(callbacks)...) { }

         /** Runs a number of episodes asynchronously (i.e. agents take turns to send messages)
          *
          * @param nEpisodes number of epsiodes to run
          */
        void runAsync() {
            events::callback(events::StartEpisode{agent0, agent1}, callbacks);
            passMessageRightAndRecurse(agent0.startEpisode());
            events::callback(events::EndEpisode{agent0, agent1}, agent0, agent1, callbacks);
        }

    protected:

        template<class MESSAGE>
        void passMessageRightAndRecurse(std::optional<MESSAGE> message) {
            if(!message.has_value()) return;
            events::callback(events::Message{agent0,agent1,*message},callbacks);
            passMessageLeftAndRecurse(agent1.handleMessage(*message)); // tail-recursion will be optimised out (if optimisation is on)
        }

        template<class MESSAGE>
        void passMessageRightAndRecurse(MESSAGE message) {
            events::callback(events::Message{agent0,agent1,message},callbacks);
            passMessageLeftAndRecurse(agent1.handleMessage(message)); // tail-recursion will be optimised out (if optimisation is on)
        }

        template<class MESSAGE>
        void passMessageLeftAndRecurse(std::optional<MESSAGE> message) {
            if(!message.has_value()) return;
            events::callback(events::Message{agent1,agent0,*message},callbacks);
            passMessageRightAndRecurse(agent0.handleMessage(*message));
        }

        template<class MESSAGE>
        void passMessageLeftAndRecurse(MESSAGE message) {
            events::callback(events::Message{agent1,agent0,message},callbacks);
            passMessageRightAndRecurse(agent0.handleMessage(message));
        }

//        template<class MESSAGE0, class MESSAGE1>
//        void passMessagesSynchronously(MESSAGE0 messageFor0, MESSAGE1 messageFor1) {
//            events::callback(events::Message{agent1,agent0,messageFor0},callbacks);
//            events::callback(events::Message{agent0,agent1,messageFor1},callbacks);
//            passMessagesSynchronously(agent1.handleMessage(messageFor1), agent0.handleMessage(messageFor0));
//        }
    };
    // Allow universal reference passing (runner will move rvalue refs and reference lvalue refs)
    template<class AGENT0, class AGENT1, class...CALLBACKS>
    Runner(AGENT0 &&agent0, AGENT1 &&agent1, CALLBACKS &&...callbacks) -> Runner<AGENT0,AGENT1,CALLBACKS...>;

    /** Execute some number of turns-based episodes between two agents
     * @param agent0 first mover agent
     * @param agent1 second mover agent
     * @param verbose it true, prints the passed messages to stdout
     * @return total number of messages sent in the episode
     */
    template<class AGENT0, class AGENT1, class... CALLBACKS>
    inline void runAsync(AGENT0 &&agent0, AGENT1 &&agent1, CALLBACKS &&...callbacks) {
        Runner(std::forward<AGENT0>(agent0), std::forward<AGENT1>(agent1), std::forward<CALLBACKS>(callbacks)...).runAsync();
    }



//    template<class AGENT0, class AGENT1>
//    uint runAsyncEpisode(std::pair<AGENT0,AGENT1> &agentPair, bool verbose = false) {
//        return runAsyncEpisode(agentPair.first, agentPair.second, verbose);
//    }

//    template<class SAMPLER0, class SAMPLER1> // TODO: requires SAMPLER0/1 to have operator()()
//    uint sampleAndRunAsync(SAMPLER0 &&firstMoverSampler, SAMPLER1 &&secondMoverSampler, size_t nEpisodes) {
//        uint totalMessages = 0;
//        while (nEpisodes > 0) {
//            totalMessages += runAsyncEpisode(firstMoverSampler(), secondMoverSampler(), verbose);
//            --nEpisodes;
//        }
//        return totalMessages;
//    }

//    template<class AGENT1, class AGENT2, class SAMPLER, class... CALLBACKS> // TODO: requires SAMPLER has operator()()
//    uint sampleAndRunAsync(AGENT1 &firstAgent, AGENT2 &secondAgent, SAMPLER &&episode, size_t nEpisodes, CALLBACKS... callbacks) {
//        uint totalMessages = 0;
//        while (nEpisodes > 0) {
//            callOnEpisodeStart(callbacks);
//            totalMessages += runAsyncEpisode(firstAgent, secondAgent, callbacks);
//            --nEpisodes;
//        }
//        return totalMessages;
//    }


//    /** Execute a single episode between two agents where each agent sends messages at the same time
//     * (e.g. rock-paper-scissors)
//     * @param agent0 first mover agent
//     * @param agent1 second mover agent
//     * @param verbose it true, prints the passed messages to stdout
//     * @return total number of messages sent in the episode
//     */
//    template<class AGENT0, class AGENT1>
//    int runSynchronousEpisode(AGENT0 &agent0, AGENT1 &agent1, bool verbose = false) {
//        if (verbose) {
//            std::cout << "------- Starting episode -------" << std::endl;
//            std::cout << agent0.body << agent1.body << std::endl;
//        }
//        std::optional<typename AGENT0::out_message_type> message0 = agent0.startEpisode();
//        std::optional<typename AGENT1::out_message_type> message1 = agent1.startEpisode();
//        int nMessages = 0;
//        while (message0.has_value() && message1.has_value()) {
//            if (verbose) std::cout << message0 << " <--> " << message1 << std::endl;
//            auto tmpMessage = agent1.handleMessage(message0.value());
//            message0 = agent0.handleMessage(message1.value());
//            message1 = tmpMessage;
//            ++nMessages;
//        }
//        agent0.endEpisode();
//        agent1.endEpisode();
//        if (verbose) {
//            std::cout << agent0.body << agent1.body << std::endl;
//            std::cout << "------- Ending episode -------" << std::endl;
//        }
//        return nMessages;
//    }
//
//    template<class AGENT0, class AGENT1>
//    int runSynchronousEpisode(std::pair<AGENT0,AGENT1> &agentPair, bool verbose = false) {
//        return runSynchronousEpisode(agentPair.first, agentPair.second, verbose);
//    }

//    /** An episode where the initial states of the agents are drawn at random from
//      * separate distributions.
//      * @tparam AGENT1 The first mover in the episode
//      * @tparam AGENT2 The second mover in the episode
//      */
//    template<class AGENT1, class AGENT2 = AGENT1>
//    class SimpleEpisode {
//    public:
//        std::function<AGENT1()> firstMoverPriorDistribution;
//        std::function<AGENT2()> secondMoverPriorDistribution;
//
//        SimpleEpisode(
//                std::function<AGENT1()> firstMoverPriorDistribution,
//                std::function<AGENT2()> secondMoverPriorDistribution) :
//        firstMoverPriorDistribution(std::move(firstMoverPriorDistribution)),
//        secondMoverPriorDistribution(std::move(secondMoverPriorDistribution)) {}
//
//        /** Sample the prior player states and run n episodes
//         * @param verbose if true, debug info will be printed to stdout
//         * @return total number of messages sent
//         */
//        inline uint runAsync(uint nEpisodes, bool verbose = false) {
//            uint totalMessages = 0;
//            while (nEpisodes > 0) {
//                totalMessages += runAsyncEpisode(firstMoverPriorDistribution(), secondMoverPriorDistribution(), verbose);
//                --nEpisodes;
//            }
//            return totalMessages;
//        }
//
//        /** Sample the prior player states and run n episodes
//        * @param verbose if true, debug info will be printed to stdout
//        * @return total number of messages sent
//        */
//        inline uint runSync(uint nEpisodes, bool verbose = false) {
//            uint totalMessages = 0;
//            while (nEpisodes > 0) {
//                totalMessages += runSynchronousEpisode(firstMoverPriorDistribution(), secondMoverPriorDistribution(), verbose);
//                --nEpisodes;
//            }
//            return totalMessages;
//        }
//    };


//    /** An episode where the initial states of the agents are drawn at random from
//    * a joint distribution.
//    */
//    template<class AGENTPAIRSAMPLER>
//    class BinaryEpisode {
//    public:
//
//        AGENTPAIRSAMPLER agentPairSampler;
//        typedef decltype(agentPairSampler()) pair_type;
//
//        BinaryEpisode(AGENTPAIRSAMPLER agentpairsampler) : agentPairSampler(std::move(agentpairsampler)) { }
//
//        template<class FIRSTMOVERSAMPLER, class SECONDMOVERSAMPLER>
//        BinaryEpisode(
//                FIRSTMOVERSAMPLER  firstMoverPriorDistribution,
//                SECONDMOVERSAMPLER secondMoverPriorDistribution) -> BinaryEpisode<std::function<std::pair<decltype(firstMoverPriorDistribution()),decltype(secondMoverPriorDistribution())>()>> :
//                agentPairSampler([firstMoverPriorDistribution, secondMoverPriorDistribution]() mutable {
//                    return std::pair(firstMoverPriorDistribution(), secondMoverPriorDistribution());
//                }) {}
//
//        /** Sample the prior player states and run n episodes
//         * @param verbose if true, debug info will be printed to stdout
//         * @return total number of messages sent
//         */
//        inline uint runAsync(uint nEpisodes, bool verbose = false) {
//            uint totalMessages = 0;
//            while (nEpisodes > 0) {
//                totalMessages += runAsyncEpisode(agentPairSampler(), verbose);
//                --nEpisodes;
//            }
//            return totalMessages;
//        }
//
//        /** Sample the prior player states and run n episodes
//        * @param verbose if true, debug info will be printed to stdout
//        * @return total number of messages sent
//        */
//        inline uint runSync(uint nEpisodes, bool verbose = false) {
//            uint totalMessages = 0;
//            while (nEpisodes > 0) {
//                totalMessages += runSynchronousEpisode(agentPairSampler(), verbose);
//                --nEpisodes;
//            }
//            return totalMessages;
//        }
//    };


}

#endif //MULTIAGENTGOVERNMENT_SIMPLEEPISODE_H
