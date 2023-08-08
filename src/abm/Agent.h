//
// Created by daniel on 02/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_AGENT_H
#define MULTIAGENTGOVERNMENT_AGENT_H

#include <concepts>
#include <bitset>
#include <algorithm>
#include <cassert>
#include <utility>
#include "Body.h"
#include "Mind.h"

namespace abm {

    template<class BODY, class MIND>
    concept Compatible = Body<BODY> && Mind<MIND> && requires(
            MIND mind,
            BODY body,
            BODY::in_message_type message,
            MIND::reward_type reward,
            MIND::action_mask mask
            ) {
        { mind.act(body, mask, reward) }    -> std::convertible_to<typename BODY::action_type>;
        { body.messageToReward(message) }   -> std::convertible_to<typename MIND::reward_type>;
        { body.legalActs() }                -> std::convertible_to<typename MIND::action_mask>;
    };

    /**
     *  An agent consists of a body and a mind.
     *  The body stores any local state, handles incoming
     *  messages from other agents, defines rewards for the mind, defines which acts are physically possible
     *  at any time and translates the mind's "actions" into messages to send to other agents.
     *  The mind can "observe" the state of the body (and nothing else) and, based upon that, must
     *  decide how to act.
     * @tparam BODY
     * @tparam MIND
     */
    template<Body BODY, Mind MIND> requires(Compatible<BODY,MIND>)
    class Agent {
    public:
        typedef BODY body_type;
        typedef MIND mind_type;
        typedef BODY::in_message_type in_message_type;
        typedef BODY::action_type action_type;
        typedef decltype(std::declval<BODY>().actToMessage(std::declval<action_type>())) out_message_type;

        BODY body;
        MIND mind;

        /** the sum of actual received rewards, exponentially discounted into the past
         * (for monitoring purposes only, doesn't affect behaviour) */
        double meanReward = 0.0;

    private:
        double meanRewardDecay; // rate of exponential decay of the weight of rewards into the past when calculating the mean
        double halfStepRewardDecay; // exponential decay of the mean weight for half a step (at beginning or end of an episode)

    public:

        /**
         *
         * @param body
         * @param mind
         * @param meanRewardDecayRate  The discount factor per message/response interaction for "meanReward",
         */
        Agent(BODY body, MIND mind, double meanRewardDecayRate = 0.99) : body(std::move(body)), mind(std::move(mind)) {
            setMeanRewardDecay(meanRewardDecayRate);
        }

        // ------ Agent Interface ------


//        void reset(BODY &&bodyState) { body = std::forward<BODY>(bodyState); }

        /**
         * Call this to nudge the agent to be the first mover in an episodic interaction
         * @return message that begins the episode
         */
        out_message_type startEpisode() {
            action_type lastAct = mind.act(body, body.legalActs(), 0);
            out_message_type initialMessage = body.actToMessage(lastAct);
            callOutgoingMessageHook(mind, initialMessage);
            callHalfStepObservationHook(mind, body);
            if(body.isEndOfEpisode()) {
                const double residualReward = body.endEpisode();
                const double residualFlux = 2.0*residualReward;
                meanReward = meanReward * halfStepRewardDecay + (1.0 - halfStepRewardDecay) * residualFlux;
                mind.endEpisode(residualReward);
            }
            return initialMessage;
        }


        /**
         * Call this when the agent receives a message from another agent
         * @param incomingMessage message originating from another agent
         * @return response to incoming message
         */
        std::optional<out_message_type> handleMessage(in_message_type incomingMessage) {
            double reward = body.messageToReward(incomingMessage);
            callIncomingMessageHook(mind,incomingMessage);
            meanReward = meanReward*meanRewardDecay + (1.0-meanRewardDecay)*reward;
            if(body.isEndOfEpisode()) {
                double residualReward = body.endEpisode();
                assert(residualReward == 0.0);
                mind.endEpisode(reward);
                return {};
            }
            action_type lastAct = mind.act(body, body.legalActs(), 0);
            out_message_type response = body.actToMessage(lastAct);
            callOutgoingMessageHook(mind,response);
            callHalfStepObservationHook(mind,body);
            if (body.isEndOfEpisode()) {
                const double residualReward = body.endEpisode();
                const double residualFlux = 2.0*residualReward;
                meanReward = meanReward * halfStepRewardDecay + (1.0 - halfStepRewardDecay) * residualFlux;
                mind.endEpisode(residualReward);
            }
            return response;
        }

        /** We define the exponentially weighted mean flux of reward as
         * m = int_{-inf}^{0} w(t)f(t) dt
         * where w(t) = ke^{kt} is a weighting that sums to 1 and decays into the past
         * and f(t) is the historical flux of reward per unit time.
         *
         * If we let d = e^-k (and so int_{-1}^0 ke^{kt} = 1-d) and assume a total reward of r
         * is received with a constant flux over one unit of time then
         * m' = md + (1-d)r
         *
         * For a half step we shift the weights by 0.5 so if we let
         * d' = e^-0.5*k = sqrt(d)
         * so if we get a reward r' in a half step then
         * m' = md' + (1-d')2r'
         * where the flux is twice the total reward as it is sustained over half a unit of time
         *
         * N.B. to see this is correct, consider a single unit total reward at time 0 sustained over
         * a duration, DT. In this case
         * m = (1-d^DT)(1/DT) = (1 - e^{log(d)DT})/DT
         * as DT tends to 0, the flux tends to a delta function and m tends to -log(d) which is k above,
         * (i.e. w(0)) which is what we would expect.
         *
         * @param exponentialDecayRatePerTransaction
         */
        void setMeanRewardDecay(double exponentialDecayRatePerTransaction) {
            meanRewardDecay = exponentialDecayRatePerTransaction;
            halfStepRewardDecay = sqrt(exponentialDecayRatePerTransaction);
        }
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
    template<Body BODY0, Mind MIND0, Body BODY1, Mind MIND1>
    requires(std::is_convertible_v<typename BODY0::message_type, typename BODY1::message_type> &&
             std::is_convertible_v<typename BODY1::message_type, typename BODY0::message_type>)
    int episode(Agent<BODY0, MIND0> &agent0, Agent<BODY1, MIND1> &agent1, bool verbose = false) {
        if(verbose) {
            std::cout << "------- Starting episode -------" << std::endl;
            std::cout  << agent0.body << agent1.body << std::endl;
        }
        std::optional<typename BODY0::message_type> message0 = agent0.startEpisode();
        std::optional<typename BODY1::message_type> message1;
        int nMessages = 0;
        while(message0.has_value()) {
            if (verbose) std::cout << message0 << std::endl;
            message1 = agent1.handleMessage(message0.value());
            ++nMessages;
            if(message1.has_value()) {
                if (verbose) std::cout << message1 << std::endl;
                message0 = agent0.handleMessage(message1.value());
                ++nMessages;
            } else {
                message0.reset();
            }
        }
        if(verbose) {
            std::cout << agent0.body << agent1.body << std::endl;
            std::cout << "------- Ending episode -------" << std::endl;
        }
        return nMessages;

    }
}


#endif //MULTIAGENTGOVERNMENT_AGENT_H
