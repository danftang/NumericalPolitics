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
    template<Body BODY, Mind MIND> requires Compatible<BODY,MIND>
    class Agent {
    public:
        typedef BODY body_type;
        typedef MIND mind_type;
        typedef BODY::in_message_type in_message_type;
        typedef BODY::action_type action_type;
        typedef Traits<BODY>::out_message_type out_message_type;

        BODY body;
        MIND mind;

    private:
        double reward = 0.0;
    public:
//        /** the sum of actual received rewards, exponentially discounted into the past
//         * (for monitoring purposes only, doesn't affect behaviour) */
//        double meanReward = 0.0;
//    private:
//        double meanRewardDecay; // rate of exponential decay of the weight of rewards into the past when calculating the mean
//        double halfStepRewardDecay; // exponential decay of the mean weight for half a step (at beginning or end of an episode)
//    public:

        /**
         *
         * @param body
         * @param mind
         * @param meanRewardDecayRate  The discount factor per message/response interaction for "meanReward",
         */
        Agent(BODY body, MIND mind) : body(std::move(body)), mind(std::move(mind)) { }

        // ------ Agent Interface ------


//        void reset(BODY &&bodyState) { body = std::forward<BODY>(bodyState); }

        /** An episode should begin with a call to initEpisode. Here is where all the initial setup of the episode
         * happens.
         *
         * @param initBodyState     The initial body state of the agent at the start of the episode (this should encode
         *                          any private information that the agent has)
         * @param sharedinformation Any prior information that is publicly shared at the beginning of the episode,
         *                          i.e. agents know is shared, and know that the other agent knows it is shared etc...
         * TODO: could also include any information that this agent is given that isn't known to be public.
         *   Also, can we make this generic without templating the classes? [in full generality, we have an
         *   infinite sequence of PDFs over both my state and yours representing n'th level beliefs about
         *   a state. What we're saying with this interface is that there is only public information and private
         *   information (i.e. information that is in-fact private and it is public information that it is private)]
         *
         *   TODO: this could be inplemented as an agent that is listening for open channel requests, which upon
         *     request, returns an open channel to which the opening agent can send the first message.
         */
        template<class PUBLICINFORMATION>
        inline void initEpisode(BODY initBodyState, PUBLICINFORMATION sharedinformation) {
            body = std::move(initBodyState);
            callInitEpisodeHook(mind, body, sharedinformation);
        }

        inline void initEpisode(BODY initBodyState) {
            body = std::move(initBodyState);
            callInitEpisodeHook(mind, body);
        }

        /** This method is called after initiation.
         * If the episode is turns-based it is called on the first mover only in order to start the episode.
         * If the episode is synchronous (e.g. rock-paper-scisors) it is called on both agents.
         * @return message that begins the episode
         */
        out_message_type startEpisode() {
            action_type lastAct = mind.act(body, body.legalActs(), 0);
            out_message_type initialMessage = body.actToMessage(lastAct);
            callOutgoingMessageHook(mind, initialMessage);
            return initialMessage;
        }

//        out_message_type startEpisode() {
//            action_type lastAct = mind.act(body, body.legalActs(), 0);
//            out_message_type initialMessage = body.actToMessage(lastAct);
//            callOutgoingMessageHook(mind, initialMessage);
//            if(body.isEndOfEpisode()) {
//                const double residualReward = body.endEpisode();
////                const double residualFlux = 2.0*residualReward;
////                meanReward = meanReward * halfStepRewardDecay + (1.0 - halfStepRewardDecay) * residualFlux;
//                callHalfStepObservationHook(mind,body);
//                mind.endEpisode(residualReward);
//            }
//            return initialMessage;
//        }



        /** This method is called when the agent receives a message from another agent
         * @param incomingMessage message sent by the other agent
         * @return response to incoming message. If unset, the agent has (perhaps unilaterally)
         *   ended the episode.
         */
        std::optional<out_message_type> handleMessage(in_message_type incomingMessage) {
            callHalfStepObservationHook(mind,body);
            reward = body.messageToReward(incomingMessage);
            callIncomingMessageHook(mind,incomingMessage);
            auto legalActs = body.legalActs();
            if(!legalActs.any()) return {}; // no legal acts: end episode
            action_type lastAct = mind.act(body, legalActs, reward);
//            std::cout << "got reward " << reward << " for " << this << std::endl;
            reward = 0.0;
            out_message_type response = body.actToMessage(lastAct);
            callOutgoingMessageHook(mind,response);
            return response;
        }

//        std::optional<out_message_type> handleMessage(in_message_type incomingMessage) {
//            callHalfStepObservationHook(mind,body);
//            double reward = body.messageToReward(incomingMessage);
////            std::cout << "reward = " << reward << std::endl;
//            callIncomingMessageHook(mind,incomingMessage);
////            meanReward = meanReward*meanRewardDecay + (1.0-meanRewardDecay)*reward;
//            if(body.isEndOfEpisode()) {
//                double residualReward = body.endEpisode();
//                assert(residualReward == 0.0);
//                mind.endEpisode(reward);
//                return {};
//            }
//            action_type lastAct = mind.act(body, body.legalActs(), reward);
//            out_message_type response = body.actToMessage(lastAct);
//            callOutgoingMessageHook(mind,response);
//            if (body.isEndOfEpisode()) {
//                const double residualReward = body.endEpisode();
////                const double residualFlux = 2.0*residualReward;
////                meanReward = meanReward * halfStepRewardDecay + (1.0 - halfStepRewardDecay) * residualFlux;
////                std::cout << "Residual reward = " << residualReward << std::endl;
//                mind.endEpisode(residualReward);
//            }
//            return response;
//        }



        /** This method is called when any agent ends the episode.
         * It is called on both agents, irrespective of who ended the episode.
         * TODO: can the environment end an episode, or should this be modelled as an
         *   intermediary agent with two open communication channels [yes, better this way]?
         */
        void endEpisode() {
            double residualReward = body.endEpisode();
            mind.endEpisode(reward + residualReward);
            // std::cout << "Got final reward " << reward << " + " << residualReward << " = " << reward + residualReward << " for " << this << std::endl;
        }

//        void setMeanRewardDecay(double exponentialDecayRatePerTransaction) {
//            meanRewardDecay = exponentialDecayRatePerTransaction;
//            halfStepRewardDecay = sqrt(exponentialDecayRatePerTransaction);
//        }
    };



}


#endif //MULTIAGENTGOVERNMENT_AGENT_H
