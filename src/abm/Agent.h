// An agent is a node in a computational graph which implements handleMessage(...) methods
// for some number of message types. The handling of a message may (or may not)
// cause other messages to be sent.
//   * If handle(.) can deliver the outgoing messages by tail recursion, it should
//     do so and return void.
//   * If handla(.) needs to return only one message to the sender, then it should
//     do so in the return value
//   * Otherwise, handle(.) should return a tuple of AddressedMessages, and
//     a delivery agent should deliver these.
//
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
#include "Events.h"

namespace abm {

    // TODO: A body handles
    //    acts from the mind
    //    messages from the environment
    //  and emits
    //    rewards to the mind
    //    messages to the environment
    //  A mind handles
    //    rewards from the body
    //  and emits
    //    acts to the body.
    //  An Agent is just a body with a mind
    //
    //
    template<class BODY, class MIND>
    concept BodyMindPair = requires(BODY body, MIND mind) {
        body.actToMessageAndReward(mind.act(body));
    };

    namespace events {
        template<class MESSAGE>
        struct IncomingMessage {
            IncomingMessage(MESSAGE &&message) : message(std::move<MESSAGE>(message)) {};
            IncomingMessage(const MESSAGE &message): message(message) {};
            MESSAGE message;
        };

        template<class MESSAGE>
        struct OutgoingMessage {
            OutgoingMessage(MESSAGE &&message) : message(std::move<MESSAGE>(message)) {};
            OutgoingMessage(const MESSAGE &message): message(message) {};
            MESSAGE &message;
        };

        struct Reward { double value; };

    };

    template<Body BODY, Mind<BODY> MIND>
    class BodyMindTraits {
    public:
        typedef decltype(std::declval<MIND>().act(std::declval<BODY>())) action_type;
        typedef decltype(std::declval<BODY>().actToMessageAndReward(std::declval<action_type>()).first) out_message_type;
        typedef decltype(std::declval<BODY>().actToMessageAndReward(std::declval<action_type>()).second) reward_type;
        typedef decltype(std::declval<BODY>().legalActs()) action_mask;
    };

//    template<class T>
//    concept AgentChannel = requires(T agentChannel, typename T::in_message_type incomingMessage) {
//        typename T::in_message_type;
//        { agentChannel.startEpisode() };
//        { agentChannel.handleMEssage(incomingMessage) };
//    };


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
    template<class BODY, class MIND> requires BodyMindPair<BODY,MIND>
    class Agent {
    public:
        BODY body;
        MIND mind;

        typedef BODY body_type;
        typedef MIND mind_type;
        typedef typename decltype(mind.optionalAct(body))::value_type action_type;           // N.B. these are not intrinsic to mind or body
        typedef typename decltype(body.optionalMessage(declval<action_type>()))::value_type out_message_type; // but only defined given the pair


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
//        template<class BODYARG, class MINDARG>
//        inline void init(BODYARG &&bodyArg, MINDARG &&mindArg) { // TODO: init is better in the episode
//            initCallback(body,std::forward<BODYARG>(bodyArg));
//            initCallback(mind,std::forward<MINDARG>(mindArg)); // mind had better have this otherwise sharedInformation is not used
////            callInitEpisodeHook(mind, body, sharedinformation);
//        }


//        inline void initEpisode(BODY initBodyState) {
//            body = std::move(initBodyState);
//            callInitEpisodeHook(mind, body);
//        }

        /** This method is called after initiation.
         * If the episode is turns-based it is called on the first mover only in order to start the episode.
         * If the episode is synchronous (e.g. rock-paper-scisors) it is called on both agents.
         * @return message that begins the episode
         */
        std::optional<out_message_type> startEpisode() {
            std::optional<out_message_type> result;
            std::optional<action_type> act = mind.optionalAct(body);
            if(!act.has_value()) return result;
            std::optional<out_message_type> initialMessage = body.optionalMessage(*act);
            if(!initialMessage.has_value()) return result;
            events::callback(events::OutgoingMessage(*initialMessage),mind);
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
         template<class INMESSAGE>
        std::optional<out_message_type> handleMessage(INMESSAGE incomingMessage) {
            std::optional<out_message_type> response;
            events::callback(events::IncomingMessage(incomingMessage),mind);
            reward = body.handle(incomingMessage);
            events::callback(events::Reward(reward),mind);
            std::optional lastAct = mind.act(body);
            if(!lastAct.has_value()) return response;
//            reward = 0.0;
            response = body.optionalMessage(*lastAct);
            if(response.has_value()) events::callback(events::OutgoingMessage(*response));
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
         * TODO: should this be done in handleMessage with an unset optional?...or with an EndEpisode() message?
         *   or even as optional callback? [probably this as it may be null of no learning]
         */
        void endEpisode() {
            double residualReward = body.endEpisode();
            mind.endEpisode(reward + residualReward);
            // std::cout << "Got final reward " << reward << " + " << residualReward << " = " << reward + residualReward << " for " << this << std::endl;
        }

        friend std::ostream &operator <<(std::ostream &out, const Agent<BODY,MIND> &agent) {
            deselby::constexpr_if<deselby::IsStreamable<MIND>>([&out](auto &mind) { out << mind << std::endl; }, agent.mind);
            deselby::constexpr_if<deselby::IsStreamable<BODY>>([&out](auto &body) { out << body << std::endl; }, agent.body);
            return out;
        }

//        void setMeanRewardDecay(double exponentialDecayRatePerTransaction) {
//            meanRewardDecay = exponentialDecayRatePerTransaction;
//            halfStepRewardDecay = sqrt(exponentialDecayRatePerTransaction);
//        }
    };



}


#endif //MULTIAGENTGOVERNMENT_AGENT_H
