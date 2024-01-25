// An Agent is a mind/body pair. The Agent generates certain events in response to
// messages, and passes on externally generated events.
//
//
// Message passing view (deprecated):
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
//#include "Body.h"
//#include "Mind.h"
#include "CallbackUtils.h"
#include "../DeselbyStd/OptionalDouble.h"
#include "Concepts.h"
#include "episodes/SimpleEpisode.h"

namespace abm::events {

    struct IsEndEpisodeMessage {
        IsEndEpisodeMessage(bool isEndEpisode) : isEndEpisode(isEndEpisode) {}
        bool isEndEpisode;
    };

    /** StartEpisode event for an agent that doesn't communicate with any other agent */
    template<class FIRSTMOVERBODY, class SECONDMOVERBODY>
    struct AgentStartEpisode {
        FIRSTMOVERBODY &    firstMoverBody;
        SECONDMOVERBODY &   secondMoverBody;
        bool isFirstMover;

        explicit AgentStartEpisode(FIRSTMOVERBODY &firstmoverbody, SECONDMOVERBODY &secondmoverbody, bool isfirstmover):
        firstMoverBody(firstmoverbody),
        secondMoverBody(secondmoverbody),
        isFirstMover(isfirstmover) {}

        friend std::ostream &operator <<(std::ostream &out, const AgentStartEpisode<FIRSTMOVERBODY,SECONDMOVERBODY> &event) {
            out << "Starting episode...";
            return out;
        }
    };

    template<class BODY>
    struct AgentEndEpisode {
        BODY &body;

        explicit AgentEndEpisode(BODY &body): body(body) {}

        friend std::ostream &operator <<(std::ostream &out, const AgentEndEpisode<BODY> &event) {
            out << "Ending episode...";
            return out;
        }
    };

    struct Reward {
        double reward;

        Reward(const double &reward): reward(reward) {}
//        Reward(const std::nullopt_t &): reward() {}
//        Reward() : reward() {}

        friend std::ostream &operator <<(std::ostream &out, const Reward &event) {
            out << "reward: " << event.reward;
            return out;
        }
    };


    /** return type for a body's handleAct(.) method */
    template<class MESSAGE>
    struct MessageReward : Reward {
        MESSAGE message;

        MessageReward(MESSAGE message, const double &reward): Reward(reward), message(std::move(message)) {}
//        MessageReward(MessageReward<MESSAGE> &&)=default;

        friend std::ostream &operator <<(std::ostream &out, const MessageReward<MESSAGE> &event) {
            out << "(message: " << event.message << ", reward: " << event.reward << ")";
            return out;
        }
    };


    template<class BODY>
    struct PreActBodyState {
        BODY &body;
        PreActBodyState(BODY &body): body(body) {}

        friend std::ostream &operator <<(std::ostream &out, const PreActBodyState<BODY> &event) {
            out << "Preparing to act...";
            return out;
        }
    };


    template<class BODY>
    struct PostActBodyState {
        BODY &body;
        PostActBodyState(BODY &body): body(body) {}

        friend std::ostream &operator <<(std::ostream &out, const PostActBodyState<BODY> &event) {
            out << "Acted...";
            return out;
        }
    };

    struct IncomingMessageResponse : Reward {
        bool isEndEpisode;

        friend std::ostream &operator <<(std::ostream &out, const IncomingMessageResponse &event) {
            out << "Reward: " << event.reward << " isEndEpisode: " << event.isEndEpisode;
            return out;
        }
    };

    template<class MESSAGE>
    struct IncomingMessage : IncomingMessageResponse {
        MESSAGE message;
    };


    template<class MESSAGE>
    struct OutgoingMessage : MessageReward<MESSAGE> {
        OutgoingMessage(MESSAGE &&message, const double &reward): MessageReward<MESSAGE>(std::move(message),reward) {};
    };


    /** return type for a body's handleAct(.) method
     * when the body never communicates with any other agent.
     * In this case, the message is a bool, indicating whether the episode has ended
     */
     // Use OutgoingMessage<EmptyMessage>
//    struct MonadicMessageReward : OutgoingMessage<bool> {
//        MonadicMessageReward(bool isEndEpisode, const double &reward): OutgoingMessage<bool>(std::move(isEndEpisode), reward) {}
//
//        bool isEndEpisode() { return message; }
//    };


    /** Mind's act and Body's reward and outgoing message in response to the act */
    template<class ACTION, class MESSAGE>
    struct AgentStep : OutgoingMessage<MESSAGE> {
        ACTION act;

        AgentStep(ACTION &&act, OutgoingMessage<MESSAGE> &&messageReward):
                OutgoingMessage<MESSAGE>(std::move(messageReward)),
                act(std::move(act)) { }

        friend std::ostream &operator <<(std::ostream &out, const AgentStep<ACTION,MESSAGE> &event) {
            out << "Agent performed act " << event.act << " and body returned " << static_cast<OutgoingMessage<MESSAGE> &>(event);
            return out;
        }
    };


}


namespace abm::callbacks {
    /** Use this class to record the exponentially weighted mean
     *  reward, which we define as
     *
     * E_n[R] = a_n.R_n + a_n.r.R_{n-1} + a_n.r^2.R_{n-2} + ... + a_n.r^{n-1}.R_1
     *
     * where
     * R_t is the reward at time t,
     * r is a supplied constant decay rate
     * a_n = (1-r)/(1-r^n)
     *
     * This gives the recurrence relation
     * E_n[R] = a_n.R_n + (a_n.r/a_{n-1}).E_{n-1}[R]
     * and, by expansion
     * a_n.r/a_{n-1} + a_n = (r-r^n)/(1-r^n) + (1-r)/(1-r^n) = 1
     * so
     * a_n.r/a_{n-1} = 1-a_n
     * and
     * E_n[R] = a_n.R_n + (1-a_n).E_{n-1}[R]
     *
     * The mean is just the weighted mean of the intercepted Reward messages.
     */
    class MeanReward {
    public:
        const double decayRate;
        size_t nSamples     = 0;
        double meanReward   = 0.0;

        MeanReward(double decayRate) : decayRate(decayRate) {}

        void on(const events::Reward &event) {
            double a_n = (1.0 - decayRate) / (1.0 - std::pow(decayRate, ++nSamples));
            meanReward *= 1.0 - a_n;
            meanReward += a_n * event.reward;
        }
    };

    /** Log the mean reward per episode */
    class MeanRewardPerEpisode {
    public:
        size_t nEpisodes     = 0;
        double totalReward   = 0.0;

        template<class BODY1, class BODY2>
        void on(const events::AgentStartEpisode<BODY1,BODY2> & /* event */) {
            ++nEpisodes;
        }

        void on(const events::Reward &event) {
            totalReward += event.reward;
        }

        double mean() { return totalReward / nEpisodes; }

        void reset() { nEpisodes = 0; totalReward = 0.0; }
    };

    /** Log the mean reward per episode */
    class RewardPerEpisode {
    public:
        double rewardThisEpisode   = 0.0;

        template<class BODY1, class BODY2>
        void on(const events::AgentStartEpisode<BODY1,BODY2> & /* event */) {
            rewardThisEpisode = 0.0;
        }

        void on(const events::Reward &event) {
            rewardThisEpisode += event.reward;
        }
    };

}


namespace abm {
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

        // Given BODY and MIND, action and (outgoing) message types are defined.
        // These typedefs can be used to refer to them, given BODY and MIND
        typedef decltype(mind.act(body)) action_type;
        typedef decltype(body.handleAct(mind.act(body)).message) message_type;

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


        /** Pass on events to body and mind */
        template<class EVENT> requires HasCallback<BODY, EVENT> || HasCallback<MIND, EVENT>
        inline void on(const EVENT & event) {
            callback(event, body);
            callback(event, mind);
        }

        template<class AGENT1,class AGENT2>
        void on(const events::StartEpisode<AGENT1,AGENT2> & event) {
            callback(event, body);
            callback(event, mind);
            bool isFirstMover = (static_cast<void *>(this) == static_cast<void *>(&event.agent1));
            events::AgentStartEpisode startEpisodeEvent(event.agent1.body, event.agent2.body, isFirstMover);
            callback(startEpisodeEvent, body);
            callback(startEpisodeEvent, mind);
        }

        template<class AGENT1,class AGENT2>
        void on(const events::EndEpisode<AGENT1,AGENT2> & event) {
            callback(event, body);
            callback(event, mind);
            callback(events::AgentEndEpisode(body), mind, body);
        }

        /** This method is called after initiation.
         * If the episode is turns-based it is called on the first mover only in order to start the episode.
         * If the episode is synchronous (e.g. rock-paper-scisors) it is called on both agents.
         * @return message that begins the episode
         */
        message_type startEpisode() requires BodyMindPair<BODY,MIND> {
            return getNextActEvent().message;
        }

        /** If this body/mind pair is a monad (i.e. body never sends outgoing messages) then an episode
         * consists of act/reward between mind and body until body returns a not-a-number as reward */
         template<class... CALLBACKS>
        void runEpisode(CALLBACKS &&... callbacks) requires BodyMindMonad<BODY,MIND> {
            events::AgentStartEpisode startEpisodeEvent(body,body,true);
            callback(startEpisodeEvent, body);
            callback(startEpisodeEvent, mind);
            callback(startEpisodeEvent, std::forward<CALLBACKS>(callbacks)...);
            events::AgentStep<action_type,message_type> response = getNextActEvent(std::forward<CALLBACKS>(callbacks)...);
            while(!response.message.isEndEpisode) {
                response = getNextActEvent(std::forward<CALLBACKS>(callbacks)...);
            }
            callback(events::AgentEndEpisode(body), mind, body, std::forward<CALLBACKS>(callbacks)...);
        }


        /** This method is called when the agent receives a message from another agent
         * @param incomingMessage message sent by the other agent
         * @return response to incoming message. If unset, the agent has (perhaps unilaterally)
         *   ended the episode.
         *   TODO: if we separate the handling of a message from the creation of a response, then
         *         we can run a step without having to template over message type. This takes us
         *         back to the Channel concept, but now a channel can handle any set of message types
         *         and a Channel can respond to (or ignore) any set of event types (generated by the
         *         chennel itself or sent by the agent). This solves the problem of multi-channel communication.
         */
        template<class INMESSAGE> requires BodyMindPair<BODY,MIND>
        deselby::ensure_optional_t<message_type> handleMessage(INMESSAGE incomingMessage) {
            events::IncomingMessageResponse inMessageResponse = body.handleMessage(incomingMessage);
            events::IncomingMessage<message_type> inMessageEvent{std::move(inMessageResponse), std::move(incomingMessage)};
            callback(inMessageEvent,mind);
            if(inMessageEvent.isEndEpisode) return std::nullopt;
            return getNextActEvent().message;
        }


        friend std::ostream &operator <<(std::ostream &out, const Agent<BODY,MIND> &agent) {
            deselby::constexpr_if<deselby::HasInsertStreamOperator<MIND>>([&out](auto &mind) { out << mind << std::endl; }, agent.mind);
            deselby::constexpr_if<deselby::HasInsertStreamOperator<BODY>>([&out](auto &body) { out << body << std::endl; }, agent.body);
            return out;
        }
    protected:
        template<class... EXTRACALLBACKS>
        events::AgentStep<action_type,message_type> getNextActEvent(EXTRACALLBACKS &&... extraCallbacks) {
            callback(events::PreActBodyState(body), mind, std::forward<EXTRACALLBACKS>(extraCallbacks)...);
            action_type act = mind.act(body);
            events::OutgoingMessage<message_type> outMessageEvent = body.handleAct(act);
            events::AgentStep<action_type, message_type> actEvent{std::move(act), std::move(outMessageEvent)};
            callback(actEvent, mind, std::forward<EXTRACALLBACKS>(extraCallbacks)...);
            callback(events::PostActBodyState(body), mind, std::forward<EXTRACALLBACKS>(extraCallbacks)...);
            return actEvent;
        }
    };
}


#endif //MULTIAGENTGOVERNMENT_AGENT_H
