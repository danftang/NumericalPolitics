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
#include "CallbackUtils.h"
#include "../DeselbyStd/OptionalDouble.h"

namespace abm::events {

    template<class BODY>
    struct AgentStartEpisode {
        BODY &body;

        explicit AgentStartEpisode(BODY &body): body(body) {}

        friend std::ostream &operator <<(std::ostream &out, const AgentStartEpisode<BODY> &event) {
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


//    template<class T>
//    struct Action {
//        T act;
//
//        explicit Action(const T &act): act(act) {}
//        explicit Action(T &&act): act(std::move(act)) {}
//    };


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

    /** return type for a body's handleAct(.) method
     * when the body never communicates with any other agent.
     * In this case, the message is a bool, indicating whether the episode has ended
     */
    struct MonadicMessageReward : MessageReward<bool> {
        MonadicMessageReward(bool isEndEpisode, const double &reward): MessageReward<bool>(isEndEpisode, reward) {}

        bool isEndEpisode() { return message; }
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

    template<class MESSAGE>
    struct IncomingMessage : MessageReward<MESSAGE> {
        IncomingMessage(MESSAGE &&message, double &&reward): MessageReward<MESSAGE>(std::move(message),std::move(reward)) {};
    };

    /** Mind acts and body responds */
    template<class ACTION, class RESPONSE>
    struct Act : RESPONSE {
        ACTION act;

        Act(ACTION &&act, RESPONSE &&messageReward):
                RESPONSE(std::move(messageReward)),
                act(std::move(act)) { }

        const RESPONSE &response() const { return *this; }

        friend std::ostream &operator <<(std::ostream &out, const Act<ACTION,RESPONSE> &event) {
            out << "Agent performed act " << event.act << " and body returned " << event.response();
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

        template<class BODY>
        void on(const events::AgentStartEpisode<BODY> & /* event */) {
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

        template<class BODY>
        void on(const events::AgentStartEpisode<BODY> & /* event */) {
            rewardThisEpisode = 0.0;
        }

        void on(const events::Reward &event) {
            rewardThisEpisode += event.reward;
        }
    };

}

namespace abm {
    template<class BODY, class MIND>
    concept BodyMindPair = requires(BODY body, MIND mind) {
        { body.handleAct(mind.act(body)) } -> deselby::IsSpecializationOf<events::MessageReward>;
        // and handleMessage(.) on some type of incoming message, depending on which agent it is paired with
    };


    /** A body/mind monad is one where the body doesn't send any messages out or handle any
     * incoming messages. So the communication is entirely act/reward between mind and body. */
    template<class BODY, class MIND>
    concept BodyMindMonad = requires(BODY body, MIND mind) {
        { body.handleAct(mind.act(body)) } -> std::same_as<events::MonadicMessageReward>;
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
    template<class BODY, class MIND> class Agent {
    public:
        BODY body;
        MIND mind;

        typedef BODY body_type;
        typedef MIND mind_type;

        // This design implies single action and message types
        typedef decltype(mind.act(body)) action_type;
//        typedef decltype(body.handleAct(std::declval<action_type>()).message) message_type;
//        typedef events::OutgoingMessage<BODY,action_type, message_type> outgoing_message_type;

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
        inline void on(const EVENT &event) {
            callback(event, body);
            callback(event, mind);
        }

        template<class AGENT1,class AGENT2>
        void on(const events::EndEpisode<AGENT1,AGENT2> & /* event */) {
            callback(events::AgentEndEpisode(body), mind);
        }

        /** This method is called after initiation.
         * If the episode is turns-based it is called on the first mover only in order to start the episode.
         * If the episode is synchronous (e.g. rock-paper-scisors) it is called on both agents.
         * @return message that begins the episode
         */
        auto startEpisode() requires BodyMindPair<BODY,MIND> {
            callback(events::AgentStartEpisode(body), mind);
            return getNextActEvent().message;
        }

        /** If this body/mind pair is a monad (i.e. body never sends outgoing messages) then an episode
         * consists of act/reward between mind and body until body returns a not-a-number as reward */
         template<class... CALLBACKS>
        void runEpisode(CALLBACKS &&... callbacks) requires BodyMindMonad<BODY,MIND> {
            callback(events::AgentStartEpisode(body), mind, body, std::forward<CALLBACKS>(callbacks)...);
            auto response = getNextActEvent(std::forward<CALLBACKS>(callbacks)...);
            while(!response.isEndEpisode()) {
                response = getNextActEvent(std::forward<CALLBACKS>(callbacks)...);
            }
            callback(events::AgentEndEpisode(body), mind, body, std::forward<CALLBACKS>(callbacks)...);
        }


        /** This method is called when the agent receives a message from another agent
         * @param incomingMessage message sent by the other agent
         * @return response to incoming message. If unset, the agent has (perhaps unilaterally)
         *   ended the episode.
         */
        template<class INMESSAGE> requires BodyMindPair<BODY,MIND>
        auto handleMessage(INMESSAGE incomingMessage) {
            events::IncomingMessage inMessageEvent(std::move(incomingMessage), body.handleMessage(incomingMessage));
            callback(inMessageEvent,mind);
            return getNextActEvent().message;
        }


        friend std::ostream &operator <<(std::ostream &out, const Agent<BODY,MIND> &agent) {
            deselby::constexpr_if<deselby::IsStreamable<MIND>>([&out](auto &mind) { out << mind << std::endl; }, agent.mind);
            deselby::constexpr_if<deselby::IsStreamable<BODY>>([&out](auto &body) { out << body << std::endl; }, agent.body);
            return out;
        }
    protected:
        template<class... EXTRACALLBACKS>
        auto getNextActEvent(EXTRACALLBACKS &&... extraCallbacks) {
            callback(events::PreActBodyState(body), mind, std::forward<EXTRACALLBACKS>(extraCallbacks)...);
            action_type act = mind.act(body);
            events::Act actEvent(std::move(act), body.handleAct(act));
            callback(actEvent, mind, std::forward<EXTRACALLBACKS>(extraCallbacks)...);
            return actEvent;
        }
    };
}


#endif //MULTIAGENTGOVERNMENT_AGENT_H
