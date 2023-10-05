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
#include "episodes/SimpleEpisode.h"

namespace abm::events {

    template<class BODY>
    struct AgentStartEpisode {
        BODY &body;

        explicit AgentStartEpisode(BODY &body): body(body) {}
    };

    template<class BODY>
    struct AgentEndEpisode {
        BODY &body;

        explicit AgentEndEpisode(BODY &body): body(body) {}
    };

    struct Reward {
        double reward;

        explicit Reward(const double &reward = 0.0): reward(reward) {}
        explicit Reward(double &&reward): reward(std::move(reward)) {}
    };

    template<class T>
    struct Action {
        T act;

        explicit Action(const T &act): act(act) {}
        explicit Action(T &&act): act(std::move(act)) {}
    };


    template<class MESSAGE>
    struct MessageReward : Reward {
        MESSAGE message;

        MessageReward(MESSAGE &&message, double &&reward): Reward(std::move(reward)), message(std::move(message)) {}
        MessageReward(MessageReward<MESSAGE> &&)=default;
    };

    template<class MESSAGE, class BODY>
    struct IncomingMessage : MessageReward<MESSAGE> {
        BODY &  body;     // state of body after handling message

        IncomingMessage(BODY &body, MESSAGE &&message, double &&reward): MessageReward<MESSAGE>(std::move(message),std::move(reward)), body(body) {};
    };

    template<class BODY, class ACTION, class MESSAGE>
    struct OutgoingMessage : Action<ACTION>, MessageReward<MESSAGE> {
        BODY &body;     // state of body after sending message

        /** generates an outgoing message in-place given a body and mind */
        template<class MIND>
        OutgoingMessage(BODY &body, MIND &mind):
                Action<ACTION>(mind.act(body)),
                MessageReward<MESSAGE>(body.handleAct(this->act)),
                body(body) { }
    };
    template<class BODY, class MIND>
    OutgoingMessage(BODY &body, MIND &mind) -> OutgoingMessage<BODY,decltype(mind.act(body)), decltype(body.handleAct(mind.act(body)).message)>;

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
            meanReward *= 1.0-a_n;
            meanReward += a_n * event.reward;
        }
    };
}

namespace abm {
    template<class BODY, class MIND>
    concept BodyMindPair = requires(BODY body, MIND mind) {
        { body.handleAct(mind.act(body)) } -> deselby::IsClassTemplateOf<events::MessageReward>;
        // and handleMessage(.) on some type of incoming message, depending on which agent it is paired with
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
    template<class BODY, class MIND> requires BodyMindPair<BODY,MIND>
    class Agent {
    public:
        BODY body;
        MIND mind;

        typedef BODY body_type;
        typedef MIND mind_type;

        // This design implies single action and message types
        typedef decltype(mind.act(body)) action_type;
        typedef decltype(body.handleAct(std::declval<action_type>()).message) message_type;
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
        message_type startEpisode() {
            callback(events::AgentStartEpisode(body), mind);
            return getNextOutMessage();
        }


        /** This method is called when the agent receives a message from another agent
         * @param incomingMessage message sent by the other agent
         * @return response to incoming message. If unset, the agent has (perhaps unilaterally)
         *   ended the episode.
         */
        template<class INMESSAGE>
        message_type handleMessage(INMESSAGE incomingMessage) {
            events::IncomingMessage inMessageEvent(
                    body,
                    std::move(incomingMessage),
                    body.handleMessage(incomingMessage));
            callback(inMessageEvent,mind);
            return getNextOutMessage();
        }


        friend std::ostream &operator <<(std::ostream &out, const Agent<BODY,MIND> &agent) {
            deselby::constexpr_if<deselby::IsStreamable<MIND>>([&out](auto &mind) { out << mind << std::endl; }, agent.mind);
            deselby::constexpr_if<deselby::IsStreamable<BODY>>([&out](auto &body) { out << body << std::endl; }, agent.body);
            return out;
        }
    protected:

        message_type getNextOutMessage() {
            events::OutgoingMessage outMessageEvent(body,mind);
            callback(outMessageEvent, mind);
            return outMessageEvent.message;
        }
    };




}


#endif //MULTIAGENTGOVERNMENT_AGENT_H
