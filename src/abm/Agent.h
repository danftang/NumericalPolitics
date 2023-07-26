//
// Created by daniel on 24/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_AGENT_H
#define MULTIAGENTGOVERNMENT_AGENT_H

#include <concepts>
#include <bitset>
#include <algorithm>

namespace abm {

    template<class T>
    concept Body = requires(T body,
            T::action_type  actFromMind,
            T::message_type messageFromEnvironment) {
        { T::message_type::close };
        { body.actToMessage(actFromMind) } -> std::convertible_to<typename T::message_type>;
        { body.messageToReward(messageFromEnvironment) } -> std::convertible_to<double>;
//        { body.endEpisode() } -> std::convertible_to<double>;
        { body.legalActs() } -> std::convertible_to<std::bitset<T::action_type::size>>;
    };

    template<class T>
    concept Mind = requires(T mind,
            T::observation_type observation,
            T::action_type      act) {
        { T::action_type::size } -> std::convertible_to<int>;
        { mind.act(observation, std::declval<std::bitset<T::action_type::size>>()) } -> std::same_as<typename T::action_type>;
        { mind.train(observation, act, 0.0, observation, true) };
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
    template<Body BODY, Mind MIND> requires(std::is_convertible_v<BODY, typename MIND::observation_type>)
    class Agent {
    public:
        typedef BODY                body_type;
        typedef MIND                mind_type;
        typedef BODY::message_type  message_type;
        typedef MIND::action_type   action_type;


        Agent(BODY body, MIND mind): body(std::move(body)), mind(std::move(mind)) { }

        // ------ AGENT Interface ------

        /**
         * Call this to nudge the agent to be the first mover in an episodic interaction
         * @return message that begins the episode
         */
        message_type startEpisode() {
            lastAct = mind.act(body, body.legalActs());
            lastState = body;
            return body.actToMessage(lastAct);
        }

        /**
         * Call this when the agent receives a message from another agent
         * @param incomingMessage message originating from another agent
         * @return response to incoming message
         */
        message_type handleMessage(message_type incomingMessage) {
            rewardSinceLastChoice += body.messageToReward(incomingMessage);
            std::bitset<BODY::action_type::size> legalActMask = body.legalActs();
            int nLegalActs = legalActMask.count();
            if (nLegalActs == 0) return message_type::close;
            if (nLegalActs == 1) {
                int act = 0;
                while(legalActMask[act] == false) ++act;
                return body.actToMessage(act);
            }
            mind.train(lastState, lastAct, rewardSinceLastChoice, body, false);
            lastAct = mind.act(body, legalActMask);
            lastState = body;
            rewardSinceLastChoice = 0.0;

            return body.actToMessage(lastAct);
        }

        /**
         * at the end of an episode, if we're not the agent that terminates,
         * we need to train on the remaining reward and maybe do some clean-up.
         */
//        void endEpisode() {
//            rewardSinceLastChoice += body.endEpisode();
//            mind.train(lastState, lastAct, rewardSinceLastChoice, lastState, true);
//        }

        BODY body;
        BODY lastState;
        action_type lastAct;
        MIND mind;
        double rewardSinceLastChoice = 0.0;

    };
}
#endif //MULTIAGENTGOVERNMENT_AGENT_H
