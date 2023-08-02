//
// Created by daniel on 02/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_AGENT_H
#define MULTIAGENTGOVERNMENT_AGENT_H

#include <concepts>
#include <bitset>
#include <algorithm>
#include <cassert>

namespace abm {

    template<class T>
    concept Body = requires(T body, T::action_type actFromMind, T::message_type messageFromEnvironment) {
        { T::message_type::close };
        typename T::action_mask;
        { body.actToMessage(actFromMind) } -> std::convertible_to<typename T::message_type>;
        { body.messageToReward(messageFromEnvironment) } -> std::convertible_to<double>;
        { body.legalActs() } -> std::convertible_to<typename T::action_mask>;
    };

    template<class T>
    concept Mind = requires(T mind, T::observation_type observation, T::action_type act, T::action_mask actMask, double reward) {
        { mind.act(observation, actMask, reward) } -> std::same_as<typename T::action_type>;
        { mind.endEpisode(reward) };
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
    template<Body BODY, Mind MIND> requires(
            std::is_convertible_v<BODY, typename MIND::observation_type> &&
                    std::is_convertible_v<typename BODY::action_mask, typename MIND::action_mask>)
    class Agent {
    public:
        typedef BODY body_type;
        typedef MIND mind_type;
        typedef BODY::message_type message_type;
        typedef MIND::action_type action_type;

        BODY body;
        MIND mind;
        double rewardSinceLastChoice = 0.0;

        Agent(BODY body, MIND mind) : body(std::move(body)), mind(std::move(mind)) {}

        // ------ Agent Interface ------

        /**
         * Call this to nudge the agent to be the first mover in an episodic interaction
         * @return message that begins the episode
         */
        message_type startEpisode() {
            assert(rewardSinceLastChoice == 0.0);
            action_type lastAct = mind.act(body, body.legalActs(), std::numeric_limits<double>::quiet_NaN());
            return body.actToMessage(lastAct);
        }

        /**
         * Call this when the agent receives a message from another agent
         * @param incomingMessage message originating from another agent
         * @return response to incoming message
         */
        message_type handleMessage(message_type incomingMessage) {
            double reward = body.messageToReward(incomingMessage);
            rewardSinceLastChoice += reward;
            if (incomingMessage == message_type::close) {
                mind.endEpisode(rewardSinceLastChoice);
                rewardSinceLastChoice = 0.0;
                return message_type::close; // this should end all comms
            }
            auto legalActMask = body.legalActs();
            int nLegalActs = legalActMask.count();
            if (nLegalActs == 0) {
                // No acts available so end the episode.
                mind.endEpisode(rewardSinceLastChoice);
                rewardSinceLastChoice = 0.0;
                return message_type::close; // N.B. this close will be sent to the other agent
            }
            if (nLegalActs == 1) { // if only one option, don't class as choice point. NB: no discount either
                int act = 0;
                while (legalActMask[act] == false) ++act;
                return body.actToMessage(act);
            }
            action_type lastAct = mind.act(body, legalActMask, rewardSinceLastChoice);
            rewardSinceLastChoice = 0.0;
            message_type response = body.actToMessage(lastAct);
            if (response == message_type::close) {
                // last act has caused a close, so train on the last step
                double finalReward = body.messageToReward(message_type::close); // get any final rewards from last act
                mind.endEpisode(finalReward);
            }
            return response;
        }
    };
}


#endif //MULTIAGENTGOVERNMENT_AGENT_H
