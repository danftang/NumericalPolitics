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
        typename T::action_mask;
        { body.actToMessage(actFromMind) } -> std::convertible_to<typename T::message_type>;
        { body.messageToReward(messageFromEnvironment) } -> std::convertible_to<double>;
        { body.legalActs() } -> std::convertible_to<typename T::action_mask>;
    };

    template<class T>
    concept Mind = requires(T mind,
            T::observation_type observation,
            T::action_type      act,
            T::action_mask      actMask) {
//        { T::action_type::size } -> std::convertible_to<int>;
        { mind.act(observation, actMask) } -> std::same_as<typename T::action_type>;
        { mind.train(observation, act, 0.0, observation, true) }; // TODO: send reward only? More flexible? or add messages? or be more flexible with observation?
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
        typedef BODY                body_type;
        typedef MIND                mind_type;
        typedef BODY::message_type  message_type;
        typedef MIND::action_type   action_type;

        static constexpr double meanRewardDecay = 0.996;

        BODY body;
        std::optional<BODY> lastState;
        action_type lastAct;
        MIND mind;
        double rewardSinceLastChoice = 0.0;
        double exponentialMeanReward = 0.0;
        double rewardSinceLastEpisodeStart = 0.0;


        Agent(BODY body, MIND mind): body(std::move(body)), mind(std::move(mind)) { }

        // ------ Agent Interface ------

        /**
         * Call this to nudge the agent to be the first mover in an episodic interaction
         * @return message that begins the episode
         */
        message_type startEpisode() {
            lastAct = mind.act(body, body.legalActs());
            lastState = body;
            rewardSinceLastEpisodeStart = 0.0;
            rewardSinceLastChoice = 0.0;
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
            rewardSinceLastEpisodeStart += reward;
            if(incomingMessage == message_type::close) {
                // Other agent closed
                train(true);
                lastState.reset();
                return message_type::close; // This close formally ends the episode and should never be sent.
            }
            std::bitset<BODY::action_type::size> legalActMask = body.legalActs();
            int nLegalActs = legalActMask.count();
            if (nLegalActs == 0) {
                // No acts available so end the episode.
                train(true);
                lastState.reset();
                return message_type::close; // N.B. this close will be sent to the other agent
            }
            if (nLegalActs == 1) {
                int act = 0;
                while(legalActMask[act] == false) ++act;
                return body.actToMessage(act);
            }
            train(false);
            lastAct = mind.act(body, legalActMask);
            lastState = body;
            rewardSinceLastChoice = 0.0;
            message_type response = body.actToMessage(lastAct);
            if(response == message_type::close) {
                // last act has caused a close, so train on the last step
                // TODO: there may be a reward for lastAct, how do we get this? (send close message to body or split reward?)
                rewardSinceLastChoice = body.messageToReward(message_type::close);
                train(true);
            }
            return response;
        }

    protected:

        void train(bool endEpisode) {
            if(lastState.has_value()) {
//                std::cout << "training on " << lastState.value() << " " << lastAct << " " << rewardSinceLastChoice << " " << body << " " << endEpisode << std::endl;
                exponentialMeanReward = meanRewardDecay*exponentialMeanReward + (1.0-meanRewardDecay)*rewardSinceLastChoice;
                mind.train(lastState.value(), lastAct, rewardSinceLastChoice, body, endEpisode);
            } else {
                // must be start of episode, but we didn't initiate
                rewardSinceLastEpisodeStart = 0.0;
            }
        }

    };
}
#endif //MULTIAGENTGOVERNMENT_AGENT_H
