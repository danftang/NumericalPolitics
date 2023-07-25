//
// Created by daniel on 24/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_AGENT_H
#define MULTIAGENTGOVERNMENT_AGENT_H

#include <concepts>
#include <bitset>
#include <algorithm>

namespace abm {
    /**
     *
     * @tparam BODY
     * @tparam MIND implements action_type act(body, legalActs) and train(lastBodyState, lastAct, reward, currentBodyState)
     */
    template<class BODY, class MIND> requires(std::is_convertible_v<BODY, typename MIND::body_type>)
    class Agent {
    public:
        typedef BODY                body_type;
        typedef MIND                mind_type;
        typedef BODY::message_type  message_type;
        typedef BODY::action_type   action_type;


        Agent(BODY body, MIND mind): body(std::move(body)), mind(std::move(mind)) {

        }

        // ------ AGENT Interface ------

        /**
         * Call this to nudge the agent to be the first mover in an episodic interaction
         * @return message that begins the episode
         */
        message_type startEpisode() {
            lastAct = mind.act(body, body.legalActs());
            lastState = body;
            return body.handleAct(lastAct);
        }

        /**
         *
         * @param incomingMessage
         * @return response to incoming message
         */
        message_type handleMessage(message_type incomingMessage) {
            rewardSinceLastChoice += body.handleMessage(incomingMessage);
            std::bitset<BODY::action_type::size> legalActMask = body.legalActs();
            int nLegalActs = legalActMask.count();
            if (nLegalActs == 0) return message_type::close;
            if (nLegalActs == 1) {
                int act = 0;
                while(legalActMask[act] == false) ++act;
                return body.handleAct(act);
            }
            mind.train(lastState, lastAct, rewardSinceLastChoice, body, false);
            lastAct = mind.act(body, legalActMask);
            lastState = body;
            rewardSinceLastChoice = 0.0;

            return body.handleAct(lastAct);
        }

        /**
         * at the end of an episode, if we're not the agent that terminates,
         * we need to train on the remaining reward
         */
        void endEpisode() {
            rewardSinceLastChoice += body.endEpisode();
            mind.train(lastState, lastAct, rewardSinceLastChoice, lastState, true);
        }

        BODY body;
        BODY lastState;
        int  lastAct;
        MIND mind;
        double rewardSinceLastChoice = 0.0;

    };
}
#endif //MULTIAGENTGOVERNMENT_AGENT_H
