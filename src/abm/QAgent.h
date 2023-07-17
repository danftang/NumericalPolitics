//
// Created by daniel on 17/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_QAGENT_H
#define MULTIAGENTGOVERNMENT_QAGENT_H

#include "DQN.h"
#include <bitset>

namespace abm {
    template<class BODY, class QFUNCTION>
    class QAgent {
    public:
        typedef BODY::Action Action;
        typedef BODY::State State;


        State                                   state;
        QFUNCTION                               qFunction;
        std::function<int(const arma::colvec &, const std::bitset<Action::size> &)>   qPolicy;
        arma::mat                               lastState;
        int                                     lastAction;

        // qPolicy is a function from a vector of qValues and a legal-move mask to an action index
        QAgent(QFUNCTION qfunction, std::function<int(const arma::colvec &, const std::bitset<Action::size> &)> qPolicy = maxQPolicy):
        qFunction(std::move(qfunction)),
        qPolicy(std::move(qPolicy)),
        lastAction(-1) { }

        // The agent has just transitioned to the given state
        // and received the given reward
        // At the beginning of an episode, reward is ignored and newState is the initial state
        Action chooseAction(const State &newState, const double &reward) {
            arma::mat newStateMatrix = newState.Encode();
            arma::colvec qValues = qFunction.Predict(newStateMatrix);
            int nextAction = qPolicy(qValues, newState.legalActions());
            if(lastAction != -1) {
                qFunction.train(lastState, lastAction, reward, newStateMatrix, false);
            }
            lastState = std::move(newStateMatrix);
            lastAction = nextAction;
            return { nextAction };
        }

        // at the end of an episode, we want to train on the final action and reset the lastAction register
        // we're not interested in the final state as by definition it has zero value
        void endEpisode(const double &reward) {
            qFunction.train(lastState, lastAction, reward, lastState, true);
            lastAction = -1;
        }


        Action reactTo(Action incomingMessage) {
            double reward = state.transition(lastAction, incomingMessage);
            if(Action::isTerminal(lastAction, incomingMessage)) {
                endEpisode(reward);
                return -1;
            }
            Action response = chooseAction(state, reward);
            if(Action::isTerminal(incomingMessage, response)) {
                double finalReward = state.terminalHalfTransition(incomingMessage, response);
                endEpisode(finalReward);
            }
            return response;
        }

        static int maxQPolicy(const arma::colvec &qValues, const std::bitset<Action::size> &legalMoves) {
            double bestQ = -std::numeric_limits<double>::infinity();
            int besti = -1;
            for(int i=0; i< legalMoves.size(); ++i) {
                if(legalMoves[i] && qValues(i) > bestQ) {
                    bestQ = qValues[i];
                    besti = i;
                }
            }
            return besti;
        }

    };
}

#endif //MULTIAGENTGOVERNMENT_QAGENT_H
