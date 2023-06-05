//
// Created by daniel on 05/06/23.
//

#ifndef MULTIAGENTGOVERNMENT_SUGARSPICETRADINGAGENT_H
#define MULTIAGENTGOVERNMENT_SUGARSPICETRADINGAGENT_H

#include <deque>
#include "../abm.h"
#include "../../DeselbyStd/random.h"
#include "mlpack.hpp"

namespace abm {
    namespace agents {


        class SugarSpiceTradingAgent {
        public:
            typedef uint time_type;
            const int NACTS = 6;
            const double discount = 0.9;
//            const int STATE_DIM = NACTS*maxHistoryLength + 3;

            enum ActionEnum {
                GiveSugar,
                GiveSpice,
                Fight,
                WalkAway,
                Say0,
                Say1
            };

            class Action {
            public:

                Action &operator =(const ActionEnum act) {
                    action = act;
                    return *this;
                }

                ActionEnum action;
                static const int size = 6;
            };


            class State {
            public:
                static const int maxHistoryLength = 6;
                static const int utilityOfPreferred = 10;      // utility of holding sugar/spice
                static const int utilityOfNonPreferred = 1;
                // Agent state
                int sugar; // amount of sugar
                int spice; // amount of spice
                bool prefersSugar; // this agent prefers sugar
                std::deque<ActionEnum> gameHistory;

                const arma::colvec& Encode() const { return data; }

                ActionEnum lastMove() const { return gameHistory.back(); }

                void insertMove(ActionEnum act) {
                    gameHistory.push_back(act);
                    if(gameHistory.size() > maxHistoryLength) gameHistory.pop_front();
                }

                double utility() const {
                    return prefersSugar?(sugar*utilityOfPreferred + spice*utilityOfNonPreferred):(sugar*utilityOfNonPreferred + spice*utilityOfPreferred);
                }

                static const size_t dimension = Action::size*6+3;

            };

            State state;
            State lastState;
            Action lastAction;

            mlpack::SimpleDQN<> learningNet = mlpack::SimpleDQN<>(100,50, NACTS);
            mlpack::SimpleDQN<> targetNet = mlpack::SimpleDQN<>(100,50, NACTS);
            mlpack::RandomReplay<SugarSpiceTradingAgent> replayBuffer = mlpack::RandomReplay<SugarSpiceTradingAgent>(64,100000);
            mlpack::GreedyPolicy<SugarSpiceTradingAgent> policy;

            CommunicationChannel<Schedule<time_type>, ActionEnum> otherPlayer;

            //SugarSpiceTradingAgent *otherPlayer;

            Schedule<time_type> handleTradingAct(ActionEnum otherPlayerMove, time_type time) {
                state.insertMove(otherPlayerMove);
                bool isEnd = false;
                std::optional<ActionEnum> myNextMove;
                switch(otherPlayerMove) {
                    case GiveSugar:
                        state.sugar += 1;
                        break;
                    case GiveSpice:
                        state.spice += 1;
                        break;
                    case Fight:
                        if(lastState.lastMove() == Fight) {
                            // I started fight but lost
                            state.sugar = 0;
                            state.spice = 0;
                            isEnd = true;
                        } else if(deselby::Random::nextBool(0.6)) {
                            // other agent started fight and won
                            state.sugar = 0;
                            state.spice = 0;
                            isEnd = true;
                            myNextMove = WalkAway;
                        } else {
                            // this agent won fight
                            state.sugar = 1;
                            state.spice = 1;
                            isEnd = true;
                            myNextMove = Fight;
                        }
                        break;
                    case WalkAway:
                        if(lastState.lastMove() == WalkAway || lastState.lastMove() == Fight) isEnd = true;
                        break;
                    case Say0:
                        break;
                    case Say1:
                        break;
                }
                double reward = state.utility() - lastState.utility();
                replayBuffer.Store(lastState, lastAction, reward, state, isEnd, discount);
                lastState = state;
                lastAction = myNextMove.value();

                if(isEnd) {
                    if(myNextMove.has_value()) return otherPlayer.send(myNextMove.value(), time);
                    return Schedule<time_type>();
                }
                arma::mat actionValue;
                learningNet.Predict(state.Encode(), actionValue);
                myNextMove = policy.Sample(actionValue).action;
                state.insertMove(myNextMove.value());
                if(myNextMove.value() == WalkAway && otherPlayerMove == WalkAway) { // end of game
                    replayBuffer.Store(lastState, lastAction, 0.0, state, true, discount);
                }
                return otherPlayer.send(myNextMove.value(), time);
            }
        };

    }
}


#endif //MULTIAGENTGOVERNMENT_SUGARSPICETRADINGAGENT_H
