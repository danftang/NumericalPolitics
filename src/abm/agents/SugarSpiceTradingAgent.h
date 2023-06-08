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
            static inline bool verboseMode = false;

            enum ActionEnum {
                GiveSugar,
                GiveSpice,
                Fight,
                WalkAway,
                Say0,
                Say1

            };

            // Action class so that this is a valid mlpack Environment
            class Action {
            public:
                Action(ActionEnum act): action(act) {}
                Action() {}

                ActionEnum action;
                static const int size = 6;
            };


        class State {
            public:
                static const int conversationHistoryLength = 3; // including conversation end tag
                static const int utilityOfPreferred = 10;      // utility of holding sugar/spice
                static const int utilityOfNonPreferred = 1;

                // Agent state
                arma::colvec netInput; // state in the form of one-hot vectors of action history, plus sugar, spice and preference

                State(): netInput(dimension) { }

                double &sugar() { return netInput[2*conversationHistoryLength]; }
                double &spice() { return netInput[2*conversationHistoryLength+1]; }
                double &prefersSugar() { return netInput[2*conversationHistoryLength+2]; }

                double sugar() const { return netInput[2*conversationHistoryLength]; }
                double spice() const { return netInput[2*conversationHistoryLength+1]; }
                double prefersSugar() const { return netInput[2*conversationHistoryLength+2]; }

                const arma::colvec &Encode() const {
                    return netInput;
                }

                operator const arma::colvec &() const {
                    return netInput;
                }

               operator int() const {
                    int ordinal = 0;
                    for(int i=0; i<netInput.size(); ++i) {
                        if(netInput[i] > 0.5) ordinal += 1<<i;
                    }
                    return ordinal;
                }

                void recordSpeechAct(double word, bool isOtherPlayer) {
                    int bufferStart = isOtherPlayer?0:conversationHistoryLength;
                    arma::colvec truncatedHistory = netInput.subvec(bufferStart, bufferStart + conversationHistoryLength-2);
                    netInput.subvec(bufferStart + 1, bufferStart + conversationHistoryLength-1) = truncatedHistory;
                    netInput[bufferStart] = word;
                }

                void reset(bool hasSugar, bool hasSpice, bool prefersSugar) {
                    netInput.subvec(0, 2*conversationHistoryLength-1).fill(0.0); // reset conversation
                    netInput[0] = 1.0; // add end-of-conversation tags
                    netInput[conversationHistoryLength] = 1.0;
                    sugar() = hasSugar;
                    spice() = hasSpice;
                    this->prefersSugar() = prefersSugar;
                }

                double utility() const {
                    double utilityOfSugar = (prefersSugar()>0.5?utilityOfPreferred:utilityOfNonPreferred);
                    double utilityOfSpice = (prefersSugar()>0.5?utilityOfNonPreferred:utilityOfPreferred);
                    return sugar()*utilityOfSugar + spice()*utilityOfSpice;
                }

                friend std::ostream &operator <<(std::ostream &out, const SugarSpiceTradingAgent::State &state) {
//                    out << state.sugar() << state.spice() << state.prefersSugar();
                    out << state.netInput.t();
                    return out;
                }

                static constexpr size_t dimension = 2*conversationHistoryLength+3;
                static constexpr size_t nstates = 1<<dimension;

            };

//            static constexpr double rewardForNegativeInventory = -25;
            static constexpr double priceOfVerbosity = 0.75;

            State state;
            State stateBeforeLastAction;
            std::optional<ActionEnum> myLastAction;
//            DQNPolicy<Action::size> policy = DQNPolicy<Action::size>(State::dimension, 100,50,64,100000,1.0);
            QTablePolicy<State::nstates, Action::size> policy = QTablePolicy<State::nstates, Action::size>(1.0, 0.2, 0.9999, 0.01);

            CommunicationChannel<Schedule<time_type>, ActionEnum> otherPlayer;

//            SugarSpiceTradingAgent():
//                    policy(State::dimension, 100,50,64,100000,1.0) {}

            void connectTo(SugarSpiceTradingAgent &otherAgent) {
                otherPlayer.connectTo(otherAgent, &SugarSpiceTradingAgent::handleTradingAct, 1);
            }

            Schedule<time_type> start(time_type time=0) {
                makeMove();
                return otherPlayer.send(myLastAction.value(), time);
            }

            void reset(bool hasSugar, bool hasSpice, bool prefersSugar) {
                state.reset(hasSugar, hasSpice, prefersSugar);
                myLastAction.reset();
            }


            Schedule<time_type> handleTradingAct(ActionEnum otherPlayerMove, time_type time) {
//                std::cout << "Starting handler with state " << state << std::endl;
  //              state.insertMove(otherPlayerMove);
                bool isEnd = false;
                std::optional<ActionEnum> myForcedMove;
                switch(otherPlayerMove) {
                    case GiveSugar:
                        if (verboseMode) std::cout << "Give sugar" << std::endl;
                        state.sugar() += 1;
                        break;
                    case GiveSpice:
                        if (verboseMode) std::cout << "Give spice" << std::endl;
                        state.spice() += 1;
                        break;
                    case Fight:
                        isEnd = true;
                        if (myLastAction == Fight) {
                            // I started fight but lost
                            if (verboseMode) std::cout << "Agressor loses" << std::endl;
                            if (state.sugar() > 0) state.sugar() = 0;
                            if (state.spice() > 0) state.spice() = 0;
                        } else {
                            if (verboseMode) std::cout << "Start fight" << std::endl;
                            if (deselby::Random::nextBool(0.5)) {
                                // other agent started fight and won
                                if (state.sugar() > 0) state.sugar() = 0;
                                if (state.spice() > 0) state.spice() = 0;
                                myForcedMove = WalkAway;
                            } else {
                                // other agent started fight but I won
                                state.sugar() = 1;
                                state.spice() = 1;
                                myForcedMove = Fight;
                            }
                        }
                        break;
                    case WalkAway:
                        if(myLastAction == WalkAway) isEnd = true;
                        if(myLastAction == Fight) {
                            // I started fight and won
                            if(verboseMode) std::cout << "Agressor wins" << std::endl;
                            state.sugar() = 1;
                            state.spice() = 1;
                            isEnd = true;
                        } else {
                            if(verboseMode) std::cout << "Walk away" << std::endl;
                        }
                        break;
                    case Say0:
                        if(verboseMode) std::cout << "Say 0" << std::endl;
                        state.recordSpeechAct(0, true);
                        break;
                    case Say1:
                        if(verboseMode) std::cout << "Say 1" << std::endl;
                        state.recordSpeechAct(1, true);
                        break;
                }
//                if(myForcedMove.has_value()) state.insertMove(myForcedMove.value()); // not strictly necessary as end state
                if(myLastAction.has_value()) {
                    double reward = state.utility() - stateBeforeLastAction.utility() - priceOfVerbosity;
//                    if(verboseMode) std::cout << stateBeforeLastAction.Encode().t() << state.Encode().t() << myLastAction.value() << " " <<  reward << " " << isEnd << std::endl;
                    policy.train(stateBeforeLastAction, myLastAction.value(), reward, state, isEnd);
                }

                if(isEnd) {
                    return myForcedMove.has_value() ? otherPlayer.send(myForcedMove.value(), time) : Schedule<time_type>();
                }

                makeMove();

                if(myLastAction == WalkAway && otherPlayerMove == WalkAway) { // end of game
//                    state.insertMove(WalkAway); // not strictly necessary
                    policy.train(stateBeforeLastAction, WalkAway, 0.0, state, true);
                }
//                std::cout << "Agent " << this << "\tsending message " << myLastAction << std::endl;
                return otherPlayer.send(myLastAction.value(), time);
            }

            void makeMove() {
                stateBeforeLastAction = state;
//                arma::mat actionValue;
//                learningNetwork.Predict(state.Encode(), actionValue);
//                myLastAction = policy.Sample(actionValue).action;
                myLastAction = static_cast<ActionEnum>(policy.getAction(state));

                // giving something one doesn't have is interpreted as WalkAway
                if((state.sugar() < 1.0 && myLastAction == GiveSugar) || (state.spice() < 1.0 && myLastAction == GiveSpice)) {
                    myLastAction = WalkAway;
                }
//                state.insertMove(myLastAction.value());
                switch(myLastAction.value()) {
                    case GiveSugar:
                        state.sugar() -= 1;
                        break;
                    case GiveSpice:
                        state.spice() -= 1;
                        break;
                    case Say0:
                        state.recordSpeechAct(0, false);
                        break;
                    case Say1:
                        state.recordSpeechAct(1, false);
                    default:
                        break;
                }
            };
        };

    }
}


#endif //MULTIAGENTGOVERNMENT_SUGARSPICETRADINGAGENT_H
