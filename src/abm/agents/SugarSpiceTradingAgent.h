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
            typedef abm::Schedule<time_type> schedule_type;
            static inline bool verboseMode = false;

            enum MessageEnum {
                // Agent actions
                GiveSugar,
                GiveSpice,
                Fight,
                WalkAway,
                Say0,
                Say1,
                // Out of band comms
                YouLostFight,
                YouWonFight,
                Bandits,
                EndByMutualConsent
            };

            // Action class so that this is a valid mlpack Environment
            class Action {
            public:
                Action(MessageEnum act): action(act) {}
                Action() {}

                MessageEnum action;
                static const int size = 6;
            };


        class State {
            public:
                static const int conversationHistoryLength = 2; // including conversation end tag
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
//                    if(!isOtherPlayer) return; // TODO: TEST!!!
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
            static constexpr double initialCostOfVerbosity = 0.05;
            static constexpr double deltaCostOfVerbosity = 0.25; // change in cost of verbosity per agent action
            static constexpr double kappaCostOfVerbosity = 1.13; // change in cost of verbosity per agent action
            static constexpr double costOfFighting = 1.5;
            static constexpr double pBanditAttack = 0.02; // probability of a bandit attack per message received


            double costOfVerbosity = initialCostOfVerbosity;

            State state;
            State stateBeforeLastAction;
            std::optional<MessageEnum> myLastAction;
//            DQNPolicy<Action::size> policy = DQNPolicy<Action::size>(State::dimension, 100,50,64,1024,1.0);
            QTablePolicy<State::nstates, Action::size> policy = QTablePolicy<State::nstates, Action::size>(1.0, 0.5, std::pow(0.02, 1.0/1000000.0), 0.01);

            CommunicationChannel<Schedule<time_type>, MessageEnum> otherPlayer;

//            SugarSpiceTradingAgent():
//                    policy(State::dimension, 100,50,64,100000,1.0) {}

            void connectTo(SugarSpiceTradingAgent &otherAgent) {
                otherPlayer.connectTo(otherAgent, &SugarSpiceTradingAgent::handleTradingAct, 1);
            }



            Schedule<time_type> start(time_type time=0) {
                MessageEnum message = makeMove(false);
                return otherPlayer.send(message, time);
            }

            void reset(bool hasSugar, bool hasSpice, bool prefersSugar) {
                state.reset(hasSugar, hasSpice, prefersSugar);
                myLastAction.reset();
                costOfVerbosity = initialCostOfVerbosity;
            }



            Schedule<time_type> handleTradingAct(MessageEnum opponentMessage, time_type time) {
                bool isEnd = false;
                switch (opponentMessage) {
                    case GiveSugar:
                        if (verboseMode) std::cout << "Give sugar" << std::endl;
                        state.sugar() += 1;
                        break;
                    case GiveSpice:
                        if (verboseMode) std::cout << "Give spice" << std::endl;
                        state.spice() += 1;
                        break;
                    case WalkAway:
                        if (verboseMode) std::cout << "Walk away" << std::endl;
                        break;
                    case Say0:
                        if (verboseMode) std::cout << "Say ogg" << std::endl;
                        state.recordSpeechAct(0, true);
                        break;
                    case Say1:
                        if (verboseMode) std::cout << "Say igg" << std::endl;
                        state.recordSpeechAct(1, true);
                        break;

                    case YouLostFight:
                        if (verboseMode) std::cout << "Fight, agressor wins" << std::endl;
                        if (state.sugar() > 0) state.sugar() = 0;
                        if (state.spice() > 0) state.spice() = 0;
                        isEnd = true;
                        break;
                    case YouWonFight:
                        if (verboseMode) std::cout << "Fight, agressor loses" << std::endl;
                        state.sugar() = 1;
                        state.spice() = 1;
                        isEnd = true;
                        break;
                    case Bandits:
                        if (verboseMode) std::cout << "Bandits!" << std::endl;
                        if (state.sugar() > 0) state.sugar() = 0;
                        if (state.spice() > 0) state.spice() = 0;
                        isEnd = true;
                        break;
                    case EndByMutualConsent:
                        if (verboseMode) std::cout << "Walk away" << std::endl;
                        isEnd = true;
                        break;
                }
//                if(myForcedMove.has_value()) state.insertMove(myForcedMove.value()); // not strictly necessary as end state
                if(myLastAction.has_value()) { // train on time from immediately before last action to immediately before next action
                    double reward = state.utility() - stateBeforeLastAction.utility();// - costOfVerbosity;
//                    costOfVerbosity += deltaCostOfVerbosity;
//                    costOfVerbosity *= kappaCostOfVerbosity;
                    if(opponentMessage == YouWonFight || opponentMessage == YouLostFight) reward -= costOfFighting;
//                    if(verboseMode) std::cout << stateBeforeLastAction.Encode().t() << state.Encode().t() << myLastAction.value() << " " <<  reward << " " << isEnd << std::endl;
                    policy.train(stateBeforeLastAction, myLastAction.value(), reward, state, isEnd);
                }

                if(isEnd) return Schedule<time_type>();

                MessageEnum myMessage = makeMove(opponentMessage == WalkAway);

                if(myMessage == EndByMutualConsent) { // I'm returning a WalkAway, which ends the game
                    policy.train(stateBeforeLastAction, WalkAway, 0.0, state, true);
                }
                if(myMessage == YouLostFight || myMessage == YouWonFight) {
                    double reward = state.utility() - stateBeforeLastAction.utility() - costOfFighting;
                    policy.train(stateBeforeLastAction, myLastAction.value(), reward, state, true);
                }
                if(myMessage == Bandits) {
                    double reward = state.utility() - stateBeforeLastAction.utility();
                    policy.train(stateBeforeLastAction, myLastAction.value(), reward, state, true);
                }

                return otherPlayer.send(myMessage, time);
            }


            MessageEnum makeMove(bool opponentMessageIsWalkaway) {
                MessageEnum message;
                stateBeforeLastAction = state;
                myLastAction = static_cast<MessageEnum>(policy.getAction(state));
                if(deselby::Random::nextBool(pBanditAttack)) {
                    message = Bandits;
                } else if((state.sugar() < 1.0 && myLastAction == GiveSugar) || (state.spice() < 1.0 && myLastAction == GiveSpice)) {
                    message = WalkAway;
                } else if(myLastAction == Fight) {
                    message = deselby::Random::nextBool(0.5)?YouWonFight:YouLostFight;
                } else {
                    message = myLastAction.value();
                }

                if(opponentMessageIsWalkaway && message == WalkAway) {
                    message = EndByMutualConsent;
                }
//                state.insertMove(myLastAction.value());
                switch(message) {
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
                        break;
                    case YouWonFight:
                        // I started fight and other won
                        if (state.sugar() > 0) state.sugar() = 0;
                        if (state.spice() > 0) state.spice() = 0;
                        break;
                    case YouLostFight:
                        // I started fight and other lost
                        state.sugar() = 1;
                        state.spice() = 1;
                        break;
                    case Bandits:
                        if (state.sugar() > 0) state.sugar() = 0;
                        if (state.spice() > 0) state.spice() = 0;
                        break;
                    default:
                        break;
                }
                return message;
            };
        };

    }
}


#endif //MULTIAGENTGOVERNMENT_SUGARSPICETRADINGAGENT_H
