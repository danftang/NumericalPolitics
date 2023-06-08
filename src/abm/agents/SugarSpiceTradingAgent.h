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
            const double discount = 1.0;
            static inline bool verboseMode = false;
//            const int STATE_DIM = NACTS*maxHistoryLength + 3;

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
                static const int conversationHistoryLength = 2; // including conversation end tag
                static const int utilityOfPreferred = 10;      // utility of holding sugar/spice
                static const int utilityOfNonPreferred = 1;

                // Agent state
                arma::colvec netInput; // state in the form of one-hot vectors of action history, plus sugar, spice and preference

//                double &sugar; // amount of sugar
//                double &spice; // amount of spice
//                double &prefersSugar; // this agent prefers sugar


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

//                ActionEnum lastMove() const {
//                    ActionEnum act = static_cast<ActionEnum>(0);
//                    while(netInput[act] == 0.0) act = static_cast<ActionEnum>(act + 1);
//                    return act;
//                }

//                void insertMove(ActionEnum act) {
//                    arma::colvec truncatedHistory = netInput.subvec(0, Action::size*(maxHistoryLength-1)-1);
//                    netInput.subvec(Action::size, Action::size*maxHistoryLength-1) = truncatedHistory;
//                    netInput.subvec(0, Action::size-1).fill(0.0);
//                    netInput[act] = 1.0;
//                }

                void recordSpeechAct(double word, bool isOtherPlayer) {
                    int bufferStart = isOtherPlayer?0:conversationHistoryLength;
                    arma::colvec truncatedHistory = netInput.subvec(bufferStart, bufferStart + conversationHistoryLength-2);
                    netInput.subvec(bufferStart + 1, bufferStart + conversationHistoryLength-1) = truncatedHistory;
                    netInput[bufferStart] = word;
                }

                void reset(bool hasSugar, bool hasSpice, bool prefersSugar) {
                    netInput.subvec(0, 2*conversationHistoryLength-1).fill(0.0);
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

                static const size_t dimension = 2*conversationHistoryLength+3;

            };

            static constexpr bool doubleQLearning = false;
            static constexpr bool noisyQLearning = false;
            static constexpr double optimisationStepSize = 0.001;
            static constexpr int targetNetworkSyncInterval = 10;
            static constexpr int explorationSteps = 100;

            static constexpr double rewardForNegativeInventory = -25;
            static constexpr double priceOfVerbosity = 0.0;

            State state;
            State stateBeforeLastAction;
            std::optional<ActionEnum> myLastAction;
            int totalTrainingSteps;

            mlpack::SimpleDQN<> learningNetwork;
            mlpack::SimpleDQN<> targetNetwork;
            mlpack::RandomReplay<SugarSpiceTradingAgent> replayBuffer;
            mlpack::GreedyPolicy<SugarSpiceTradingAgent> policy;
            ens::AdamUpdate optimisation;
            ens::AdamUpdate::Policy<arma::mat, arma::mat> optimisationStep;

            CommunicationChannel<Schedule<time_type>, ActionEnum> otherPlayer;

            SugarSpiceTradingAgent():
            totalTrainingSteps(0),
            learningNetwork(100,50, NACTS),
            targetNetwork(100,50, NACTS),
            replayBuffer(64,100000),
            policy(0.2, 100000, 0.005, 0.5),
            optimisationStep(optimisation, learningNetwork.Parameters().n_rows, learningNetwork.Parameters().n_cols)
            {

            }

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

            //SugarSpiceTradingAgent *otherPlayer;

            Schedule<time_type> handleTradingAct(ActionEnum otherPlayerMove, time_type time) {
//                std::cout << "Starting handler with state " << state << std::endl;
  //              state.insertMove(otherPlayerMove);
                bool isEnd = false;
                std::optional<ActionEnum> myForcedMove;
                switch(otherPlayerMove) {
                    case GiveSugar:
                        if(verboseMode) std::cout << "Give sugar" << std::endl;
                        state.sugar() += 1;
                        break;
                    case GiveSpice:
                        if(verboseMode) std::cout << "Give spice" << std::endl;
                        state.spice() += 1;
                        break;
                    case Fight:
                        if(verboseMode) std::cout << "Fight" << std::endl;
                        isEnd = true;
                        if(myLastAction == Fight) {
                            // I started fight but lost
                            if(state.sugar() > 0) state.sugar() = 0;
                            if(state.spice() > 0) state.spice() = 0;
                        } else if(deselby::Random::nextBool(0.6)) {
                            // other agent started fight and won
                            if(state.sugar() > 0) state.sugar() = 0;
                            if(state.spice() > 0) state.spice() = 0;
                            myForcedMove = WalkAway;
                        } else {
                            // other agent started fight but I won
                            state.sugar() = 1;
                            state.spice() = 1;
                            myForcedMove = Fight;
                        }
                        break;
                    case WalkAway:
                        if(verboseMode) std::cout << "Walk away" << std::endl;
                        if(myLastAction == WalkAway) isEnd = true;
                        if(myLastAction == Fight) {
                            // I started fight and won
                            state.sugar() = 1;
                            state.spice() = 1;
                            isEnd = true;
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
                double reward = state.utility() - stateBeforeLastAction.utility() - priceOfVerbosity;
                if(myLastAction.has_value()) storeAndTrain(stateBeforeLastAction, myLastAction.value(), reward, state, isEnd);

                if(isEnd) {
                    return myForcedMove.has_value() ? otherPlayer.send(myForcedMove.value(), time) : Schedule<time_type>();
                }

                makeMove();

                if(myLastAction == WalkAway && otherPlayerMove == WalkAway) { // end of game
//                    state.insertMove(WalkAway); // not strictly necessary
                    storeAndTrain(stateBeforeLastAction, WalkAway, 0.0, state, true);
                }
//                std::cout << "Agent " << this << "\tsending message " << myLastAction << std::endl;
                return otherPlayer.send(myLastAction.value(), time);
            }

            void makeMove() {
                stateBeforeLastAction = state;
                arma::mat actionValue;
                learningNetwork.Predict(state.Encode(), actionValue);
                myLastAction = policy.Sample(actionValue).action;

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

            void storeAndTrain(const State &startState, ActionEnum action, double reward, const State &endState, bool isEnd) {
//                std::cout << startState.netInput << std::endl;
//                std::cout << action << std::endl;
//                std::cout << reward << std::endl;
//                std::cout << endState.netInput << std::endl;
//                std::cout << isEnd << std::endl;
                replayBuffer.Store(startState, action, reward, endState, isEnd, discount);

                if(totalTrainingSteps < explorationSteps) return;

                arma::mat sampledStates;
                std::vector<Action> sampledActions;
                arma::rowvec sampledRewards;
                arma::mat sampledNextStates;
                arma::irowvec isTerminal;

                replayBuffer.Sample(sampledStates, sampledActions, sampledRewards,
                                    sampledNextStates, isTerminal);

                // Compute action value for next state with target network.

                arma::mat nextActionValues;
                targetNetwork.Predict(sampledNextStates, nextActionValues);

                arma::Col<size_t> bestActions;
                if (doubleQLearning)
                {
                    // If use double Q-Learning, use learning network to select the best action.
                    arma::mat nextActionValues;
                    learningNetwork.Predict(sampledNextStates, nextActionValues);
                    bestActions = BestAction(nextActionValues);
                }
                else
                {
                    bestActions = BestAction(nextActionValues);
                }

                // Compute the update target.
                arma::mat target;
                learningNetwork.Forward(sampledStates, target);

//                double discount = std::pow(, replayMethod.NSteps());

                /**
                 * If the agent is at a terminal state, then we don't need to add the
                 * discounted reward. At terminal state, the agent wont perform any
                 * action.
                 */
                for (size_t i = 0; i < sampledNextStates.n_cols; ++i)
                {
                    target(sampledActions[i].action, i) = sampledRewards(i) + discount *
                                                                              nextActionValues(bestActions(i), i) * (1 - isTerminal[i]);
                }

                // Learn from experience.
                arma::mat gradients;
                learningNetwork.Backward(sampledStates, target, gradients);

                replayBuffer.Update(target, sampledActions, nextActionValues, gradients);

                optimisationStep.Update(learningNetwork.Parameters(), optimisationStepSize,
                                     gradients);

                if (noisyQLearning == true)
                {
                    learningNetwork.ResetNoise();
                    targetNetwork.ResetNoise();
                }
                // Update target network.
                if (totalTrainingSteps % targetNetworkSyncInterval == 0)
                    targetNetwork.Parameters() = learningNetwork.Parameters();

                if (totalTrainingSteps > explorationSteps)
                    policy.Anneal();
                ++totalTrainingSteps;
            }

            static arma::Col<size_t> BestAction(const arma::mat& actionValues)
            {
                // Take best possible action at a particular instance.
                arma::Col<size_t> bestActions(actionValues.n_cols);
                arma::rowvec maxActionValues = arma::max(actionValues, 0);
                for (size_t i = 0; i < actionValues.n_cols; ++i)
                {
                    bestActions(i) = arma::as_scalar(
                            arma::find(actionValues.col(i) == maxActionValues[i], 1));
                }
                return bestActions;
            };


        };

    }
}


#endif //MULTIAGENTGOVERNMENT_SUGARSPICETRADINGAGENT_H
