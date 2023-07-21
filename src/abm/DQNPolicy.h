//
// Created by daniel on 08/06/23.
//

#ifndef MULTIAGENTGOVERNMENT_DQNPOLICY_H
#define MULTIAGENTGOVERNMENT_DQNPOLICY_H

#include "mlpack.hpp"

// TODO: separate Q-function from Q-policy which takes a Q-function
//   Do we also distinguish between policy to action and policy under which the Q-values are defined?
//   Pure policy:
//     sample(QValues)
//     probVector(QValues)
//   Pure Q function
//     Qvalues(state) - returns vector of Q values
//   Joint
//     VValue(state)
//     train(...) [needs VValue under some policy]
namespace abm {
    template<class AGENT>
    class DQNPolicy {
    public:
        static constexpr bool doubleQLearning = true;
        static constexpr bool noisyQLearning = false;
        static constexpr double optimisationStepSize = 0.001;
        static constexpr int targetNetworkSyncInterval = 100;
        static constexpr int explorationSteps = 128;

        static constexpr double initialExploration = 0.5;
        static constexpr size_t explorationDecayInterval = 10000;
        static constexpr double minExploration = 0.01;

        static constexpr size_t NACTIONS = AGENT::Action::size;

        inline static bool verbose = false;


//    mlpack::SimpleDQN<mlpack::HuberLoss,mlpack::HeInitialization> learningNetwork;
//    mlpack::SimpleDQN<mlpack::HuberLoss,mlpack::HeInitialization> targetNetwork;
        mlpack::SimpleDQN <mlpack::MeanSquaredError, mlpack::HeInitialization> learningNetwork;
        mlpack::SimpleDQN <mlpack::MeanSquaredError, mlpack::HeInitialization> targetNetwork;
        mlpack::RandomReplay <AGENT> replayBuffer;
        mlpack::GreedyPolicy <AGENT> policy;
        ens::AdamUpdate optimisation;
        ens::AdamUpdate::Policy <arma::mat, arma::mat> optimisationStep;
        int totalTrainingSteps;
        double discount;

        DQNPolicy(int inputLayerSize, int layer1size, int layer2size, int batchSize, int replayBufferSize,
                  double discount) :
                learningNetwork(layer1size, layer2size, NACTIONS),
                targetNetwork(layer1size, layer2size, NACTIONS),
                replayBuffer(batchSize, replayBufferSize, 1, inputLayerSize),
                policy(initialExploration, explorationDecayInterval, minExploration),
                optimisation(),
                optimisationStep(std::move(getOptPolicy(inputLayerSize))),
                totalTrainingSteps(0),
                discount(discount) {}


        int getAction(const AGENT::State &state) {
            arma::mat actionValue;
            learningNetwork.Predict(state.Encode(), actionValue);
            if (verbose) std::cout << actionValue.t();
            return policy.Sample(actionValue).action;
        }


        void setExploration(double epsilon) {
            policy = mlpack::GreedyPolicy<AGENT>(epsilon, explorationDecayInterval,
                                                 std::min(minExploration, epsilon));
        }


        void train(const AGENT::State &startState, AGENT::Action action, double reward, const AGENT::State &endState, bool isEnd) {
//                std::cout << startState << std::endl;
//                std::cout << action << std::endl;
//                std::cout << reward << std::endl;
//                std::cout << endState  << std::endl;
//                std::cout << isEnd << std::endl << std::endl;
            replayBuffer.Store(startState, action, reward, endState, isEnd, discount);
            ++totalTrainingSteps;
            if (totalTrainingSteps < explorationSteps) return;

            arma::mat sampledStates;
            std::vector<typename AGENT::Action> sampledActions;
            arma::rowvec sampledRewards;
            arma::mat sampledNextStates;
            arma::irowvec isTerminal;

            replayBuffer.Sample(sampledStates, sampledActions, sampledRewards,
                                sampledNextStates, isTerminal);
            // Compute action value for next state with target network.

            arma::mat nextTargetQValues;
            targetNetwork.Predict(sampledNextStates, nextTargetQValues);

            arma::Col<size_t> bestActions;
            if (doubleQLearning) {
                // If use double Q-Learning, use learning network to select the best action.
                arma::mat nextLearningQValues;
                learningNetwork.Predict(sampledNextStates, nextLearningQValues);
                bestActions = BestAction(nextLearningQValues);
            } else {
                bestActions = BestAction(nextTargetQValues);
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

            for (size_t i = 0; i < sampledNextStates.n_cols; ++i) {
                target(sampledActions[i].action, i) = sampledRewards(i) + discount *
                                                                          nextTargetQValues(bestActions(i), i) *
                                                                          (1 - isTerminal[i]);
            }

            // Learn from experience.
            arma::mat gradients;
            learningNetwork.Backward(sampledStates, target, gradients);

//            replayBuffer.Update(target, sampledActions, [target before updating], gradients);

            optimisationStep.Update(learningNetwork.Parameters(), optimisationStepSize,
                                    gradients);


            if (noisyQLearning == true) {
                learningNetwork.ResetNoise();
                targetNetwork.ResetNoise();
            }
            // Update target network.
            if (totalTrainingSteps % targetNetworkSyncInterval == 0)
                targetNetwork.Parameters() = learningNetwork.Parameters();


            if (totalTrainingSteps > explorationSteps)
                policy.Anneal();

        }

        static arma::Col <size_t> BestAction(const arma::mat &actionValues) {
            // Take best possible action at a particular instance.
            arma::Col<size_t> bestActions(actionValues.n_cols);
            arma::rowvec maxActionValues = arma::max(actionValues, 0);
            for (size_t i = 0; i < actionValues.n_cols; ++i) {
                bestActions(i) = arma::as_scalar(
                        arma::find(actionValues.col(i) == maxActionValues[i], 1));
            }
            return bestActions;
        };

    private:
        // generates an adamParameters step object during construction
        ens::AdamUpdate::Policy <arma::mat, arma::mat> getOptPolicy(int inputLayerSize) {
            learningNetwork.Reset(inputLayerSize);
            targetNetwork.Reset(inputLayerSize);
            targetNetwork.Parameters() = learningNetwork.Parameters();
            return {optimisation, learningNetwork.Parameters().n_rows, learningNetwork.Parameters().n_cols};
        }

    };
};

#endif //MULTIAGENTGOVERNMENT_DQNPOLICY_H
