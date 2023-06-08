//
// Created by daniel on 08/06/23.
//

#ifndef MULTIAGENTGOVERNMENT_DQNPOLICY_H
#define MULTIAGENTGOVERNMENT_DQNPOLICY_H

#include "mlpack.hpp"

template<int NACTIONS>
class DQNPolicy {
public:
    static constexpr bool   doubleQLearning = false;
    static constexpr bool   noisyQLearning = false;
    static constexpr double optimisationStepSize = 0.001;
    static constexpr int    targetNetworkSyncInterval = 10;
    static constexpr int    explorationSteps = 100;

    class DummyEnvironment {
    public:
        class Action {
        public:
            int action;
            static constexpr int size=NACTIONS;
        };

//        class State {
//
//        };
    };

    mlpack::SimpleDQN<> learningNetwork;
    mlpack::SimpleDQN<> targetNetwork;
    mlpack::RandomReplay<DummyEnvironment> replayBuffer;
    mlpack::GreedyPolicy<DummyEnvironment> policy;
    ens::AdamUpdate optimisation;
    ens::AdamUpdate::Policy<arma::mat, arma::mat> optimisationStep;
    int totalTrainingSteps;
    double discount;

    DQNPolicy(int layer1size, int layer2size, int batchSize, int replayBufferSize, double discount):
            totalTrainingSteps(0),
            learningNetwork(layer1size,layer2size, NACTIONS),
            targetNetwork(layer1size,layer2size, NACTIONS),
            replayBuffer(batchSize, replayBufferSize),
            policy(0.2, 100000, 0.005, 0.5),
            optimisationStep(optimisation, learningNetwork.Parameters().n_rows, learningNetwork.Parameters().n_cols),
            discount(discount)
            {}


    int getAction(const arma::mat &state) {
        arma::mat actionValue;
        learningNetwork.Predict(state, actionValue);
        return policy.Sample(actionValue).action;
    }

    void train(const arma::mat &startState, int action, double reward, const arma::mat &endState, bool isEnd) {
//                std::cout << startState.netInput << std::endl;
//                std::cout << action << std::endl;
//                std::cout << reward << std::endl;
//                std::cout << endState.netInput << std::endl;
//                std::cout << isEnd << std::endl;
        replayBuffer.Store(startState, action, reward, endState, isEnd, discount);

        if(totalTrainingSteps < explorationSteps) return;

        arma::mat sampledStates;
        std::vector<int> sampledActions;
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


#endif //MULTIAGENTGOVERNMENT_DQNPOLICY_H
