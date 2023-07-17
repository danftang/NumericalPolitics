// Represents a Deep Q Network
//
// Created by daniel on 06/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_DQN_H
#define MULTIAGENTGOVERNMENT_DQN_H

#include "mlpack.hpp"
#include "MlPackAction.h"

class DQN {
public:
    static constexpr double optimisationStepSize = 0.001;
    static constexpr int    targetNetworkSyncInterval = 100;

    class TrainingBatch {
    public:
        arma::mat startStates;
        std::vector<int> actions;
        arma::rowvec rewards;
        arma::mat endStates;
        arma::irowvec isTerminal;
        arma::mat learningStartStateQValues;
        arma::mat targetEndStateQValues;

        size_t size() const { return actions.size(); }
    };

    // Dummy environment for the mlpack replay
    class MatrixEnvironment {
    public:
        typedef int Action;
        class State: public arma::mat {
        public:
            const arma::mat &Encode() const { return *this; }
        };
    };

    mlpack::SimpleDQN<mlpack::MeanSquaredError,mlpack::HeInitialization> learningNetwork;
    mlpack::SimpleDQN<mlpack::MeanSquaredError,mlpack::HeInitialization> targetNetwork;
    mlpack::RandomReplay<MatrixEnvironment> replayBuffer;
    ens::AdamUpdate optimisation;
    ens::AdamUpdate::Policy<arma::mat, arma::mat> optimisationStep;
    uint totalTrainingSteps;
    double discount;
    int batchSize;

    template<class AGENTBODY>
    DQN(int layer1size, int layer2size, int batchSize, int replayBufferSize, double discount):
    DQN(AGENTBODY::State::dimension, layer1size, layer2size, AGENTBODY::Action::size, batchSize, replayBufferSize, discount) {}


    DQN(int inputLayerSize, int layer1size, int layer2size, int outputLayerSize,
        int batchSize, int replayBufferSize, double discount):
            learningNetwork(layer1size, layer2size, outputLayerSize),
            targetNetwork(layer1size, layer2size, outputLayerSize),
            replayBuffer(batchSize, replayBufferSize, 1, inputLayerSize),
            optimisation(),
            optimisationStep(std::move(getOptimisationPolicy(inputLayerSize))),
            totalTrainingSteps(0),
            discount(discount),
            batchSize(batchSize)
    { }


    // get the predicted Q-values for actions given agent state
    arma::colvec predict(const arma::mat &state) {
        arma::colvec actionValue;
        learningNetwork.Predict(state, actionValue);
        return actionValue;
    }


    void train(const arma::mat &startState, int action, double reward, const arma::mat &endState, bool isEnd) {
        replayBuffer.Store(reinterpret_cast<const MatrixEnvironment::State &>(startState), action, reward, reinterpret_cast<const MatrixEnvironment::State &>(endState), isEnd, discount);
        ++totalTrainingSteps;
        if(totalTrainingSteps < batchSize) return;

        // sample from training data and calculate predicted Q-values
        TrainingBatch trainingData;
        replayBuffer.Sample(trainingData.startStates, trainingData.actions, trainingData.rewards,
                            trainingData.endStates, trainingData.isTerminal);
        learningNetwork.Forward(trainingData.startStates,trainingData.learningStartStateQValues);
        targetNetwork.Forward(trainingData.endStates, trainingData.targetEndStateQValues);

        // calculate the Q values of the best action in the end state
        arma::rowvec nextStateMaxQ(trainingData.size());
        for (size_t i = 0; i < trainingData.size(); ++i) {
            nextStateMaxQ(i) = trainingData.isTerminal[i] == 0 ? trainingData.targetEndStateQValues.col(i).max() : 0.0;
        }

        // calculate Q-values for the learning network to learn from
        for (size_t i = 0; i < trainingData.size(); ++i) {
            trainingData.learningStartStateQValues(trainingData.actions[i], i) = trainingData.rewards(i) + discount * nextStateMaxQ(i);
        }

        // Update network parameters towards the calculated Q-values.
        arma::mat gradients;
        learningNetwork.Backward(trainingData.startStates, trainingData.learningStartStateQValues, gradients);
//        replayBuffer.Update(trainingData.learningStartStateQValues, trainingData.actions, trainingData.targetEndStateQValues, gradients);
        optimisationStep.Update(learningNetwork.Parameters(), optimisationStepSize,gradients);

        // Periodically copy learning network params to target network params.
        if (totalTrainingSteps % targetNetworkSyncInterval == 0)
            targetNetwork.Parameters() = learningNetwork.Parameters();
    }


private:
    // generates an optimisation step object during construction
    ens::AdamUpdate::Policy<arma::mat, arma::mat> getOptimisationPolicy(int inputLayerSize) {
        learningNetwork.Reset(inputLayerSize);
        targetNetwork.Reset(inputLayerSize);
        targetNetwork.Parameters() = learningNetwork.Parameters();
        return {optimisation, learningNetwork.Parameters().n_rows, learningNetwork.Parameters().n_cols};
    }

};


#endif //MULTIAGENTGOVERNMENT_DQN_H
