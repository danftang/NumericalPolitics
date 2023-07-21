// Represents a Deep Q Network
//
// Created by daniel on 06/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_DQN_H
#define MULTIAGENTGOVERNMENT_DQN_H

#include "mlpack.hpp"
#include "MlPackAction.h"
#include "mlpack.hpp"

class DQN {
public:
    static constexpr double optimisationStepSize = 0.001;
    static constexpr int    targetNetworkSyncInterval = 100;

    class TrainingBatch {
    public:
        arma::mat startStates;
        std::vector<int> intents;
        arma::rowvec rewards;
        arma::mat endStates;
        arma::irowvec isTerminal;
        arma::mat learningStartStateQValues;
        arma::mat targetEndStateQValues;

        size_t size() const { return intents.size(); }
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

    typedef mlpack::SimpleDQN<mlpack::MeanSquaredError,mlpack::HeInitialization> network_type;
    typedef arma::mat input_type;

    network_type learningNetwork;
    network_type targetNetwork;
    mlpack::RandomReplay<MatrixEnvironment> replayBuffer;
    uint totalTrainingSteps;
    double discount;
    int batchSize;
    ens::AdamUpdate adamParameters;
    ens::AdamUpdate::Policy<arma::mat, arma::mat> optimisationStep;

    template<class AGENTBODY>
    DQN(int layer1size, int layer2size, int batchSize, int replayBufferSize, double discount):
    DQN(AGENTBODY::State::dimension, layer1size, layer2size, AGENTBODY::Action::size, batchSize, replayBufferSize, discount) {}


    DQN(int inputLayerSize, int layer1size, int layer2size, int outputLayerSize,
        int batchSize, int replayBufferSize, double discount):
            learningNetwork(layer1size, layer2size, outputLayerSize),
            targetNetwork(layer1size, layer2size, outputLayerSize),
            replayBuffer(batchSize, replayBufferSize, 1, inputLayerSize),
            adamParameters(),
            totalTrainingSteps(0),
            discount(discount),
            batchSize(batchSize),
            optimisationStep(
                    getAdemUpdatePolicy(
                            adamParameters,
                            resetNetwork(learningNetwork, inputLayerSize).Parameters()
                            )
                            )
    {
        resetNetwork(targetNetwork, inputLayerSize);
        targetNetwork.Parameters() = learningNetwork.Parameters();
    }


    // get the predicted Q-values for actions given agent state
    arma::colvec predict(const arma::mat &state) {
        arma::colvec actionValue;
        learningNetwork.Predict(state, actionValue);
        return actionValue;
    }


    void train(const arma::mat &startState, int intent, double reward, const arma::mat &endState, bool isEnd) {
//        std::cout << "Training on " << std::endl << startState.t() << endState.t() << intent << " " << reward  << " " << isEnd << std::endl << std::endl;
        replayBuffer.Store(reinterpret_cast<const MatrixEnvironment::State &>(startState), intent, reward, reinterpret_cast<const MatrixEnvironment::State &>(endState), isEnd, discount);
        ++totalTrainingSteps;
        if(totalTrainingSteps < batchSize) return;

        // sample from training data and calculate predicted Q-values
        TrainingBatch trainingData;
        replayBuffer.Sample(trainingData.startStates, trainingData.intents, trainingData.rewards,
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
            trainingData.learningStartStateQValues(trainingData.intents[i], i) = trainingData.rewards(i) + discount * nextStateMaxQ(i);
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
    // generates an adamParameters step object during construction
    static network_type &resetNetwork (network_type &network, int inputLayerSize) {
        network.Reset(inputLayerSize);
        return network;
    }

    static ens::AdamUpdate::Policy<arma::mat, arma::mat> getAdemUpdatePolicy(ens::AdamUpdate &parent, const arma::mat &parameters) {
        return {parent, parameters.n_rows, parameters.n_cols};
    }

};


#endif //MULTIAGENTGOVERNMENT_DQN_H
