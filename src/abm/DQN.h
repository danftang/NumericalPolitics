// Represents a Deep Q Network
//
// Created by daniel on 06/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_DQN_H
#define MULTIAGENTGOVERNMENT_DQN_H

#include "mlpack.hpp"
#include "RandomNoReplacementReplayBuffer.h"
#include "Transition.h"
#include "RandomReplay.h"

namespace abm {
    template<
            size_t input_dimension,
            size_t output_dimension,
            class network_type = mlpack::SimpleDQN<mlpack::MeanSquaredError, mlpack::HeInitialization>,
            class replay_buffer_type = RandomReplay>

    class DQN {
    public:
        static constexpr double optimisationStepSize = 0.001;
        static constexpr int targetNetworkSyncInterval = 100;

        class TrainingBatch {
        public:
            arma::mat startStates;
            std::vector<int> actions;
            arma::rowvec rewards;
            arma::mat endStates;
            arma::irowvec isTerminal;

            size_t size() const { return actions.size(); }
        };



        typedef arma::mat::fixed<input_dimension,1>     input_type;  // fixed size column vector
        typedef arma::mat::fixed<output_dimension,1>    output_type;

        network_type learningNetwork;
        network_type targetNetwork;
        replay_buffer_type replayBuffer;
        uint totalTrainingSteps;
        double discount;
        int batchSize;
        ens::AdamUpdate adamParameters;
        ens::AdamUpdate::Policy<arma::mat, arma::mat> optimisationStep;

//        template<class AGENTBODY>
//        DQN(int layer1size, int layer2size, int batchSize, int replayBufferSize, double discount):
//                DQN(AGENTBODY::State::dimension, layer1size, layer2size, AGENTBODY::Action::size, batchSize,
//                    replayBufferSize, discount) {
//
//                }

        // note that the input dimension of the network should have been set before calling
        // this constructor (e.g. by calling Reset).
        DQN(const network_type &network, int batchSize, int replayBufferSize, double discount):
        learningNetwork(network),
        targetNetwork(network),
        replayBuffer(batchSize, replayBufferSize, input_dimension),
        adamParameters(),
        totalTrainingSteps(0),
        discount(discount),
        batchSize(batchSize),
        optimisationStep(adamParameters,network.Parameters().n_rows, network.Parameters().n_cols)  {}

        DQN(int layer1size, int layer2size, int batchSize, int replayBufferSize, double discount) :
        DQN(makeSimpleDQN(layer1size, layer2size), batchSize, replayBufferSize, discount) {}


        // get the predicted Q-values for actions given agent state
        arma::mat::fixed<output_dimension,1> predict(const arma::mat &state) {
            arma::mat::fixed<output_dimension,1> actionValue;
            arma::mat mutableMatRef(actionValue.memptr(), output_dimension, 1, false, true);
            learningNetwork.Predict(state, mutableMatRef);
            return actionValue;
        }

        template<class STATE> requires(std::is_convertible_v<STATE,arma::mat>)
        void train(const STATE &startState, int action, const double &reward, const STATE &endState, bool isTerminal) {
//        std::cout << "Training on " << std::endl << startState.t() << endState.t() << action << " " << reward  << " " << isEnd << std::endl << std::endl;
            replayBuffer.Store(startState, action, reward, endState, isTerminal);
            ++totalTrainingSteps;
            if (totalTrainingSteps < batchSize) return;

            // TODO: Make replay buffer deal in matrices so that we can send the whole batch through the network
            //  in one go.
            // sample from training data and calculate predicted Q-values
            TrainingBatch trainingData;
            replayBuffer.Sample(trainingData.startStates, trainingData.actions, trainingData.rewards,
                                trainingData.endStates, trainingData.isTerminal);
            arma::mat learningStartStateQValues;
            arma::mat targetEndStateQValues;

            learningNetwork.Forward(trainingData.startStates, learningStartStateQValues);
            targetNetwork.Forward(trainingData.endStates, targetEndStateQValues);

            // calculate the Q values of the best action in the end state
            arma::rowvec nextStateMaxQ(trainingData.size());
            for (size_t i = 0; i < trainingData.size(); ++i) {
                nextStateMaxQ(i) =
                        trainingData.isTerminal[i] == 0 ? targetEndStateQValues.col(i).max() : 0.0;
            }

            // calculate Q-values for the learning network to learn from
            for (size_t i = 0; i < trainingData.size(); ++i) {
                learningStartStateQValues(trainingData.actions[i], i) =
                        trainingData.rewards(i) + discount * nextStateMaxQ(i);
            }

            // Update network parameters towards the calculated Q-values.
            arma::mat gradients;
            learningNetwork.Backward(trainingData.startStates, learningStartStateQValues, gradients);
//        replayBuffer.Update(trainingData.learningStartStateQValues, trainingData.actions, trainingData.targetEndStateQValues, gradients);
            optimisationStep.Update(learningNetwork.Parameters(), optimisationStepSize, gradients);

            // Periodically copy learning network params to target network params.
            if (totalTrainingSteps % targetNetworkSyncInterval == 0)
                targetNetwork.Parameters() = learningNetwork.Parameters();
        }


    private:
        // generates an adamParameters step object during construction
//        static network_type &resetNetwork(network_type &network, int inputLayerSize) {
//            network.Reset(inputLayerSize);
//            return network;
//        }


        static mlpack::SimpleDQN<mlpack::MeanSquaredError, mlpack::HeInitialization> makeSimpleDQN(int layer1Size, int layer2Size) {
            mlpack::SimpleDQN<mlpack::MeanSquaredError, mlpack::HeInitialization> network(layer1Size, layer2Size, output_dimension);
            network.Reset(input_dimension);
            return network;
        }

    };
}

#endif //MULTIAGENTGOVERNMENT_DQN_H
