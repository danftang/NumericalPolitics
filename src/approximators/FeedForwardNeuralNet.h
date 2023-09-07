// Represents a Deep Q Network
//
// Created by daniel on 06/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_FEEDFORWARDNEURALNET_H
#define MULTIAGENTGOVERNMENT_FEEDFORWARDNEURALNET_H

#include "mlpack.hpp"
#include "../observations/InputOutput.h"

namespace approximators {
    /** A parameterised function that is trainable on input/output batches and backed
     * by a neural net.
     *
     * @tparam INPUT_DIMENSION
     * @tparam OUTPUT_DIMENSION
     * @tparam network_type
     */
    template<typename ObjectiveType = mlpack::MeanSquaredError,
             typename InitializationRuleType = mlpack::HeInitialization,
             typename OptimiserType = ens::Adam,
             typename MatType = arma::mat>
class FeedForwardNeuralNet: public mlpack::FFN<ObjectiveType,InitializationRuleType,MatType> {
    public:

        OptimiserType optimiser;

        /**
         * note that the input dimension of the network should have been set before calling
         * this constructor (e.g. by calling Reset).
         *
         * @param network       neural network to use. Note that the input dimension of the
         *                      network should have been set before calling this constructor
         *                      (e.g. by calling Reset on the network)
         * @param batchSize     size of the
         * @param replayBufferSize
         * @param discount
         */
         template<std::ranges::range LAYERS> requires std::convertible_to<typename LAYERS::value_type, mlpack::Layer<MatType> *>
        FeedForwardNeuralNet(
                size_t inputDimension,
                LAYERS layers,
                OptimiserType optimiser = ens::Adam(0.001,32,0.9,0.999,1e-8,10000,1e-5,false,false,false),
                ObjectiveType outputLayer = ObjectiveType(),
                InitializationRuleType initializeRule = InitializationRuleType())
                :
                mlpack::FFN<ObjectiveType, InitializationRuleType, MatType>(outputLayer, initializeRule),
                optimiser(optimiser)
                {
                    for(mlpack::Layer<MatType> *layer : layers) {
                        this->Add(layer);
                    }
                    this->Reset(inputDimension);
                }


        /**
         * @param input batched inputs, each column is one input point
         * @return the output of this function, f(input), for the input. Each column is the
         *          mapped value of the corresponding input column.
         */
        MatType operator()(const MatType &input) {
            MatType batchedOutput;
            this->Predict(input, batchedOutput);
            return batchedOutput; // rely on NRVO
        }

        /** Train on a batch of <domain,range> pairs.
         *
         * @param trainingPairs domain points on which we're training. Packed into columns of a matrix
         */
        void train(const observations::InputOutput<MatType,MatType> &trainingPairs) {
            assert(trainingPairs.input.n_rows == INPUT_DIMENSION);
            assert(trainingPairs.output.n_rows == OUTPUT_DIMENSION);
            assert(trainingPairs.input.n_cols == trainingPairs.output.n_cols);
//            optimiser.MaxIterations() = trainingPairs.input.n_cols; // ensure we stop after one epoch TODO: put this in the higher level.
            this->train(trainingPairs.input, trainingPairs.output, optimiser);
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_FEEDFORWARDNEURALNET_H
