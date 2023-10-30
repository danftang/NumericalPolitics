// Represents a Deep Q Network
//
// Created by daniel on 06/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_FEEDFORWARDNEURALNET_H
#define MULTIAGENTGOVERNMENT_FEEDFORWARDNEURALNET_H

#include "mlpack.hpp"

namespace approximators {
    /** use FNN instead  */
//    /** A parameterised function that is trainable on input/output batches and backed
//     * by a neural net.
//     *
//     * @tparam INPUT_DIMENSION
//     * @tparam OUTPUT_DIMENSION
//     * @tparam network_type
//     */
//    template<typename ObjectiveType = mlpack::MeanSquaredError,
//             typename InitializationRuleType = mlpack::HeInitialization,
//             typename OptimiserType = ens::Adam,
//             typename MatType = arma::mat>
//class FeedForwardNeuralNet: public mlpack::FFN<ObjectiveType,InitializationRuleType,MatType> {
//    public:
//
//        OptimiserType optimiser;
//
//        /**
//         * note that the input dimension of the network should have been set before calling
//         * this constructor (e.g. by calling Reset).
//         *
//         * @param network       neural network to use. Note that the input dimension of the
//         *                      network should have been set before calling this constructor
//         *                      (e.g. by calling Reset on the network)
//         * @param batchSize     size of the
//         * @param replayBufferSize
//         * @param discount
//         */
//         template<std::ranges::range LAYERS = std::initializer_list<mlpack::Layer<MatType> *>> requires std::convertible_to<typename LAYERS::value_type, mlpack::Layer<MatType> *>
//        FeedForwardNeuralNet(
////                size_t inputDimension,
//                LAYERS layers,
//                OptimiserType optimiser = ens::Adam(0.001,32,0.9,0.999,1e-8,10000,1e-5,false,false,false),
//                ObjectiveType outputLayer = ObjectiveType(),
//                InitializationRuleType initializeRule = InitializationRuleType())
//                :
//                mlpack::FFN<ObjectiveType, InitializationRuleType, MatType>(outputLayer, initializeRule),
//                optimiser(optimiser)
//                {
//                    for(mlpack::Layer<MatType> *layer : layers) {
//                        this->Add(layer);
//                    }
//  //                  this->Reset(inputDimension);
//                }
//
//
//        /**
//         * @param input batched inputs, each column is one input point
//         * @return the output of this function, f(input), for the input. Each column is the
//         *          mapped value of the corresponding input column.
//         */
//        MatType operator()(const MatType &input) {
//            MatType batchedOutput;
//            this->Predict(input, batchedOutput);
//            return batchedOutput; // rely on NRVO
//        }
//
//        /** Train on a batch of <input,output> pairs.
//         *
//         * @param trainingPairs domain points on which we're training. Packed into columns of a matrix.
//         *                      These can be matrices, references, const references etc. If matrices,
//         *                      be sure to use std::move when passing if you don't want to keep the values.
//         */
//        void on(events::InputOutput<MatType,MatType> &&trainingPairs) {
//            assert(trainingPairs.input.n_rows == this->InputDimenstions());
//            assert(trainingPairs.output.n_rows == this->network.outputDimensions());
//            assert(trainingPairs.input.n_cols == trainingPairs.output.n_cols);
//            this->Train(std::move(trainingPairs.input), std::move(trainingPairs.output), optimiser);
//        }
//
//
//        template<class OBJECTIVE>
//        MatType dObjective_dParameters(const MatType& inputs, OBJECTIVE objective) {
//            size_t begin = 0;
//            size_t end = this->network.Network().size();
//
//            // Ensure the network is valid.
//            CheckNetwork("FFN::Forward()", inputs.n_rows);
//
//            // We must always store a copy of the forward pass in `networkOutputs` in case
//            // we do a left pass.
//            // Forward pass, storing outputs in networkOutput
//            this->networkOutput.set_size(this->network.OutputSize(), inputs.n_cols);
//            this->network.Forward(inputs, this->networkOutput, begin, end);
////            // It's possible the user passed `networkOutput` as `results`; in this case,
////            // we don't need to create an alias.
////            if (&results != &this->networkOutput)
////                results = networkOutput;
//
//            const typename MatType::elem_type residual = objective(this->networkOutput);
////                    outputLayer.Forward(this->networkOutput, targets) + network.Loss();
////
////            // Compute the error of the output layer.
////            outputLayer.Backward(networkOutput, targets, error);
//            this->error = objective.dObj_dY(this->networkOutput);
//
//            // Perform the left pass.
//            this->network.Backward(this->networkOutput, this->error, this->networkDelta);
//
//            // Now compute the gradients.
//            // The gradient should have the same size as the parameters.
//            MatType gradients;
//            gradients.set_size(this->parameters.n_rows, this->parameters.n_cols);
//            this->network.Gradient(inputs, this->error, gradients);
//
//            return gradients;
//
//        }
//    };
}

#endif //MULTIAGENTGOVERNMENT_FEEDFORWARDNEURALNET_H
