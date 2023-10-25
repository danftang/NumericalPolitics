//
// Created by daniel on 10/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_FNN_H
#define MULTIAGENTGOVERNMENT_FNN_H

#include "mlpack.hpp"
#include "Concepts.h"

namespace abm::approximators {

    template<class T>
    concept InitializationRule = requires(T rule, arma::mat weights) { rule.Initialize(weights); };

    template<class MatType = arma::mat>
    class FNN {
    public:
        mlpack::MultiLayer<MatType> network;
        MatType params;


        template<InitializationRule INITRULE, class... LAYERS>
        FNN(INITRULE initializeRule, size_t inputDimensions, LAYERS &&... layers) {
            (network.Add(new std::remove_cvref_t<LAYERS>(std::forward<LAYERS>(layers))), ... );
            // set dimensionality
            network.InputDimensions() = {inputDimensions};
            network.ComputeOutputDimensions();

            // initialize the params
            mlpack::NetworkInitialization<INITRULE> networkInit(initializeRule);
            networkInit.Initialize(network.Network(), params);

            // Override the weight matrix.
            network.CustomInitialize(params, network.WeightSize());
            network.SetWeights(params.memptr());
        }

        template<class... LAYERS>
        FNN(size_t inputDimensions, LAYERS... layers) : FNN(mlpack::HeInitialization(), inputDimensions, layers...) {}

        FNN(const FNN<MatType> &other) : network(other.network), params(other.params) {
            // Set alias matrices to point to new parameter matrix
            network.CustomInitialize(params, network.WeightSize());
            network.SetWeights(params.memptr());
        }

        FNN(FNN<MatType> &&other) : network(std::move(other.network)), params(std::move(other.params)) {
            // Set alias matrices to point to new parameter matrix
            network.CustomInitialize(params, network.WeightSize());
            network.SetWeights(params.memptr());
        }

        /** Calculate network output given input */
        MatType operator()(const MatType &inputs) {
            MatType Y(network.OutputSize(), inputs.n_cols);
            assert(inputs.n_rows == network.InputDimensions()[0]);
            network.Training() = false;
            network.Forward(inputs, Y);
            return Y;
        }

        MatType &parameters() { return params; }

        template<LossFunction LOSS>
        MatType gradientByParams(LOSS &&loss) {
            // get the training set on which the loss function is defined
            MatType inputs(network.InputDimensions()[0], loss.nPoints());
            loss.trainingSet(inputs);

            // Ensure the inputs are of the right dimension
            assert(inputs.n_rows == network.InputDimensions()[0]);

            // Forward pass, storing outputs in Y
            network.Training() = true;
            MatType prediction(network.OutputSize(), inputs.n_cols);
            network.Forward(inputs, prediction);

            // calculate gradient of loss in output space
            MatType dLoss_dPred(prediction.n_rows, prediction.n_cols);
            loss.gradientByPrediction(prediction, dLoss_dPred);

            // Perform the back prop with the gradients in output space
            {
                MatType dObj_dinputs(inputs.n_rows, inputs.n_cols); // not used.
                network.Backward(prediction, dLoss_dPred, dObj_dinputs);
            }
            // Now compute the gradients in parameter space.
            // The gradient should have the same size as the params.
            MatType dLoss_dParams(this->params.n_rows, this->params.n_cols);
            network.Gradient(inputs, dLoss_dPred, dLoss_dParams);

            return dLoss_dParams;
        }


//        template<class LOSS>
//        std::pair<MatType, MatType>
//        outputAndGradientByParams(LOSS &loss) {
//            std::pair<MatType, MatType> result;
//
//            // Ensure the inputs are of the right dimension
//            assert(inputs.n_rows == network.InputDimensions()[0]);
//
//            network.Training() = true;
//
//            // Forward pass, storing outputs in networkOutput
//            result.first.set_size(network.OutputSize(), inputs.n_cols);
//            network.Forward(inputs, result.first);
//
//            MatType dObj_dY = gradientByOutput(result.first);
//
//            // Perform the left pass.
//            MatType dObj_dinputs; // not used.
//            network.Backward(result.first, dObj_dY, dObj_dinputs);
//
//            // Now compute the gradients.
//            // The gradient should have the same size as the parameters.
//            result.second.set_size(this->params.n_rows, this->params.n_cols);
//            network.Gradient(inputs, dObj_dY, result.second);
//
//            return result;
//        }
    };
}

#endif //MULTIAGENTGOVERNMENT_FNN_H
