//
// Created by daniel on 17/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_IOLOSS_H
#define MULTIAGENTGOVERNMENT_IOLOSS_H

#include <armadillo>
#include "../approximators/InputOutput.h"

namespace abm::lossFunctions {
    /** loss function for a set of input/output points, giving 0.5 sum of squared error */
    class IOLoss {
    public:
        arma::mat inputs;
        arma::mat outputs;
        size_t insertCol;

        arma::ucolvec batchCols;


        IOLoss(size_t bufferSize, size_t inputSize, size_t outputSize, size_t batchSize) :
                inputs(inputSize, bufferSize),
                outputs(outputSize, bufferSize),
                batchCols(batchSize) {
        }


        template<class INPUT, class OUTPUT>
        void on(const approximators::events::InputOutput<INPUT,OUTPUT> &event) {
            inputs.col(insertCol) = event.input;
            outputs.col(insertCol) = event.output;
            insertCol = (insertCol + 1)%bufferSize();
        }


        size_t bufferSize() const { return inputs.n_cols; }


        template<class INPUTS>
        void trainingSet(INPUTS &trainingPoints) {
            batchCols = arma::randi<arma::ucolvec>(batchCols.n_rows, arma::distr_param(0, bufferSize() - 1));
            trainingPoints = inputs.cols(batchCols);
        }


        template<class OUTPUTS, class RESULT>
        void gradientByPrediction(const OUTPUTS &predictions, RESULT &gradient) {
            gradient = predictions - outputs.cols(batchCols);
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_IOLOSS_H
