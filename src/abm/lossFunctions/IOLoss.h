//
// Created by daniel on 17/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_IOLOSS_H
#define MULTIAGENTGOVERNMENT_IOLOSS_H

#include <armadillo>
#include <cassert>

namespace abm::events {
    /** Represents an observation of a (possibly batched) input/output pair of a function.
     * @tparam INPUT
     * @tparam OUTPUT
     */
    template<class INPUT, class OUTPUT>
    struct InputOutput {
        INPUT input;
        OUTPUT output;
    };
    template<class IN, class OUT>
    InputOutput(IN &&in, OUT &&out) -> InputOutput<IN,OUT>;
}

namespace abm::lossFunctions {
    /** loss function for a set of input/output points, giving 0.5 sum of squared error */
    class IOLoss {
    public:
        arma::mat inputs;
        arma::mat outputs;
        size_t insertCol = 0;
        bool isFull = false;

        IOLoss(size_t bufferSize, size_t inputSize, size_t outputSize) :
                inputs(inputSize, bufferSize),
                outputs(outputSize, bufferSize) {
        }


        template<class INPUT, class OUTPUT>
        void on(const abm::events::InputOutput<INPUT,OUTPUT> &event) {
            inputs.col(insertCol) = static_cast<const arma::mat &>(event.input);
            outputs.col(insertCol) = static_cast<const arma::mat &>(event.output);
            insertCol = (insertCol + 1)%capacity();
            if(insertCol == 0) isFull = true;
        }


        size_t capacity() const { return inputs.n_cols; }

        size_t batchSize() const { return isFull?capacity():insertCol; }


        template<class INPUTS>
        void trainingSet(INPUTS &&trainingPoints) {
            assert(batchSize() > 0);
            trainingPoints = inputs.cols(0, batchSize()-1);
        }


        template<class OUTPUTS, class RESULT>
        void gradientByPrediction(const OUTPUTS &predictions, RESULT &&gradient) {
            gradient = predictions - outputs.cols(0,batchSize()-1);
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_IOLOSS_H
