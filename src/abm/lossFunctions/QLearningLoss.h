//
// Created by daniel on 17/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_QLEARNINGLOSS_H
#define MULTIAGENTGOVERNMENT_QLEARNINGLOSS_H

#include <armadillo>
#include "../minds/qLearning/QLearningStepMixin.h"

namespace abm::lossFunctions {
    template<class ENDSTATEPREDICTOR>
    class QLearningLoss {
    public:
        // the buffer...
        arma::mat startStates;
        arma::urowvec actionIndices;
        arma::rowvec rewards;
        arma::mat endStates;
        size_t insertCol;
        ENDSTATEPREDICTOR endStatePredictor; // a function from end state to qVector
        double discount;

        arma::ucolvec batchCols;     // columns of the buffer in the current batch


        QLearningLoss(size_t bufferSize, size_t stateSize, size_t batchSize, double discount, ENDSTATEPREDICTOR endStatePredictor) :
                startStates(stateSize, bufferSize),
                actionIndices(bufferSize),
                rewards(bufferSize),
                endStates(stateSize, bufferSize),
                insertCol(0),
                batchCols(batchSize),
                endStatePredictor(std::move(endStatePredictor)),
                discount(discount) {
        }


        template<class INPUTS>
        void trainingSet(INPUTS &trainingPoints) {
            batchCols = arma::randi<arma::ucolvec>(batchCols.n_rows,arma::distr_param(0, bufferSize()-1));
            trainingPoints = startStates.cols(batchCols);
        }

        template<class OUTPUTS, class RESULT>
        void gradientByPrediction(const OUTPUTS &predictions, RESULT &gradient) {
            size_t nActions = predictions.n_rows;
            auto batchedActions = actionIndices.cols(batchCols);
            arma::urowvec batchActionElementIds = batchCols.t()*nActions + batchedActions;
            auto endStateQVals = endStatePredictor(endStates.cols(batchCols)).elem(batchActionElementIds);
            gradient.zeros();
            gradient.elem(batchActionElementIds) = predictions.elem(batchActionElementIds) - rewards(batchCols) - endStateQVals * discount;
        }


        template<class STATE>
        void on(const events::QLearningStep<STATE> &event) {
            startStates.col(insertCol) = *event.startStatePtr;
            actionIndices(insertCol) = event.action;
            rewards(insertCol) = event.reward;
            endStates(insertCol) = *event.endStatePtr;
            insertCol = (insertCol+1)%bufferSize();
        }

        size_t bufferSize() const {
            return startStates.n_cols;
        }


    };
}

#endif //MULTIAGENTGOVERNMENT_QLEARNINGLOSS_H
