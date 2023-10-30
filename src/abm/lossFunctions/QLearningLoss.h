//
// Created by daniel on 17/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_QLEARNINGLOSS_H
#define MULTIAGENTGOVERNMENT_QLEARNINGLOSS_H

#include <armadillo>
#include "../minds/qLearning/QLearningStepMixin.h"


namespace abm::lossFunctions {
    template<approximators::ParameterisedFunction ENDSTATEPREDICTOR>
    class QLearningLoss {
    public:
        // the buffer...
        arma::mat stateHistory;
        arma::rowvec effectiveDiscount;
        arma::urowvec actionIndices;
        arma::rowvec rewards;
        size_t insertCol;
        bool bufferIsFull;

        ENDSTATEPREDICTOR endStatePredictor; // a function from end state to qVector
        size_t endStateParameterUpdateInterval;
        uint nParameterUpdates;
        double discount;

        arma::uvec batchCols;     // columns of the buffer in the current batch


        QLearningLoss(size_t bufferSize, size_t stateSize, size_t batchSize, double discount, const ENDSTATEPREDICTOR &endStatePredictor, size_t endStateParameterUpdateInterval) :
                stateHistory(stateSize, bufferSize),
                effectiveDiscount(bufferSize),
                actionIndices(bufferSize),
                rewards(bufferSize),
                insertCol(-1),
                bufferIsFull(false),
                endStatePredictor(endStatePredictor),
                endStateParameterUpdateInterval(endStateParameterUpdateInterval),
                nParameterUpdates(0),
                discount(discount),
                batchCols(batchSize)
        {
        }


        /** Remember body state directly before act */
        template<class BODY>
        void on(const events::PreActBodyState<BODY> &event) {
            if(++insertCol >= bufferSize()) {
                bufferIsFull = true;
                insertCol = 0;
            }
            effectiveDiscount(insertCol) = discount; // reset pending AgentStartEpisode event
            stateHistory.col(insertCol) = static_cast<const arma::mat &>(event.body);
        }


        /** Remember last act and reward and increment buffer pointer */
        template<class ACTION, class MESSAGE>
        void on(const events::Act<ACTION, MESSAGE> &actEvent) {
            rewards(insertCol) = actEvent.reward;
            actionIndices(insertCol) = actEvent.act;
        }


        template<class BODY>
        void on(const events::AgentEndEpisode<BODY> & /* event */) {
            effectiveDiscount(insertCol) = 0.0;
        }

        template<class PARAMS>
        void on(const events::ParameterUpdate<PARAMS> & event) {
            if(++nParameterUpdates % endStateParameterUpdateInterval == 0) {
                endStatePredictor.parameters() = event.parameters;
            }
        }

        size_t nPoints() { return batchCols.n_rows; }

        template<class INPUTS>
        void trainingSet(INPUTS &trainingPoints) {
            if(bufferIsFull) {
                batchCols = arma::randi<arma::uvec>(batchCols.n_rows, arma::distr_param(1, bufferSize() - 1));
                batchCols.transform([insertCol = insertCol, buffSize = bufferSize()](auto i) {
                            return (i + insertCol) % buffSize;
                        });
            } else {
                assert(insertCol > 0);
                batchCols = arma::randi<arma::uvec>(batchCols.n_rows, arma::distr_param(0, insertCol-1));
            }
            trainingPoints = stateHistory.cols(batchCols);
        }


        template<class OUTPUTS, class RESULT>
        void gradientByPrediction(const OUTPUTS &predictions, RESULT &gradient) {
            size_t batchSize = predictions.n_cols;
            arma::uvec endStateBatchCols(batchSize, arma::fill::none);
            for(int i=0; i<batchSize; ++i) endStateBatchCols(i) = (batchCols(i) + 1) % bufferSize();
            arma::mat batchEndStates  = stateHistory.cols(endStateBatchCols);
            arma::mat batchEndStateQVectors = endStatePredictor(batchEndStates);

            gradient.zeros();
            double scale = 2.0/batchSize;
            for (size_t i = 0; i < batchSize; ++i)
            {
                uint batchIndex = batchCols(i);
                uint action = actionIndices(batchIndex);
                gradient(action, i) = (predictions(action, i)
                        - rewards(batchIndex)
                        - batchEndStateQVectors.col(i).max() * effectiveDiscount(batchIndex)) * scale;
            }
        }


        size_t bufferSize() const { return stateHistory.n_cols; }

    protected:
    };
}

#endif //MULTIAGENTGOVERNMENT_QLEARNINGLOSS_H
