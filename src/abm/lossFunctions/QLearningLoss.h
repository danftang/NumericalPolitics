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
        arma::Row<char> isEndEpisode;
        arma::urowvec actionIndices;
        arma::rowvec rewards;
        size_t insertCol;
        ENDSTATEPREDICTOR endStatePredictor; // a function from end state to qVector
        size_t endStateParameterUpdateInterval;
        uint nParameterUpdates;
        double discount;
        bool bufferIsFull;

        arma::ucolvec batchCols;     // columns of the buffer in the current batch


        QLearningLoss(size_t bufferSize, size_t stateSize, size_t batchSize, double discount, ENDSTATEPREDICTOR endStatePredictor, size_t endStateParameterUpdateInterval) :
                stateHistory(stateSize, bufferSize),
                actionIndices(bufferSize),
                rewards(bufferSize),
                insertCol(-1),
                batchCols(batchSize),
                endStatePredictor(std::move(endStatePredictor)),
                endStateParameterUpdateInterval(endStateParameterUpdateInterval),
                nParameterUpdates(0),
                discount(discount),
                bufferIsFull(false) {
        }


        /** Remember last act, body state and reward */
        template<class BODY>
        void on(const events::PreActBodyState<BODY> &event) {
            if(++insertCol >= bufferSize()) {
                bufferIsFull = true;
                insertCol = 0;
            }
            isEndEpisode(insertCol) = 0;
            stateHistory.col(insertCol) = event.body.asMat();
        }


        /** Remember last act, body state and reward */
        template<class ACTION, class MESSAGE>
        void on(const events::Act<ACTION, MESSAGE> &actEvent) {
            rewards(insertCol) = actEvent.reward;
            actionIndices(insertCol) = actEvent.act;
        }


        /** learn from residual reward of end-game */
        template<class BODY>
        void on(const events::AgentEndEpisode<BODY> & /* event */) {
            isEndEpisode(insertCol) = 1;
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
            assert(bufferIsFull || insertCol > 0);
            if(bufferIsFull) {
                batchCols = arma::randi<arma::ucolvec>(batchCols.n_rows, arma::distr_param(1, bufferSize() - 1)).transform(
                        [insertCol = insertCol, buffSize = bufferSize()](auto i) {
                            return (i + insertCol) % buffSize;
                        });
            } else {
                batchCols = arma::randi<arma::ucolvec>(batchCols.n_rows, arma::distr_param(0, insertCol-1));
            }
            trainingPoints = stateHistory.cols(batchCols);
        }


        template<class OUTPUTS, class RESULT>
        void gradientByPrediction(const OUTPUTS &predictions, RESULT &gradient) {
            size_t nActions = predictions.n_rows;
            auto batchedActions = actionIndices.cols(batchCols);
            arma::urowvec batchActionElementIds = batchCols.t()*nActions + batchedActions;
            auto batchEndStates = stateHistory.cols(batchCols.transform([buffSize = bufferSize()](auto i) { return (i + 1) % buffSize; }));
            auto endStateQVals = endStatePredictor(batchEndStates).elem(batchActionElementIds);
            gradient.zeros();

            gradient.elem(batchActionElementIds) =
                    predictions.elem(batchActionElementIds)
                    - rewards(batchCols)
                    - endStateQVals %
                        isEndEpisode
                            .transform([discount = discount](bool isEnd) {
                                return isEnd?0.0:discount;
                            })
                            .cols(batchCols);
        }


        size_t bufferSize() const { return stateHistory.n_cols; }

    protected:
    };
}

#endif //MULTIAGENTGOVERNMENT_QLEARNINGLOSS_H
