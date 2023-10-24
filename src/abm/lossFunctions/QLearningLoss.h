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

        arma::ucolvec batchCols;     // columns of the buffer in the current batch


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
            stateHistory.col(insertCol) = event.body.asMat();
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
            size_t batchSize = predictions.n_cols;
            arma::urowvec batchElements = arma::regspace<arma::urowvec>(0, nActions, batchSize*nActions - 1) + actionIndices.cols(batchCols);
            arma::mat batchEndStates = stateHistory.cols(batchCols.transform([buffSize = bufferSize()](auto i) { return (i + 1) % buffSize; }));
            arma::mat batchEndStateQVectors = endStatePredictor(batchEndStates);
//            auto batchEndStateQVals = arma::max(batchEndStateQVectors);
//            gradient.resize(nActions, batchSize);
            gradient.zeros();


            gradient.elem(batchElements) =
                    predictions.elem(batchElements)
                    - rewards.cols(batchCols).t()
                    - (arma::max(batchEndStateQVectors) % effectiveDiscount.cols(batchCols)).t();


//            std::cout << "Predictions = \n" << predictions.elem(batchElements).t() << std::endl;
//            std::cout << "Rewards = \n" << rewards.cols(batchCols) << std::endl;
//            std::cout << "EndState QVals = \n" << arma::max(batchEndStateQVectors) << std::endl;
//            std::cout << "EpisodeEnd = \n" << isEpisodeEnd.cols(batchCols) << std::endl;
//            std::cout << "Gradient = " << gradient.n_rows << " x " << gradient.n_cols << std::endl;
//            std::cout << gradient << std::endl;
        }


        size_t bufferSize() const { return stateHistory.n_cols; }

    protected:
    };
}

#endif //MULTIAGENTGOVERNMENT_QLEARNINGLOSS_H
