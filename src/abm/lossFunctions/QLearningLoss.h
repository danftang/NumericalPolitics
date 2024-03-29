//
// Created by daniel on 17/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_QLEARNINGLOSS_H
#define MULTIAGENTGOVERNMENT_QLEARNINGLOSS_H

#include <armadillo>
#include "../minds/qLearning/QLearningStepMixin.h"
#include "../Concepts.h"
#include "../approximators/AdaptiveFunction.h"

namespace abm::lossFunctions {
    template<ParameterisedFunction ENDSTATEPREDICTOR>
    class QLearningLoss {
    public:
        static constexpr int unsetAct = -1; // value in action column that indicates "empty"

        // the buffer...
        arma::mat stateHistory;
        arma::rowvec effectiveDiscount;
        arma::irowvec actionIndices;
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
                insertCol(0),
                bufferIsFull(false),
                endStatePredictor(endStatePredictor),
                endStateParameterUpdateInterval(endStateParameterUpdateInterval),
                nParameterUpdates(0),
                discount(discount),
                batchCols(batchSize)
        {
            actionIndices.fill(-1);
        }


        /** Remember body state directly before act */
        template<class BODY>
        void on(const events::PreActBodyState<BODY> &event) {
            stateHistory.col(insertCol) = static_cast<const arma::mat &>(event.body);
        }


        /** Remember last act and reward and increment buffer pointer */
        template<class ACTION, class MESSAGE>
        void on(const events::AgentStep<ACTION, MESSAGE> &actEvent) {
            rewards(insertCol) = actEvent.reward;
            actionIndices(insertCol) = actEvent.act;
        }

        template<class MESSAGE>
        void on(const events::IncomingMessage<MESSAGE> &event) {
            if(actionIndices(insertCol) != unsetAct) { // don't record second mover's first incoming message of episode
                rewards(insertCol) += event.reward;
                effectiveDiscount(insertCol) = (event.isEndEpisode?0.0:discount);
                advanceInsertCol();
            }
        }


        template<class BODY>
        void on(const events::AgentEndEpisode<BODY> & /* event */) {
            if(actionIndices(insertCol) != unsetAct) { // if we sent last outgoing message.
                effectiveDiscount(insertCol) = 0.0;
                advanceInsertCol();
            }
        }

        template<class PARAMS>
        void on(const events::ParameterUpdate<PARAMS> & event) {
            if(++nParameterUpdates % endStateParameterUpdateInterval == 0) {
                endStatePredictor.parameters() = event.parameters;
            }
        }

        size_t batchSize() { return batchCols.n_rows; }
        size_t capacity() const { return stateHistory.n_cols; }
        size_t bufferSize() const { return bufferIsFull?capacity():insertCol; } // number of items in the buffer


        template<class INPUTS>
        void trainingSet(INPUTS &trainingPoints) {
            if(bufferIsFull) {
                batchCols = arma::randi<arma::uvec>(batchCols.n_rows, arma::distr_param(1, capacity() - 1));
                batchCols.transform([insertCol = insertCol, buffSize = capacity()](auto i) {
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
            for(int i=0; i<batchSize; ++i) endStateBatchCols(i) = (batchCols(i) + 1) % capacity();
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
//            std::cout << "Predictions:\n" << predictions << std::endl;
//            std::cout << "Start states:\n" << stateHistory.cols(batchCols) << std::endl;
//            std::cout << "Actions:\n" << actionIndices.cols(batchCols) << std::endl;
//            std::cout << "Rewards:\n" << rewards.cols(batchCols) << std::endl;
//            std::cout << "End States:\n" << batchEndStates << std::endl;
//            std::cout << "End State Q-vectors: \n" << batchEndStateQVectors << std::endl;
//            std::cout << "Effective discount:\n" << effectiveDiscount.cols(batchCols) << std::endl;
//            std::cout << "Training on gradient: \n" << gradient << std::endl;
            assert(!gradient.has_nan());
        }

    protected:
        void advanceInsertCol() {
            if(++insertCol >= capacity()) {
                bufferIsFull = true;
                insertCol = 0;
            }
            actionIndices(insertCol) = unsetAct;
        }

    };
}

#endif //MULTIAGENTGOVERNMENT_QLEARNINGLOSS_H
