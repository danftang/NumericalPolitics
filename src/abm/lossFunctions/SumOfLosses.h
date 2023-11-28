//
// Created by daniel on 16/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_SUMOFLOSSES_H
#define MULTIAGENTGOVERNMENT_SUMOFLOSSES_H

#include <tuple>
#include <armadillo>
#include "../../DeselbyStd/tupleutils.h"
#include "../CallbackUtils.h"
#include "../Concepts.h"

namespace abm::lossFunctions {

    /** A loss function which has the form of a sum of loss functions  */
    template<LossFunction...LOSSES>
    class SumOfLosses {
    public:
        std::tuple<LOSSES...> losses;

        SumOfLosses(LOSSES...losses) : losses(std::move(losses)...) {}

        template<class EVENT> requires (HasCallback<LOSSES,EVENT> || ...)
        void on(const EVENT &event) {
//            std::cout << "Intercepting event via SumOfLosses" << std::endl;
            callback(event, losses);
        }

        size_t batchSize() {
            return std::apply([](auto... losses) { return (losses.batchSize() + ...); }, losses);
        }

        template<class INPUT>
        void trainingSet(INPUT &&input) { // possibly different every call, in the case of stochastic loss
            size_t col = 0;
            deselby::for_each(losses, [&col, &input](auto &loss) {
                size_t startCol = col;
                auto batchSize = loss.batchSize();
                if(batchSize > 0) {
                    col += batchSize;
                    auto subMat = input.cols(startCol, col-1);
                    loss.trainingSet(subMat);
                }
            });
        }

        template<class OUTPUT, class GRAD>
        void gradientByPrediction(const OUTPUT &outputs, GRAD &&grad) {
            size_t col = 0;
            deselby::for_each(losses, [&col, &outputs, &grad](auto &loss) {
                size_t startCol = col;
                auto batchSize = loss.batchSize();
                if(batchSize > 0) {
                    col += batchSize - 1;
                    auto subMat = grad.cols(startCol, col);
                    loss.gradientByPrediction(outputs.cols(startCol, col), subMat);
                    ++col;
                }
            });
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_SUMOFLOSSES_H
