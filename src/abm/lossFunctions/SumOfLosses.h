//
// Created by daniel on 16/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_SUMOFLOSSES_H
#define MULTIAGENTGOVERNMENT_SUMOFLOSSES_H

#include <tuple>
#include <armadillo>
#include "../../DeselbyStd/tupleutils.h"
#include "../CallbackUtils.h"

namespace abm::lossFunctions {

    /** A loss function which has the form of a sum of loss functions  */
    template<class...LOSSES>
    class SumOfLosses {
    public:
        std::tuple<LOSSES...> losses;

        SumOfLosses(LOSSES...losses) : losses(losses...) {}

        template<class EVENT> requires (HasCallback<LOSSES,EVENT> || ...)
        void on(const EVENT &event) {
            callback(event, losses);
        }

        size_t nPoints() {
            return std::apply([](auto... losses) { (losses.nPoints() + ...); }, losses);
        }

        template<class INPUT>
        void trainingSet(INPUT &input) { // possibly different every call, in the case of stochastic loss
            size_t col = 0;
            deselby::for_each(losses, [&col, &input](auto &loss) {
                loss.input(input.cols(col, col += loss.nPoints()));
            });
        }

        template<class OUTPUT, class GRAD>
        void gradientByPrediction(const OUTPUT &outputs, GRAD &grad) {
            size_t col = 0;
            deselby::for_each(losses, [&col, &outputs, &grad](auto &loss) {
                size_t endCol = col + loss.nPoints();
                loss.gradientByPrediction(outputs.cols(col, endCol), grad.cols(col, endCol));
                col = endCol;
            });
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_SUMOFLOSSES_H
