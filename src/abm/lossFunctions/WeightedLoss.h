//
// Created by daniel on 31/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_WEIGHTEDLOSS_H
#define MULTIAGENTGOVERNMENT_WEIGHTEDLOSS_H

#include <utility>
#include "../Concepts.h"

namespace abm::lossFunctions {
    /** Helper class to generate a loss that is another loss multiplied by a constant weight */
    template<LossFunction BASELOSS>
    class WeightedLoss : public BASELOSS {
    public:
        const double weight;

        WeightedLoss(double weight, BASELOSS baseloss) : BASELOSS(std::move(baseloss)), weight(weight) {
        }

        template<class OUTPUTS, class RESULT>
        void gradientByPrediction(const OUTPUTS &prediction, RESULT &gradient) {
            this->BASELOSS::gradientByPrediction(prediction, gradient);
            gradient *= weight;
        }


    };
}


#endif //MULTIAGENTGOVERNMENT_WEIGHTEDLOSS_H
