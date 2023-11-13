//
// Created by daniel on 03/11/23.
//

#ifndef MULTIAGENTGOVERNMENT_SOFTMAXPOLICY_H
#define MULTIAGENTGOVERNMENT_SOFTMAXPOLICY_H

#include <ranges>

#include "../../Concepts.h"
#include "../../../DeselbyStd/random.h"

namespace abm::minds {

    /** Given a vector (x_1....x_n), SoftMax is defined as
     * P(x_i) = e^(x_i)/(sum_j e^(x_j))
     */
    class SoftMaxPolicy {
    public:
        template<std::ranges::sized_range QVECTOR, IntegralActionMask ACTIONMASK>
        size_t sample(const QVECTOR &qValues, const ACTIONMASK &legalActs) {
            auto maxIt = std::ranges::max_element(qValues); // scale everything by e^-max to avoid overflow (so 0 < e^q <= 1)
            auto probs =
                    std::views::iota(0,qValues.size())
                    | std::views::transform([&qValues, &legalActs, qMax = *maxIt](auto &i) {
                        return legalActs[i]?exp(qValues[i]-qMax):0.0;
                    });
            return deselby::random::discrete(probs);
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_SOFTMAXPOLICY_H
