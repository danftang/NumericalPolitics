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
     * P(x_i) = e^(a.x_i)/(sum_j e^(a.x_j))
     * where 'a' is a parameter which should be in [0,infinity].
     *  a=0.0 gives the uniform distribution for all Q-Vectors,
     *  a=infinity gives the max function
     * by default a = 1.0
     */
    class SoftMaxPolicy {
    public:
        const double a;

        SoftMaxPolicy(double steepness = 1.0) : a(steepness) {
            assert(steepness > 0.0);
        }

        template<std::ranges::sized_range QVECTOR, IntegralActionMask ACTIONMASK>
        size_t sample(const QVECTOR &qValues, const ACTIONMASK &legalActs) {
            auto maxIt = std::ranges::max_element(qValues); // scale everything by e^-max to avoid overflow (so 0 < e^q <= 1)
            auto probs =
                    std::views::iota(0,qValues.size())
                    | std::views::transform([&qValues, &legalActs, a = a, qMax = *maxIt](auto &i) {
                        return legalActs[i]?exp(a*(qValues[i]-qMax)):0.0;
                    });
            return deselby::random::discrete(probs);
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_SOFTMAXPOLICY_H
