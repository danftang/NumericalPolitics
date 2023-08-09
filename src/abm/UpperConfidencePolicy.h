//
// Created by daniel on 06/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_UPPERCONFIDENCEPOLICY_H
#define MULTIAGENTGOVERNMENT_UPPERCONFIDENCEPOLICY_H

#include "ActionMask.h"
#include "QVector.h"
#include <bitset>
#include <cassert>
#include <ranges>

namespace abm {

    /** Implements a version of UCT that makes use of the standard deviation of QValue samples
     * to provide better performance.
     * @tparam ACTION the action type to be returned from sample(). Should be convertible from an integer
     */
    template<class ACTION>
    class UpperConfidencePolicy {
    public:
        typedef ACTION action_type;

        /**
         * Calculates the best action according to UCT. i.e. the action with maximum
         * meanQ + S*sqrt(ln(N))
         * where
         *  meanQ = the mean of all samples
         *  S = standard error in the mean
         *  N = the total number of samples
         *
         * @return the chosen act
         */
        template<size_t SIZE, DiscreteActionMask MASK>
        ACTION sample(const QVector<SIZE> &qValues, const MASK &legalActs) {
            assert(legalActs.size() == SIZE);
            assert(legalActs.count() > 0);
            double bestQ = -std::numeric_limits<double>::infinity();
            std::vector<size_t> actIndices = abm::legalIndices(legalActs);

            auto unsampledActs = actIndices
                    | std::ranges::views::filter([&qValues](size_t i) { return qValues[i].sampleCount == 0; });
            auto randomUnsampledActIt = deselby::Random::chooseElement(unsampledActs);
            if(randomUnsampledActIt != unsampledActs.end()) return static_cast<ACTION>(*randomUnsampledActIt);

            auto onceSampledActs = actIndices
                    | std::ranges::views::filter([&qValues](size_t i) { return qValues[i].sampleCount == 1; });
            auto randomOnceSampledActIt = deselby::Random::chooseElement(onceSampledActs);
            if(randomOnceSampledActIt != onceSampledActs.end()) return static_cast<ACTION>(*randomOnceSampledActIt);

            double nStandardErrors = sqrt(log(qValues.totalSamples()));
            int bestActId;
            for (size_t actId : actIndices) {
                const QValue &qVal = qValues[actId];
                double upperConfidenceQ = qVal.mean() + nStandardErrors * qVal.standardErrorOfMean();
                assert(!isnan(upperConfidenceQ));
                if (upperConfidenceQ >= bestQ) {
                    bestQ = upperConfidenceQ;
                    bestActId = actId;
                }
            }
            assert(bestQ > -std::numeric_limits<double>::infinity()); // make sure we found an act
            return static_cast<action_type>(bestActId);
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_UPPERCONFIDENCEPOLICY_H
