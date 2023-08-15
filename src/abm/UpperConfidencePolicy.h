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
        bool i;
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
            std::vector<size_t> legalActIndices = abm::legalIndices(legalActs);

            constexpr int minSamples = 10;
            auto undersampledActs = legalActIndices
                                 | std::ranges::views::filter([&qValues](size_t i) { return qValues[i].sampleCount < minSamples; });
            auto randomUnsampledActIt = deselby::Random::chooseElement(undersampledActs);
            if(randomUnsampledActIt != undersampledActs.end()) return static_cast<ACTION>(*randomUnsampledActIt);

//            auto unsampledActs = legalActIndices
//                                 | std::ranges::views::filter([&qValues](size_t i) { return qValues[i].sampleCount == 0; });
//            auto randomUnsampledActIt = deselby::Random::chooseElement(unsampledActs);
//            if(randomUnsampledActIt != unsampledActs.end()) return static_cast<ACTION>(*randomUnsampledActIt);
//
//            auto onceSampledActs = legalActIndices
//                                   | std::ranges::views::filter([&qValues](size_t i) { return qValues[i].sampleCount == 1; });
//            auto randomOnceSampledActIt = deselby::Random::chooseElement(onceSampledActs);
//            if(randomOnceSampledActIt != onceSampledActs.end()) return static_cast<ACTION>(*randomOnceSampledActIt);

            const double N = qValues.totalSamples();
            assert(N > 1);
            double nStandardErrors = 2.0*sqrt(log(N));
            size_t bestActId;
            double bestQ = -std::numeric_limits<double>::infinity();
            for (size_t actId : legalActIndices) {
                const QValue &qVal = qValues[actId];
                double upperConfidenceQ = qVal.mean() + nStandardErrors * qVal.standardErrorOfMean();
//                double upperConfidenceQ = qVal.mean() + nStandardErrors * 16.0/sqrt(qVal.sampleCount);
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
