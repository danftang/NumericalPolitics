//
// Created by daniel on 06/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_UPPERCONFIDENCEPOLICY_H
#define MULTIAGENTGOVERNMENT_UPPERCONFIDENCEPOLICY_H

#include <bitset>
#include <cassert>
#include <ranges>
#include <cmath>

#include "../../Concepts.h"
#include "QVector.h"
#include "../../../DeselbyStd/random.h"

namespace abm::minds {

    /** Implements a version of UCT that makes use of the standard deviation of QValue samples
     * to provide better performance.
     * @tparam ACTION the action type to be returned from sample(). Should be convertible from an integer
     */
    template<class ACTION>
    class UpperConfidencePolicy {
    public:
//        bool i;
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
        template<size_t SIZE, IntegralActionMask MASK>
        ACTION sample(const QVector<SIZE> &qValues, const MASK &legalActs) {
            assert(legalActs.size() == SIZE);
            assert(legalActs.count() > 0);
            std::vector<size_t> legalActIndices = abm::legalIndices(legalActs);

            constexpr int minSamples = 1;
            auto undersampledActs = legalActIndices
                                 | std::ranges::views::filter([&qValues](size_t i) { return qValues[i].sampleCount < minSamples; });
            auto randomUnsampledActIt = deselby::random::uniformIterator(undersampledActs);
            if(randomUnsampledActIt != undersampledActs.end()) return static_cast<ACTION>(*randomUnsampledActIt);

            const double N = qValues.totalSamples();
            assert(N > 1);
            double nStandardErrors = sqrt(2.0*log(N));
            double minQ = std::numeric_limits<double>::infinity();
            double maxQ = -std::numeric_limits<double>::infinity();
            for (size_t actId : legalActIndices) {
                double q = qValues[actId].mean();
                if (q < minQ) minQ = q;
                if (q > maxQ) maxQ = q;
            }
            assert(minQ <= maxQ);
            double qScale = maxQ - minQ;
            size_t bestActId;
            double bestQ = -std::numeric_limits<double>::infinity();
            int nTies = 0; // number of tied bestQ states
            for (size_t actId : legalActIndices) {
                const QValue &qVal = qValues[actId];
//                double upperConfidenceQ = qVal.mean() + nStandardErrors * 3.0 * qVal.standardErrorOfMean();
//                double upperConfidenceQ = qVal.mean() + 16.0 * nStandardErrors/sqrt(qVal.sampleCount);
                double upperConfidenceQ = qVal.mean() + 2.0*qScale * nStandardErrors/sqrt(qVal.sampleCount);
                assert(!std::isnan(upperConfidenceQ));
                if (upperConfidenceQ > bestQ) {
                    bestQ = upperConfidenceQ;
                    bestActId = actId;
                    nTies = 0;
                } else if(upperConfidenceQ == bestQ) {
                    if(deselby::random::uniform<int, true>(0, ++nTies) == 0) bestActId = actId;
                }
            }
            assert(bestQ > -std::numeric_limits<double>::infinity()); // make sure we found an act
            return static_cast<action_type>(bestActId);
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_UPPERCONFIDENCEPOLICY_H
