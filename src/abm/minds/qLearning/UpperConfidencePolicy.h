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

    /** Implements UCT as described in
     *  Kocsis, Szepesvári 2006: Bandit Based Monte-Carlo Planning. in ECML 2006, LNAI 4212, pp. 282–293.
     *
     * Given a QVector with means q_i and sample counts n_i, sample(.) chooses the action, i, that maximises
     * q_i + sqrt(2ln(\sum_j n_j) / n_i)
     *
     * Note that the range of possible Qvalues, i.e. q_max - q_min should be around 1.0. The absolute value
     * can be anything.
     *
     * @tparam ACTION the action type to be returned from sample(). Should be convertible from an integer.
     */
    template<class ACTION>
    class UpperConfidencePolicy {
    public:
        typedef ACTION action_type;

        /**
         * Calculates the best action according to UCT. i.e. given a QVector with means q_i and sample counts n_i
         * choose the action, i, that maximises
         *   q_i + sqrt(2ln(\sum_j n_j) / n_i)
         *
         * Note that the range of possible Qvalues, i.e. q_max - q_min should be around 1.0. The absolute value
         * can be anything.
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
            size_t bestActId;
            double bestQ = -std::numeric_limits<double>::infinity();
            int nTies = 0; // number of tied bestQ states
            for (size_t actId : legalActIndices) {
                const QValue &qVal = qValues[actId];
//                double upperConfidenceQ = qVal.mean() + nStandardErrors * 3.0 * qVal.standardErrorOfMean();
                double upperConfidenceQ = qVal.mean() + nStandardErrors/sqrt(qVal.sampleCount);
//                double upperConfidenceQ = qVal.mean() + 8.0*qScale * nStandardErrors/sqrt(qVal.sampleCount);
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
