//
// Created by daniel on 06/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_UPPERCONFIDENCEPOLICY_H
#define MULTIAGENTGOVERNMENT_UPPERCONFIDENCEPOLICY_H

#include "ActionMask.h"
#include "QVector.h"
#include <bitset>
#include <cassert>

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
        template<size_t SIZE, class MASK> requires(MASK::size() == SIZE)
        ACTION sample(const QVector<SIZE> &qValues, const MASK &legalActs) {
            double nStandardErrors = sqrt(log(qValues.totalSamples()));
            int bestActId;
            double bestQ = -std::numeric_limits<double>::infinity();
            for (int actId = 0; actId < SIZE; ++actId) {
                if(legalActs[actId]) {
                    const QValue &qVal = (*this)[actId];
                    if (qVal.sampleCount == 0) {
                        bestActId = actId;
                        break; // if there is an unexplored option, always explore.
                    }
                    double upperConfidenceQ = qVal.mean() + nStandardErrors * sqrt(qVal.standardVarianceInMean());
                    if (upperConfidenceQ >= bestQ) {
                        bestQ = upperConfidenceQ;
                        bestActId = actId;
                    }
                }
            }
            assert(bestQ > -std::numeric_limits<double>::infinity()); // make sure we found an act
            return static_cast<action_type>(bestActId);
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_UPPERCONFIDENCEPOLICY_H
