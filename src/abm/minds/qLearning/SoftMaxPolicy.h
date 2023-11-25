//
// Created by daniel on 03/11/23.
//

#ifndef MULTIAGENTGOVERNMENT_SOFTMAXPOLICY_H
#define MULTIAGENTGOVERNMENT_SOFTMAXPOLICY_H

#include <ranges>

#include "../../Concepts.h"
#include "../../../DeselbyStd/random.h"

namespace abm::minds {

    /** Given a Q-vector Q=(q_1....q_n), SoftMax is defined as
     *
     * P(Q)_i = e^(a.q_i)/(sum_j e^(a.q_j))
     *
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
        uint sample(const QVECTOR &qValues, const ACTIONMASK &legalActs) {
            return pmf(qValues, legalActs)(deselby::random::gen);
        }


        /** The rate of change of action probability with respect to an element of the Q-vector.
         *
         * dP(Q)_{i \ne j}/dq_j = -ae^{a.(q_i + q_j)}/A^2 = -aP(Q)_iP(Q)_j
         * and
         * dP(Q)_j/dq_j = a(e^{a.q_j}(1 - P(q_j)))/A = aP(Q)_j(1-P(Q)_j) = aP(Q)_j - aP(Q)_jP(Q)_j
         *
         * @param qVec      the Q-vector at which to evaluate the gradient
         * @param action    the element of the qVec that we're varying
         * @return d(pmf(qVec))/dqVec(a)
         */
        template<std::ranges::sized_range QVECTOR, IntegralActionMask ACTIONMASK>
        arma::vec gradient(const QVECTOR &qVec, const ACTIONMASK &legalActs, uint action) {
            arma::vec dP_dqa(qVec.size());
            std::vector<double> P = pmf(qVec, legalActs).probabilities();
            for(uint i=0; i<qVec.size(); ++i) {
                dP_dqa(i) = -P[action]*P[i];
            }
            dP_dqa(action) += P[action];
            dP_dqa *= a;
            return dP_dqa;
        }


        /** The probability mass function over actions given a Q-Vector
         * @param qVec the point at which to evaluate the PMF
         * @return A PMF over actions given a Q-Vector
         */
        template<std::ranges::sized_range QVECTOR, IntegralActionMask ACTIONMASK>
        std::discrete_distribution<uint> pmf(const QVECTOR &qValues, const ACTIONMASK &legalActs) {
            auto maxIt = std::ranges::max_element(qValues); // scale everything by e^-max to avoid overflow (so 0 < e^q <= 1)
            return std::views::iota(0,qValues.size())
                   | std::views::transform([&qValues, &legalActs, a = a, qMax = *maxIt](auto &i) {
                return legalActs[i]?exp(a*(qValues[i]-qMax)):0.0;
            });
        }

        /** The probability of a given action, given a Q-Vector and action mask
         */
        template<std::ranges::sized_range QVECTOR, IntegralActionMask ACTIONMASK>
        double probability(const QVECTOR &qVec, const ACTIONMASK &legalActs, uint action) {
            auto maxIt = std::ranges::max_element(qVec); // scale everything by e^-max to avoid overflow (so 0 < e^q <= 1)
            assert(maxIt != qVec.end());
            double maxQ = *maxIt;
            uint i=0;
            double sumOfExps = 0.0;
            double exp_a;
            for(const auto &qVal : qVec) {
                double e = exp(a*(qVal - maxQ));
                sumOfExps += e;
                if(i == action) exp_a = e;
                ++i;
            }
            return exp_a/sumOfExps;
        }

    };
}

#endif //MULTIAGENTGOVERNMENT_SOFTMAXPOLICY_H
