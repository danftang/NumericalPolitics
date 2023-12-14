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

        template<GenericQVector QVECTOR, IntegralActionMask ACTIONMASK>
        uint sample(const QVECTOR &qValues, const ACTIONMASK &legalActs) {
            return std::discrete_distribution<uint>(legalActs.size(), 0.0, legalActs.size(),
                    [&legalActs, a=a, &qValues, maxQ = maxQVal(qValues,legalActs)](double x) {
                       uint i = x;
                        return legalActs[i]?exp(a*(qValues[i] - maxQ)):0.0;
                    })(deselby::random::gen);
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
        template<GenericQVector QVECTOR, IntegralActionMask ACTIONMASK>
        arma::vec gradient(const QVECTOR &qVec, const ACTIONMASK &legalActs, uint action) {
            arma::vec dP_dqa(legalActs.size());
            std::vector<double> P = pmf(qVec, legalActs);
            for(uint i=0; i<legalActs.size(); ++i) {
                    dP_dqa(i) = -P[action]*P[i];
            }
            dP_dqa(action) += P[action];
            dP_dqa *= a;
            return dP_dqa;
        }


        /** The probability mass function over actions given a Q-Vector
         * pmf_i = e^{aQ_i}/sum_j e^{aQ_j}
         * @param qVec the point at which to evaluate the PMF
         * @return A PMF over actions given a Q-Vector
         */
        template<GenericQVector QVECTOR, IntegralActionMask ACTIONMASK>
        std::vector<double> pmf(const QVECTOR &qVec, const ACTIONMASK &legalActs) {
            uint size = legalActs.size();
            std::vector<double> distribution(size);
            double maxQ = maxQVal(qVec, legalActs); // scale everything by e^-max to avoid overflow (so 0 < e^q <= 1)
            double sumOfWeights = 0.0;
            for(uint i=0; i<size; ++i) {
                double weight = legalActs[i]?exp(a*(qVec[i]-maxQ)):0.0;
                distribution[i] = weight;
                sumOfWeights += weight;
            }
            for(uint i=0; i<size; ++i) distribution[i] /= sumOfWeights;
            return distribution;
        }

        /** The probability of a given action, given a Q-Vector and action mask
         */
        template<GenericQVector QVECTOR, IntegralActionMask ACTIONMASK>
        double probability(const QVECTOR &qVec, const ACTIONMASK &legalActs, uint action) {
            if(!legalActs[action]) return 0.0;
            double maxQ = maxQVal(qVec, legalActs);
            double sumOfExps = 0.0;
            double exp_a = 0.0;
            for(uint i=0; i<legalActs.size(); ++i) {
                if(legalActs[i]) {
                    double e = exp(a * (qVec[i] - maxQ));
                    sumOfExps += e;
                    if (i == action) exp_a = e;
                }
            }
            assert(exp_a != 0.0); // not technically an error, but for now detects QValues getting too big
            return exp_a/sumOfExps;
        }


        template<GenericQVector QVECTOR, IntegralActionMask ACTIONMASK>
        static auto maxQVal(const QVECTOR &qVec, const ACTIONMASK &legalActs) {
            assert(legalActs.size() > 0);
            auto maxQ = qVec[0u];
            for(uint i = 1; i<legalActs.size(); ++i) {
                if(legalActs[i] && maxQ < qVec[i]) maxQ = qVec[i];
            }
            return maxQ;
        }

    };
}

#endif //MULTIAGENTGOVERNMENT_SOFTMAXPOLICY_H
