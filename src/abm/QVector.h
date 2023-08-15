//
// Created by daniel on 06/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_QVECTOR_H
#define MULTIAGENTGOVERNMENT_QVECTOR_H

#include <cstdlib>
#include <cmath>
#include <array>
#include <numeric>
#include <iostream>
#include <cassert>

namespace abm {
    /** A QValue stores mean, sample size and standard deviation of the Q-Value of a single (state,action) pair */
    class QValue {
    public:
        double  sumOfQ      = 0.0; // Sum of Q-values of all samples so far by action
        double  sumOfQSq    = 0.0; // Sum of squares of Q-values of all samples so far by action
        uint    sampleCount = 0;   // number of samples by action

        void addSample(double cumulativeReward) {
            sumOfQ += cumulativeReward;
            sumOfQSq += cumulativeReward * cumulativeReward;
            ++sampleCount;
        }

        [[nodiscard]] double mean() const {
            assert(sampleCount > 0);
            return  sumOfQ / sampleCount;
        }

        [[nodiscard]] double standardErrorOfMean() const {
            assert(sampleCount > 1);
            const double sigmasq = variance();
            assert(sigmasq > -1e-8);
            if(sigmasq <= 0.0) return 0.0; // rounding error
            return sqrt(sigmasq/sampleCount);
        }

        double variance() const {
            return sumOfQSq/(sampleCount-1) - (sumOfQ/sampleCount)*(sumOfQ/(sampleCount-1));
        }

        operator double() const { return mean(); } // implicit conversion for use with policies that expect a single value

        friend std::ostream &operator <<(std::ostream &out, const QValue &qVal) {
            out << qVal.sampleCount << ": ";
            if(qVal.sampleCount < 2) {
                out << "##";
            } else {
                out << qVal.mean() << "+-" << qVal.standardErrorOfMean();
            }
            return out;
        }
    };


    /** A QVector is a set of QValues for all acts in a single state */
    template<size_t SIZE>
    class QVector: public std::array<QValue, SIZE> {
    public:
        int totalSamples() const {
            int sum = 0;
            for(const QValue &val : *this) sum += val.sampleCount;
            return sum;
        }
    };

}


#endif //MULTIAGENTGOVERNMENT_QVECTOR_H
