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

namespace abm::minds {
    /** A QValue stores sum of samples and number of samples to give the mean over all samples */
    class QValue {
    public:
        double  sumOfQ      = 0.0; // Sum of Q-values of all samples so far by action
        uint    sampleCount = 0;   // number of samples by action

        void addSample(double cumulativeReward) {
            sumOfQ += cumulativeReward;
            ++sampleCount;
        }

        [[nodiscard]] double mean() const {
            assert(sampleCount > 0);
            return  sumOfQ / sampleCount;
        }

        operator double() const { return mean(); } // implicit conversion for use with policies that expect a single value

        bool operator <(const QValue &other) const {
            return mean() < other.mean();
        }

        friend std::ostream &operator <<(std::ostream &out, const QValue &qVal) {
            out << qVal.sampleCount << ": " << (qVal.sampleCount==0 ? 0.0 : qVal.mean());
            return out;
        }
    };


    /** Stores an exponentially weighted sum of the samples
      * We weight the samples exponentially with more recent samples having higher
      * weight.
      * So, after receiving n samples Q_1...Q_n, we have:
      * E_n[Q] = a_n.Q_n + a_n.r.Q_{n-1} + a_n.r^2.Q_{n-2} + ... + a_n.r^{n-1}.Q_1
      *
      * The sum of the weights should be 1 so
      * S_n = a_n(1-r^n)/(1-r) = 1
      * so
      * a_n = (1-r)/(1-r^n)
      * but we have the recurrence relation
      * E_n[Q] = (a_n.r/a_{n-1})E_{n-1}[Q] + a_n.Q_n
      * and, by expansion
      * a_n.r/a_{n-1} + a_n = (r-r^n)/(1-r^n) + (1-r)/(1-r^n) = 1
      * so
      * a_n.r/a_{n-1} = 1-a_n
      * and
      * E_n[Q] = (1-a_n)E_{n-1}[Q] + a_n.Q_n
      *
      */
    template<deselby::ConstExpr<double> sampleDecay>
    class ExponentiallyWeightedQValue {
    public:
        double  qValue      = 0.0;  // weighted average of samples
        uint    sampleCount = 0;    // number of samples


        void addSample(double cumulativeReward) {
            const double a_n = (1.0-sampleDecay)/(1.0-std::pow(sampleDecay, ++sampleCount));
            qValue  = (1.0-a_n) * qValue + a_n * cumulativeReward;
        }

        operator double() const { return qValue; } // implicit conversion for use with policies that expect a single value

        bool operator <(const ExponentiallyWeightedQValue &other) const {
            return qValue < other.qValue;
        }

        friend std::ostream &operator <<(std::ostream &out, const ExponentiallyWeightedQValue<sampleDecay> &qVal) {
            out << qVal.sampleCount << ": " << qVal.qValue;
            return out;
        }
    };

    /** Allows calculation of mean and standard error of the samples */
    class QValueWithVariance {
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

        friend std::ostream &operator <<(std::ostream &out, const QValueWithVariance &qVal) {
            out << qVal.sampleCount << ": "
            << (qVal.sampleCount == 0 ? 0.0 : qVal.mean())
            << "+-" << (qVal.sampleCount < 2 ? 0.0 : qVal.standardErrorOfMean());
            return out;
        }
    };


    /** A QVector is a set of QValues for all acts in a single state */
    template<size_t SIZE, class QVALUE = QValue>
    class QVector: public std::array<QVALUE, SIZE> {
    public:
        int totalSamples() const {
            int sum = 0;
            for(const QValue &val : *this) sum += val.sampleCount;
            return sum;
        }

        std::array<QVALUE,SIZE> &asArray() { return *this; }
//        arma::mat::fixed<SIZE,1> toVector() {
//            arma::mat::fixed<SIZE,1> Qvec;
//            for(int i=0; i<SIZE; ++i) Qvec(i) = (*this)[i].mean();
//            return Qvec;
//        }
//
//        operator arma::mat::fixed<SIZE,1>() {
//            return toVector();
//        }
    };
}


#endif //MULTIAGENTGOVERNMENT_QVECTOR_H
