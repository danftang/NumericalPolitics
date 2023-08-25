//
// Created by daniel on 21/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_QTABLE_H
#define MULTIAGENTGOVERNMENT_QTABLE_H

#include <array>
#include <armadillo>
#include "../DeselbyStd/random.h"
#include "../DeselbyStd/BoundedInteger.h"

namespace abm {
    template<int NSTATES, int NACTIONS>
    class QTable {
    public:
        static constexpr int output_dimension = NACTIONS;

        typedef int                                 input_type;
        typedef const arma::mat::fixed<NACTIONS,1> &output_type;
        typedef deselby::BoundedInteger<int,0,NACTIONS-1> action_type;

        inline static bool verbose = false;

        std::array<arma::mat::fixed<NACTIONS,1>, NSTATES>   Qtable;
        std::array<std::array<uint, NACTIONS>, NSTATES>      nSamples;
        double discount;     // exponential decay factor of future reward
        double sampleDecay;
//        uint trainingStepsSinceLastPolicyChange = 0;

        /**
         *
         * @param discount
         * @param learningRate      The weight of temporal difference error when updating Q-values (see train)
         */
        QTable(
                double discount,
                double sampleDecay = 0.999
        ) :
                discount(discount), sampleDecay(sampleDecay) {

            // initialise Q values and set random bestAction
            for (int state = 0; state < NSTATES; ++state) {
                Qtable[state].zeros();
                for (int action = 0; action < NACTIONS; ++action) {
                    nSamples[state][action] = 0;
                }
            }
        }


        /**
         * Single training step for one action.
         *
         * At equilibrium we have
         * Q(s0,a) = E[reward(s0,a,s1) + discount * max_a'(Q(s1,a'))]
         *
         * We consider the expectation above as being over the weighted mean where we
         * weight the samples exponentially with more recent samples having higher
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
         * @param startState
         * @param action
         * @param reward
         * @param endState
         * @param isEndgame
         */
        void train(int startState, action_type action, double reward, int endState, bool isEndgame) {
//            std::cout << "training " << this << " on " << std::oct << startState << " " << action << " " << reward << " " << std::oct << endState << " " << isEndgame << std::endl;
            const double a_n            = (1.0-sampleDecay)/(1.0-std::pow(sampleDecay, ++nSamples[startState][action]));
            const double endStateQValue = (isEndgame ? 0.0 : Qtable[endState].max());
            const double forwardQ       = reward + discount * endStateQValue;
            Qtable[startState][action]  = (1.0-a_n) * Qtable[startState][action] + a_n * forwardQ;
            assert(!Qtable[startState].has_nan());
        }

        output_type predict(int state) {
            assert(!Qtable[state].has_nan());
            return Qtable[state];
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_QTABLE_H
