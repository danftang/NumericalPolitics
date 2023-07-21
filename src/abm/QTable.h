//
// Created by daniel on 21/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_QTABLE_H
#define MULTIAGENTGOVERNMENT_QTABLE_H

#include <array>
#include "../DeselbyStd/random.h"

namespace abm {
    template<int NSTATES, int NACTIONS>
    class QTable {
    public:
        static constexpr double DEFAULT_DISCOUNT = 1.0;
        static constexpr double DEFAULT_LEARNING_RATE = 0.001;

        typedef int input_type;

        inline static bool verbose = false;

        std::array<arma::colvec, NSTATES>   Qtable;
        std::array<std::array<uint, NACTIONS>, NSTATES>      nSamples;
        double discount;     // exponential decay factor of future reward
        double learningRate;
//        uint trainingStepsSinceLastPolicyChange = 0;

        QTable(
                double discount = DEFAULT_DISCOUNT,
                double learningRate = DEFAULT_LEARNING_RATE
        ) :
                discount(discount), learningRate(learningRate) {

            // initialise Q values and set random bestAction
            for (int state = 0; state < NSTATES; ++state) {
                Qtable[state].zeros(NACTIONS);
                for (int action = 0; action < NACTIONS; ++action) {
                    nSamples[state][action] = 0;
                }
            }
        }



        // Single training step for one action.
        // At equilibrium we have
        // Q(s0,a) = reward(s0,a,s1) + discount * max_a'(Q(s1,a'))
        // so we relax the table to equilibrium by setting
        // Q(s0,a) <- (1-l)Q(s0,a) + l*(reward(s0,a,s1) + discount * max_a'(Q(s1,a')))
        //
        // If we weight the samples with the weights:
        // a_n, a_n r, a_n r^2, a_n r^3 ... a_n r^(n-1)
        //
        // The sum of the weights must be 1 so
        // S_n = a_n(1-r^n)/(1-r) = 1
        // so
        // a_n = (1-r)/(1-r^n)
        // so
        // a_{n+1} = a_n(1-r^n)/(1-r^{n+1})
        //
        // so we multiply the current Q value by (r-r^{n+1})/(1-r^{n+1}) and weight the current sample by (1-r)/(1-r^{n+1})
        //
        // But 1-((1-r)/(1-r^{n+1})) = (r-r^{n+1})/(1-r^{n+1})
        void train(int startState, int action, double reward, int endState, bool isEndgame) {
//            std::cout << "training on " << startState << " " << action << " " << reward << " " << endState << std::endl;

            ++nSamples[startState][action];
            const double rrn = std::pow(1.0 - learningRate,nSamples[startState][action]);
            double sampleWeight = learningRate/(1.0-rrn);

            const double endStateQValue = (isEndgame ? 0.0 : Qtable[endState].max());
            const double forwardQ = reward + discount * endStateQValue;
            Qtable[startState][action] = (1.0-sampleWeight) * Qtable[startState][action] + sampleWeight * forwardQ;
        }

        const arma::colvec &predict(int state) {
            return Qtable[state];
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_QTABLE_H
