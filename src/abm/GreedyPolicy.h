// Wrapper class for policies that are no
//
// Created by daniel on 18/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_GREEDYPOLICY_H
#define MULTIAGENTGOVERNMENT_GREEDYPOLICY_H

#include <bitset>
#include <armadillo>
#include "../DeselbyStd/random.h"

namespace abm {

    template<class ACTION>
    class GreedyPolicy {
    public:

        typedef ACTION action_type;

        double pExplore;
        double exploreDeay;
        double pExploreMin;

        //  d^steps = (minimum/initialExp)^{1/steps}
        GreedyPolicy(double initialExploration, int stepsToDecayToMinimum, double minimumExploration) :
                pExplore(initialExploration),
                exploreDeay(pow(minimumExploration/initialExploration, 1.0/stepsToDecayToMinimum)),
//                exploreDeay((initialExploration - initialExploration)/stepsToDecayToMinimum),
                pExploreMin(minimumExploration) {}


//        template<unsigned long long MATSIZE, size_t BITSETSIZE> requires(MATSIZE == BITSETSIZE)
        ACTION sample(const arma::mat::fixed<action_type::size, 1> &qValues, const std::bitset<action_type::size> &legalMoves,
                   bool decayExplore = true) {
            assert(!qValues.has_nan());
//            std::cout << "sampling from " << qValues.t() << std::endl;
            int chosenMove = -1;
            assert(legalMoves.count() >= 1);
            if (deselby::Random::nextBool(pExplore)) {
                // choose a legal move at random
                chosenMove = sampleUniformly(legalMoves);
            } else {
                // choose the legal move with highest Q
                double bestQ = -std::numeric_limits<double>::infinity();
                for (int i = 0; i < legalMoves.size(); ++i) {
                    if (legalMoves[i] && qValues[i] > bestQ) {
                        bestQ = qValues[i];
                        chosenMove = i;
                    }
                }
                assert(bestQ != -std::numeric_limits<double>::infinity());
            }
            if (decayExplore) decayExploration();
//            std::cout << "Chose " << chosenMove << std::endl;
            return static_cast<ACTION>(chosenMove);
        }

        template<size_t SIZE>
        static int sampleUniformly(const std::bitset<SIZE> &legalMoves) {
            int chosenMove = -1;
            int nLegalMoves = legalMoves.count();
            assert(nLegalMoves > 0);
            int legalMovesToGo = deselby::Random::nextInt(1, nLegalMoves + 1);
            do {
                ++chosenMove;
                assert(chosenMove < SIZE);
                while (legalMoves[chosenMove] == false) { ++chosenMove; }
                --legalMovesToGo;
            } while (legalMovesToGo != 0);
            return chosenMove;
        }

        void decayExploration() {
            if (pExplore > pExploreMin) {
//                pExplore -= exploreDeay;
                pExplore *= exploreDeay;
//                std::cout << "pExplore = " << pExplore << std::endl;
            }
        }

    };
}

#endif //MULTIAGENTGOVERNMENT_GREEDYPOLICY_H
