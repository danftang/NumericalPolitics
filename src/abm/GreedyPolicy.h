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
    class GreedyPolicy {
    public:
        double pExplore;
        double exploreDeay;
        double pExploreMin;

        GreedyPolicy(double initialExploration, double explorationDecay, double minimumExploration) :
                pExplore(initialExploration),
                exploreDeay(explorationDecay),
                pExploreMin(minimumExploration) {}


        template<unsigned long long MATSIZE, size_t BITSETSIZE> requires(MATSIZE == BITSETSIZE)
        int sample(const arma::mat::fixed<MATSIZE, 1> &qValues, const std::bitset<BITSETSIZE> &legalMoves,
                   bool decayExplore = true) {
            int chosenMove = -1;
            assert(legalMoves.count() > 1);
            if (deselby::Random::nextBool(pExplore)) {
                // choose a legal move at random
                int nLegalMoves = legalMoves.count();
                assert(nLegalMoves > 0);
                int legalMovesToGo = deselby::Random::nextInt(1, nLegalMoves + 1);
                do {
                    ++chosenMove;
                    while (legalMoves[chosenMove] == false) { ++chosenMove; }
                    --legalMovesToGo;
                } while (legalMovesToGo != 0);
            } else {
                // choose the legal move with highest Q
                double bestQ = -std::numeric_limits<double>::infinity();
                for (int i = 0; i < legalMoves.size(); ++i) {
                    if (legalMoves[i] && qValues[i] > bestQ) {
                        bestQ = qValues[i];
                        chosenMove = i;
                    }
                }
            }
            if (decayExplore) decayExploration();
            return chosenMove;
        }

        void decayExploration() {
            if (pExplore > pExploreMin) {
                pExplore *= exploreDeay;
            }
        }

    };
}

#endif //MULTIAGENTGOVERNMENT_GREEDYPOLICY_H
