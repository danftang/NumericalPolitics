//
// Created by daniel on 08/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_ZEROINTELLIGENCE_H
#define MULTIAGENTGOVERNMENT_ZEROINTELLIGENCE_H

#include <cassert>
#include "../ActionMask.h"

namespace abm::minds {
    class ZeroIntelligence {
    public:
        template<class BODY>
        int operator()(const BODY &body) {
            return sampleUniformly(body.legalActs());
        }

        /** Samples uniformly from a discrete legal-action mask.
         * @param legalMoves the mask from which we wish to sample
         * @return a legal index into legalMoves chosen with uniform probability.
         */
        template<DiscreteActionMask MASK>
        static int sampleUniformly(const MASK &legalMoves) {
            int chosenMove = 0;
            int nLegalMoves = legalMoves.count();
            assert(nLegalMoves > 0);
            int legalMovesToGo = deselby::Random::nextInt(0, nLegalMoves);
            while(legalMovesToGo > 0 || legalMoves[chosenMove] == false) {
                if(legalMoves[chosenMove] == true) --legalMovesToGo;
                ++chosenMove;
                assert(chosenMove < legalMoves.size());
            }
            return chosenMove;
        }
    };
}


#endif //MULTIAGENTGOVERNMENT_ZEROINTELLIGENCE_H
