//
// Created by daniel on 08/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_ZEROINTELLIGENCE_H
#define MULTIAGENTGOVERNMENT_ZEROINTELLIGENCE_H

#include <cassert>
#include "../Concepts.h"
#include "../../DeselbyStd/random.h"

namespace abm::minds {
    class ZeroIntelligence {
    public:
        template<class BODY>
        size_t act(const BODY &body) {
            return sampleUniformly(body.legalActs());
        }


        /** Samples uniformly from a discrete legal-action mask.
         * @param legalMoves the mask from which we wish to sample
         * @return a legal index into legalMoves chosen with uniform probability.
         */
        template<IntegralActionMask MASK>
        static auto sampleUniformly(const MASK &legalMoves) {
            typedef decltype(legalMoves.size()) action_type;
//            action_type chosenMove = 0;
//            action_type nLegalMoves = legalMoves.count();
//            assert(nLegalMoves > 0);
//            action_type legalMovesToGo = deselby::random::uniform(nLegalMoves);
//            while(legalMovesToGo > 0 || legalMoves[chosenMove] == false) {
//                if(legalMoves[chosenMove] == true) --legalMovesToGo;
//                ++chosenMove;
//                assert(chosenMove < legalMoves.size());
//            }
            action_type chosenMove;
            size_t maxMoves = size_t(64)*legalMoves.size();
            std::uniform_int_distribution<action_type> randomElement(legalMoves.size());
            do { // rejection sampling has expected runtime of legalMoves.size()/legalMoves.count()
                chosenMove = randomElement(deselby::random::gen);
                assert(--maxMoves > 0);
            } while(legalMoves[chosenMove] == false);
            return chosenMove;
        }
    };
}


#endif //MULTIAGENTGOVERNMENT_ZEROINTELLIGENCE_H
