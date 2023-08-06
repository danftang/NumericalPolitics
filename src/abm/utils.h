//
// Created by daniel on 06/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_UTILS_H
#define MULTIAGENTGOVERNMENT_UTILS_H

#include <cassert>
#include "../DeselbyStd/random.h"
#include "ActionMask.h"

namespace abm {

    template<DiscreteActionMask MASK>
    static int sampleUniformly(const MASK &legalMoves) {
        size_t chosenMove = 0;
        size_t nLegalMoves = legalMoves.count();
        assert(nLegalMoves > 0);
        int legalMovesToGo = deselby::Random::nextInt(0, nLegalMoves);
        while(legalMovesToGo > 0 || legalMoves[chosenMove] == false) {
            if(legalMoves[chosenMove] == true) --legalMovesToGo;
            ++chosenMove;
            assert(chosenMove < legalMoves.size());
        }
        return chosenMove;
    }

}

#endif //MULTIAGENTGOVERNMENT_UTILS_H
