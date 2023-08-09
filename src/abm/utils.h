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
    static size_t sampleUniformly(const MASK &legalMoves) {
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


    /** Takes a bit mask and turns it into a vector of indices of true bits
     *
     * @tparam MASK
     * @param legalActs
     * @return a vector containing the indices of each bit in legalActs that is true
     */
    template<DiscreteActionMask MASK>
    static std::vector<size_t> legalIndices(const MASK &legalActs) {
        std::vector<size_t> indices;
        for(int i=0; i<legalActs.size(); ++i)
            if(legalActs[i]) indices.push_back(i);
        return indices;
    }

}

#endif //MULTIAGENTGOVERNMENT_UTILS_H
