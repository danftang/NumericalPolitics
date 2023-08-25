//
// Created by daniel on 03/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_ACTIONMASK_H
#define MULTIAGENTGOVERNMENT_ACTIONMASK_H

#include <concepts>

namespace abm {
    template<class MASK>
    concept ActionMask = requires(MASK actionMask, size_t action) {
        { actionMask.any() } -> std::convertible_to<bool>; // true if there are any legal actions
        { actionMask[action] } -> std::convertible_to<bool>;
    };

    /** Action masks are used to specify a set of legal actions from the domain of all actions
     * In the discrete case, we assume the domain is size_t
     */
    template<class T>
    concept DiscreteActionMask = ActionMask<T> && requires(T actionMask) {
        { actionMask.size()  } -> std::convertible_to<std::size_t>;
        { actionMask.count() } -> std::convertible_to<std::size_t>; // number of legal actions
    };

//    template<class ACTION, class MASK>
//    concept FitsMask = ActionMask<MASK,ACTION>;

    /** Samples uniformly from a discrete legal-action mask.
     * @param legalMoves the mask from which we wish to sample
     * @return a legal index into legalMoves chosen with uniform probability.
     */
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


    /** Takes a discrete legal-action mask and turns it into a vector of indices of true bits
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


#endif //MULTIAGENTGOVERNMENT_ACTIONMASK_H
