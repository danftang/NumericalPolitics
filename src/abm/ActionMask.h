//
// Created by daniel on 03/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_ACTIONMASK_H
#define MULTIAGENTGOVERNMENT_ACTIONMASK_H

#include <concepts>

namespace abm {
    template<class MASK, class ACTION>
    concept ActionMask = requires(MASK actionMask, ACTION action) {
        { actionMask.any() } -> std::convertible_to<bool>; // true if there are any legal actions
        { actionMask[action] } -> std::convertible_to<bool>;
    };

    template<class T, class ACTION>
    concept DiscreteActionMask = ActionMask<T,ACTION> && requires(T actionMask, ACTION action) {
        { actionMask.count() } -> std::convertible_to<std::size_t>; // number of legal actions (what about continuous?)
    };

    template<class ACTION, class MASK>
    concept FitsMask = ActionMask<MASK,ACTION>;

}


#endif //MULTIAGENTGOVERNMENT_ACTIONMASK_H
