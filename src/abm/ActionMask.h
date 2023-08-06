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

    /** Action masks are used to specify a set of legal actions from the domain of all actions
     * In the discrete case, we assume the domain is size_t
     */
    template<class T>
    concept DiscreteActionMask = ActionMask<T,size_t> && requires(T actionMask) {
        { actionMask.size()  } -> std::convertible_to<std::size_t>;
        { actionMask.count() } -> std::convertible_to<std::size_t>; // number of legal actions
    };

    template<class ACTION, class MASK>
    concept FitsMask = ActionMask<MASK,ACTION>;

}


#endif //MULTIAGENTGOVERNMENT_ACTIONMASK_H
