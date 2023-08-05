//
// Created by daniel on 03/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_MIND_H
#define MULTIAGENTGOVERNMENT_MIND_H

#include <concepts>
#include "ActionMask.h"

namespace abm {
    template<class T>
    concept Mind = requires(T mind, T::observation_type observation, T::action_type act, T::action_mask actMask, T::reward_type reward) {
        typename T::observation_type;
        // typename T::action_type;
        typename T::action_mask;
        typename T::reward_type;
        { mind.act(observation, actMask, reward) };
        { mind.endEpisode(reward) } -> std::same_as<void>;
    };
}

#endif //MULTIAGENTGOVERNMENT_MIND_H
