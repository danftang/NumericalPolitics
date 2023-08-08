//
// Created by daniel on 03/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_MIND_H
#define MULTIAGENTGOVERNMENT_MIND_H

#include <concepts>
#include "ActionMask.h"


// TODO: add hooks to Mind so that we can see messages coming in and out.
//  Also endEpisode can become a hook, to deal with the case where we know the reward
namespace abm {
    template<class T>
    concept Mind = requires(T mind, T::observation_type observation, T::action_mask actMask, T::reward_type reward) {
        typename T::observation_type;
        // typename T::action_type;
        typename T::action_mask;
        typename T::reward_type;
        { mind.act(observation, actMask, reward) };
        { mind.endEpisode(reward) } -> std::same_as<void>;
    };

    /**
     *
     * @tparam T
     * @tparam MESSAGE
     */
    template<class T, class MESSAGE>
    concept HasIncomingMessageHook = requires(T object, MESSAGE message) {
        object.incomingMessageHook(message);
    };

    template<class MESSAGE, class OBJ> requires(!HasIncomingMessageHook<OBJ,MESSAGE>)
    void callIncomingMessageHook(OBJ &object, MESSAGE &&message) { }

    template<class MESSAGE, HasIncomingMessageHook<MESSAGE> OBJ>
    void callIncomingMessageHook(OBJ &object, MESSAGE &&message) {
        object.incomingMessageHook(std::forward<MESSAGE>(message));
    }

    /**
     *
     * @tparam T
     * @tparam MESSAGE
     */
    template<class T, class MESSAGE>
    concept HasOutgoingMessageHook = requires(T object, MESSAGE message) {
        object.outgoingMessageHook(message);
    };

    template<class MESSAGE, class OBJ> requires(!HasOutgoingMessageHook<OBJ,MESSAGE>)
    void callOutgoingMessageHook(OBJ &object, MESSAGE &&message) { }

    template<class MESSAGE, HasOutgoingMessageHook<MESSAGE> OBJ>
    void callOutgoingMessageHook(OBJ &object, MESSAGE &&message) {
        object.outgoingMessageHook(std::forward<MESSAGE>(message));
    }

    /**
     *
     * @tparam T
     * @tparam OBSERVATION
     */
    template<class T, class OBSERVATION>
    concept HasHalfStepObservationHook = requires(T object, OBSERVATION observation) {
        object.halfStepObservationHook(observation);
    };

    template<class OBSERVATION, class OBJ> requires(!HasHalfStepObservationHook<OBJ,OBSERVATION>)
    void callHalfStepObservationHook(OBJ &object, OBSERVATION &&observation) { }

    template<class OBSERVATION, HasHalfStepObservationHook<OBSERVATION> OBJ>
    void callHalfStepObservationHook(OBJ &object, OBSERVATION &&observation) {
        object.halfStepObservationHook(std::forward<OBSERVATION>(observation));
    }

}

#endif //MULTIAGENTGOVERNMENT_MIND_H
