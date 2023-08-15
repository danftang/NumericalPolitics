//
// Created by daniel on 03/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_MIND_H
#define MULTIAGENTGOVERNMENT_MIND_H

#include <concepts>
#include "ActionMask.h"


namespace abm {

    /** This is the minimum requirements for a class to be a Mind. In addition, a Mind can add "hooks"
     * which are called at specific points in an episode. See the hook documentation for more details.
     */
    template<class T>
    concept Mind = requires(T mind, T::observation_type observation, T::action_mask actMask, T::reward_type reward) {
        typename T::observation_type;
        // typename T::action_type;
        typename T::action_mask;
        typename T::reward_type;
        { mind.act(observation, actMask, reward) };
        { mind.endEpisode(reward) } -> std::same_as<void>;
    };

    /** If a Mind has an incomingMessageHook method, it will be called when a
     * message is received, directly after it is processed by the body.
     */
    template<class T, class MESSAGE>
    concept HasIncomingMessageHook = requires(T object, MESSAGE message) {
        object.incomingMessageHook(message);
    };

    template<class MESSAGE, class OBJ>
    void callIncomingMessageHook(OBJ &object, MESSAGE &&message) { }

    template<class MESSAGE, HasIncomingMessageHook<MESSAGE> OBJ>
    void callIncomingMessageHook(OBJ &object, MESSAGE &&message) {
        object.incomingMessageHook(std::forward<MESSAGE>(message));
    }

    /** If a Mind has an outgoingMessageHook method, it will be called directly
     * after a body has processed an act
     */
    template<class MIND, class MESSAGE>
    concept HasOutgoingMessageHook = requires(MIND mind, MESSAGE message) {
        mind.outgoingMessageHook(message);
    };

    template<class MIND, class MESSAGE>
    inline void callOutgoingMessageHook(MIND &object, MESSAGE &&message) { }

    template<class MIND, class MESSAGE> requires HasOutgoingMessageHook<MIND,MESSAGE>
    inline void callOutgoingMessageHook(MIND &mind, MESSAGE &&message) {
        mind.outgoingMessageHook(std::forward<MESSAGE>(message));
    }

    /** If a Mind has a halfStepObservationHook method it will be called on receipt of a
     * message, directly BEFORE the body processes the message.
     */
    template<class MIND>
    concept HasHalfStepObservationHook = requires(MIND mind, MIND::observation_type observation) {
        mind.halfStepObservationHook(observation);
    };

    template<Mind MIND>
    inline void callHalfStepObservationHook(MIND &mind, const typename MIND::observation_type &observation) { }

    template<Mind MIND> requires HasHalfStepObservationHook<MIND>
    inline void callHalfStepObservationHook(MIND &mind, const typename MIND::observation_type &observation) {
        mind.halfStepObservationHook(observation);
    }


    /** If a Mind has a startEpisodeHook method it will be called at the start of an
     * episode
     */
    template<class MIND, class BODY, class SHAREDINFORMATION>
    concept HasInitEpisodeHook = requires(MIND mind, BODY body, SHAREDINFORMATION sharedinformation) {
        mind.initEpisodeHook(body, sharedinformation);
    };

    template<class MIND, class BODY, class SHAREDINFORMATION>
    inline void callInitEpisodeHook(MIND &mind, BODY &&body, SHAREDINFORMATION &&info) { }

    template<class MIND, class BODY, class SHAREDINFORMATION> requires HasInitEpisodeHook<MIND,BODY,SHAREDINFORMATION>
    inline void callInitEpisodeHook(MIND &mind, BODY &&body, SHAREDINFORMATION &&sharedInfo) {
        mind.initEpisodeHook(std::forward<BODY>(body), std::forward<SHAREDINFORMATION>(sharedInfo));
    }


    template<class MIND, class BODY>
    concept HasInitEpisodeHookNoSharedInfo = requires(MIND mind, BODY body) {
        mind.initEpisodeHook(body);
    };

    template<class MIND, class BODY>
    inline void callInitEpisodeHook(MIND &mind, BODY body) { }

    template<class MIND, class BODY> requires HasInitEpisodeHookNoSharedInfo<MIND,BODY>
    inline void callInitEpisodeHook(MIND &mind, BODY &&body) {
        mind.initEpisodeHook(std::forward<BODY>(body));
    }


}

#endif //MULTIAGENTGOVERNMENT_MIND_H
