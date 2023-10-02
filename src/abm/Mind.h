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
     *
     * TODO: A mind is just a trainable function from body to action. The mask type is defined by the body type.
     *   In the case of reinforcement learning the observation type is Reward (or step).
     *
     * TODO: Should we have callbacks or train(...) on steps/events. Train then becomes effectively an observation
     *   of any type. A body will offer a set of observation types (through typedef on a tuple, or by callbacks
     *   to the Agent) and the mind can be scanned
     *   to see which of these it can accept (using concepts or the hook mechanism at the call site). In this
     *   way we can separate the state of the body from the observations it generates.
     *
     * TODO: Should a mind be able to act after any observation? (i.e. an push actions rather than pull actions.
     *   A mind then becomes a set of functions from observations to actions with training by use. Or perhaps
     *   the body and mind are also message passers, and so are effectively separate agents. In this way, the
     *   mind can send a message to the body at any time. A person then becomes a subset of the agent communication
     *   graph...and any subset can be grouped to form an agent).
     *
     */
    template<class T, class BODY>
    concept Mind = requires(T mind, BODY body) {
        mind.act(body); // returns a (possibly optional) act which will be sent to the body
    };

    // ---------------- Optional Callbacks -----------------------
    // A Mind can intercept various observations by
    // implementing overloaded on(.) methods:
    //   on(IncomingMessage)
    //   on(OutgoingMessage)
    //   on(Reward)
    //   on(EndEpisode)

//
//    /** If a Mind has an onIncomingMessage method, it will be called when a
//     * message is received, directly after it is processed by the body.
//     */
//    template<class MIND, class MESSAGE, class BODY>
//    concept HasIncomingMessageCallback = requires(MIND mind, MESSAGE message, BODY body) {
//        mind.onIncomingMessage(message, body);
//    };
//
//    template<class MIND, class MESSAGE, class BODY>
//    inline void incomingMessageCallback(MIND &object, MESSAGE &&message, BODY &&body) { }
//
//    template<class MIND, class MESSAGE, class BODY> requires HasIncomingMessageCallback<MIND,MESSAGE,BODY>
//    inline void incomingMessageCallback(MIND &mind, MESSAGE &&message, BODY &&body) {
//        mind.onIncomingMessage(std::forward<MESSAGE>(message), std::forward<BODY>(body));
//    }
//
//
//    /** If a Mind has an onOutgoingMessage method, it will be called directly
//     * after a body has processed an act and turned it into a message, and directly
//     * before it is sent.
//     */
//    template<class MIND, class MESSAGE, class BODY>
//    concept HasOutgoingMessageCallback = requires(MIND mind, MESSAGE message, BODY body) {
//        mind.onOutgoingMessage(message, body);
//    };
//
//    template<class MIND, class MESSAGE, class BODY>
//    inline void outgoingMessageCallback(MIND &object, MESSAGE &&message, BODY &&body) { }
//
//    template<class MIND, class MESSAGE, class BODY> requires HasOutgoingMessageCallback<MIND,MESSAGE,BODY>
//    inline void outgoingMessageCallback(MIND &mind, MESSAGE &&message, BODY &&body) {
//        mind.onOutgoingMessage(std::forward<MESSAGE>(message), std::forward<BODY>(body));
//    }
//
//
//    /** If a Mind has an onInit method it will be called at the start of every
//     * episode
//     */
//    template<class T>
//    concept HasParameterlessInitCallback = requires(T obj) {
//        obj.onInit();
//    };
//
//    template<class T>
//    concept HasOneParamInitCallback = requires(T obj, typename T::init_type sharedinformation) {
//        typename T::init_type;
//        obj.onInit(sharedinformation);
//    };
//
//    template<class T, class ARG> requires(!HasOneParamInitCallback<T> && !HasParameterlessInitCallback<T>)
//    inline void initCallback(T &obj, ARG &&info) { }
//
////    template<class T, class ARG> requires(HasOneParamInitCallback<T> && std::convertible_to<ARG,typename T::init_type>)
////    inline void initCallback(T &&obj, ARG &&sharedInfo) {
////        std::forward<T>(obj).onInit(std::forward<ARG>(sharedInfo));
////    }
//
//    template<class T> requires HasOneParamInitCallback<T>
//    inline void initCallback(T &obj, typename T::init_type sharedInfo) {
//        obj.onInit(sharedInfo);
//    }
//
//    template<class T, class ARG> requires(HasParameterlessInitCallback<T> && !HasOneParamInitCallback<T>)
//    inline void initCallback(T &obj, ARG &&dummy) {
//        obj.onInit();
//    }
//
//    template<class T>
//    inline void initCallback(T &&obj) { }
//
//    template<class T> requires HasParameterlessInitCallback<T>
//    inline void initCallback(T &&obj) {
//        std::forward<T>(obj).onInit();
//    }


//
//    template<class MIND, class BODY>
//    concept HasInitEpisodeHookNoSharedInfo = requires(MIND mind, BODY body) {
//        mind.initEpisodeHook(body);
//    };
//
//    template<class MIND, class BODY>
//    inline void callInitEpisodeHook(MIND &mind, BODY body) { }
//
//    template<class MIND, class BODY> requires HasInitEpisodeHookNoSharedInfo<MIND,BODY>
//    inline void callInitEpisodeHook(MIND &mind, BODY &&body) {
//        mind.initEpisodeHook(std::forward<BODY>(body));
//    }


}

#endif //MULTIAGENTGOVERNMENT_MIND_H
