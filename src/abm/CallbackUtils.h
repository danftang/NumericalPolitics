//
// Created by daniel on 16/09/23.
//

#ifndef MULTIAGENTGOVERNMENT_CALLBACKUTILS_H
#define MULTIAGENTGOVERNMENT_CALLBACKUTILS_H

#include <tuple>

//namespace abm {
//    /** A channel is a class method that expects a message from another channel as
//     * input and outputs a message to be passed to another channel. The routing
//     * is dealt with by a delivery method.
//     *
//     * @tparam METHOD
//     */
//    template<auto METHOD> class Channel;
//    template<class NODE, class IN, class OUT, OUT(NODE::*METHOD)(IN)>
//    class Channel<METHOD> {
//    public:
//        typedef NODE    node_type;
//        typedef IN      in_type;
//        typedef OUT     out_type;
//
//        NODE &node;
//
//        Channel(NODE &node): node(node) {};
//    };
//}

/** Namespace for event types to be used when doing callbacks.
 *
 */
namespace abm {

    // ====================== callback mechanism ===================================

    template<class T, class EVENT>
    concept HasCallback = requires(T obj, EVENT event) { obj.on(event); };

    template<class T>
    concept IsTupleOrRef = deselby::IsClassTemplateOf<std::remove_reference_t<T>, std::tuple>;


    template<class EVENT, HasCallback<EVENT> CALLBACK>
    inline void callback(const EVENT &event, CALLBACK &&callback) {
        callback.on(event);
    }

    template<class EVENT, IsTupleOrRef TUPLEOFCALLBACKS>
    inline void callback(const EVENT &event, TUPLEOFCALLBACKS &&callbackTuple) {
        deselby::for_each(callbackTuple,
            [&event](auto &element) { callback(event, element); });
    }

    template<class EVENT, class CALLBACK> requires(!HasCallback<CALLBACK,EVENT> && !IsTupleOrRef<CALLBACK>)
    inline void callback(const EVENT &, CALLBACK &&) { }

    template<class EVENT, class...CALLBACKS> requires(sizeof...(CALLBACKS)>1)
    inline void callback(const EVENT &event, CALLBACKS &&...callbacks) {
        (callback(event, std::forward<CALLBACKS>(callbacks)),...);
    }
}


namespace abm::callbacks {
    // ========================= FUNCTION TO CALLBACK WRAPPER =======================

    /** Wraps a lambda, or other function, so that its call operator is moved to the
     * .on(.) method for interception of events
     */
    template<class FUNCTION>
    class CallbackFunction {
    public:
        FUNCTION function;
        CallbackFunction(FUNCTION function): function(std::move(function)) {}

        template<class EVENT> requires requires(EVENT event) { function(event); }
        inline void on(EVENT &&event) { function(std::forward<EVENT>(event)); }
    };
}



#endif //MULTIAGENTGOVERNMENT_CALLBACKUTILS_H
