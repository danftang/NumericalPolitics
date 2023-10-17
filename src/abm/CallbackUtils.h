//
// Created by daniel on 16/09/23.
//

#ifndef MULTIAGENTGOVERNMENT_CALLBACKUTILS_H
#define MULTIAGENTGOVERNMENT_CALLBACKUTILS_H

#include <tuple>

/** Namespace for event types to be used when doing callbacks.
 *
 */
namespace abm {

    // ====================== callback concepts ===================================


    /** true if T has a callback handler for EVENT */

    template<class T, class EVENT>
    concept HasSimpleCallback = requires(T obj, EVENT event) { obj.on(event); };

    /** use this kind of callback when you expect your implementation to be derived from
     * and you wish to re-emit events to (or do some other stuff with) the derived object. */
    template<class T, class EVENT>
    concept HasReEmitCallback = requires(T obj, EVENT event) { obj.on(event,obj); };

    template<class T, class EVENT>
    concept HasCallback = HasSimpleCallback<T,EVENT> || HasReEmitCallback<T,EVENT>;


    /** true if EVENT is intercepted by an on() method in T (useful for constraining functions that receive events) */
    template<class EVENT, class T>
    concept IsEventHandledBy = HasCallback<T,EVENT>;

    /** true if T is a tuple or a reference to a tuple (possibly cv-qualified) */
//    template<class T>
//    concept IsTupleOrRef = deselby::IsSpecializationOf<std::remove_reference_t<T>, std::tuple>;

    // ====================== callback mechanism ===================================
    // TODO: with callback re-emit we have the possibility of stack overflow if we don't
    //  tail recurse on recursive calls. So, we could require that re-emitting handlers
    //  only re-emit as the last operation and ensure that these callbacks are also tail
    //  recursive, or set up our own event dispatch stack (and possibly require handlers tp
    //  be thread safe?).
    //  Single thread handling:
    //  SimpleCallbacks can be dispatched immediately. HasReEmitCallbacks proceed as follows:
    //    - the (event,handler) is pushed to a static stack (after type deletion [to a runnable?])
    //    - we check a flag to see if the stack is already being looped. If not, set the flag
    //      and dispatch items on the stack until empty.
    //  Once the flag is raised (indicating we're in the re-emit loop) new re-emit callbacks
    //  will put themselves on the stack and exit, meaning we only ever have one level of recursion.
    //  ...or...
    //  we simply use the call stack and have this as a limit.

    template<class EVENT, HasSimpleCallback<EVENT> CALLBACK>
    inline void callback(EVENT &&event, CALLBACK &&handler) {
        handler.on(std::forward<EVENT>(event));
    }

    template<class EVENT, HasReEmitCallback<EVENT> CALLBACK>
    inline void callback(EVENT &&event, CALLBACK &&handler) {
        handler.on(std::forward<EVENT>(event), handler);
    }

    template<class EVENT,class... CALLBACKS>
    inline void callback(const EVENT &event, std::tuple<CALLBACKS...> &callbackTuple) {
        deselby::for_each(callbackTuple,
            [&event](auto &element) { callback(event, element); });
    }

    template<class EVENT,class... CALLBACKS>
    inline void callback(const EVENT &event, const std::tuple<CALLBACKS...> &callbackTuple) {
        deselby::for_each(callbackTuple,
                          [&event](auto &element) { callback(event, element); });
    }

    template<class EVENT, class CALLBACK> requires(!HasCallback<CALLBACK,EVENT>)
    inline void callback(EVENT &&, CALLBACK &&) { }

    template<class EVENT, class...CALLBACKS> requires(sizeof...(CALLBACKS)>1)
    inline void callback(const EVENT &event, CALLBACKS &&...callbacks) {
        (callback(event, std::forward<CALLBACKS>(callbacks)),...);
    }
}


namespace abm::callbacks {
    // ========================= FUNCTION TO CALLBACK WRAPPER =======================

    /** Wraps a lambda, or other function, so that its call operator(s) is(are) moved to the
     * .on(.) method for interception of events
     */
    template<class FUNCTION>
    class CallbackFunction {
    public:
        FUNCTION function;
        CallbackFunction(FUNCTION function): function(std::move(function)) {}

        template<class EVENT> requires std::is_invocable_v<FUNCTION,EVENT>
        inline void on(EVENT &&event) {
            function(std::forward<EVENT>(event));
        }

        template<class EVENT, class DERIVED> requires std::is_invocable_v<FUNCTION,EVENT,DERIVED>
        inline void on(EVENT &&event, DERIVED &&derived) {
            function(std::forward<EVENT>(event), std::forward<DERIVED>(derived));
        }
    };


}



#endif //MULTIAGENTGOVERNMENT_CALLBACKUTILS_H
