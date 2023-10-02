//
// Created by daniel on 16/09/23.
//

#ifndef MULTIAGENTGOVERNMENT_EVENTS_H
#define MULTIAGENTGOVERNMENT_EVENTS_H

#include <tuple>

namespace abm {
    /** A channel is a class method that expects a message from another channel as
     * input and outputs a message to be passed to another channel. The routing
     * is dealt with by a delivery method.
     *
     * @tparam METHOD
     */
    template<auto METHOD> class Channel;
    template<class NODE, class IN, class OUT, OUT(NODE::*METHOD)(IN)>
    class Channel<METHOD> {
    public:
        typedef NODE    node_type;
        typedef IN      in_type;
        typedef OUT     out_type;

        NODE &node;

        Channel(NODE &node): node(node) {};
    };
}

/** Namespace for event types to be used when doing callbacks.
 *
 */
namespace abm::events {


    template<class SOURCE, class DEST, class MESSAGE>
    struct Message {
        SOURCE &  source;
        DEST &    dest;
        MESSAGE &  message;
    };
    template<class SOURCE, class DEST, class MESSAGE> Message(SOURCE &,DEST &, MESSAGE &) -> Message<SOURCE,DEST,MESSAGE>;

    template<class AGENT1, class AGENT2>
    struct StartEpisode {
        AGENT1 &agent1;
        AGENT2 &agent2;
    };
    template<class AGENT1, class AGENT2> StartEpisode(AGENT1 &, AGENT2 &) -> StartEpisode<AGENT1,AGENT2>;

    template<class AGENT1, class AGENT2>
    struct EndEpisode {
        AGENT1 &agent1;
        AGENT2 &agent2;
    };
    template<class AGENT1, class AGENT2> EndEpisode(AGENT1 &, AGENT2 &) -> EndEpisode<AGENT1,AGENT2>;

    template<class ACTION, class OUTMESSAGE, class INMESSAGE, class BODY>
    struct ReinforcementTimestep {
        ACTION     act;
        OUTMESSAGE outMessage;
        INMESSAGE  inMessage;
        double    rewardSinceLastAct;
        BODY &      currentBody;
    };
    template<class ACTION, class OUTMESSAGE, class INMESSAGE,class BODY>
    ReinforcementTimestep(ACTION,OUTMESSAGE,INMESSAGE,double,BODY &)
    -> ReinforcementTimestep<ACTION,OUTMESSAGE,INMESSAGE,BODY>;

    /** Empty event */
    struct Ping { };


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


#endif //MULTIAGENTGOVERNMENT_EVENTS_H
