//
// Created by daniel on 06/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_QLEARNINGSTEPMIXIN_H
#define MULTIAGENTGOVERNMENT_QLEARNINGSTEPMIXIN_H

#include "../../Agent.h"

namespace abm::events {
    template <class STATE>
    struct QLearningStep {
        STATE *startStatePtr;
        size_t action;
        double reward;
        STATE *endStatePtr;

        bool isEndOfEpisode() const { return endStatePtr == nullptr; }
    };
}

namespace abm::minds {

    /** Mixin class that intercepts AgentStartEpisode, OutgoingMessage, IncomingMessage and AgentEndEpisode
     * events, and re-emits QTimestep events to PARENT
     * PARENT should derive from this class, in the 'curiously recurring' pattern.
     * TODO: We could allow any event handler to raise more events which are delivered only to
     *  the single object on which the callback was originally raised. When a class passes on events
     *  to subobjects it has the option to call the handler directly (after requiring it's there)
     *  or using callback if it doesn't want to receive any re-raised events.
     */
    template<class PARENT, class STATE> // assumes action is size_t and BODY is PARENT::arg_type
    class QLearningStepMixin {
    public:
        typedef STATE arg_type;

        events::QLearningStep<arg_type> event;
        std::pair<arg_type,arg_type> states;
        bool actionIsValid;

        void myFunc() {}

        /** Setup */
        template<class BODY>
        void on(const events::AgentStartEpisode<BODY> &startEvent) {
            event.startStatePtr = &states.first;
            event.endStatePtr = &states.second;
            states.first = startEvent.body;
            actionIsValid = false;
        }


        /** Remember last act, body state and reward */
        template<class BODY, class ACTION, class MESSAGE>
        void on(const events::OutgoingMessage<BODY, ACTION, MESSAGE> &outMessage) {
            event.reward = outMessage.reward;
            event.action = outMessage.act;
            actionIsValid = true;
        }


        /** Learn from last call/response step */
        template<class BODY, class MESSAGE>
        void on(const events::IncomingMessage<MESSAGE, BODY> &inMessage) {
            *event.endStatePtr = inMessage.body;
            if(actionIsValid) {
                event.reward += inMessage.reward;
                callback(event, static_cast<PARENT &>(*this));
            }
            std::swap(event.startStatePtr, event.endStatePtr);
        }


        /** learn from residual reward of end-game */
        template<class BODY>
        void on(const events::AgentEndEpisode<BODY> & /* event */) {
            event.endStatePtr = nullptr; // signal end of episode
            callback(event, static_cast<PARENT &>(*this));
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_QLEARNINGSTEPMIXIN_H
