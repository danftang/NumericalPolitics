//
// Created by daniel on 06/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_QLEARNINGSTEPMIXIN_H
#define MULTIAGENTGOVERNMENT_QLEARNINGSTEPMIXIN_H

#include "../../Agent.h"

namespace abm::events {
    template <class STATE>
    struct QLearningStep {
        STATE * startStatePtr = nullptr;
        size_t action;
        double reward;
        STATE * endStatePtr = nullptr;

        bool isEndOfEpisode() const { return startStatePtr != nullptr && endStatePtr == nullptr; }
        bool isStartOfEpisode() const { return startStatePtr == nullptr; }
        void setEndOfEpisode() { assert(startStatePtr != nullptr); endStatePtr = nullptr; }
    };
}

namespace abm::minds {

    /** Mixin class that intercepts AgentStartEpisode, OutgoingMessage, IncomingMessage and AgentEndEpisode
     * events, and re-emits QTimestep events to the derived class
     */
    template<class STATE> // assumes action is size_t STATE is body state
    class QLearningStepMixin {
    public:
        typedef STATE arg_type;

        events::QLearningStep<arg_type> learningStepEvent;
        std::pair<arg_type,arg_type> states;            // circular buffer for start and end states of body


        /** Remember last act, body state and reward */
        template<class BODY, class REEMIT>
        void on(const events::PreActBodyState<BODY> &event, REEMIT &&reemitCallback) {
            *learningStepEvent.endStatePtr = event.body;
            if(!learningStepEvent.isStartOfEpisode()) {
                callback(learningStepEvent, reemitCallback);
            }
            std::swap(learningStepEvent.startStatePtr, learningStepEvent.endStatePtr);
        }


        /** Remember last act, body state and reward */
        template<class ACTION, class MESSAGE>
        void on(const events::Act<ACTION, MESSAGE> &actEvent) {
            learningStepEvent.reward = actEvent.reward;
            learningStepEvent.action = actEvent.act;
        }


        /** learn from residual reward of end-game */
        template<class BODY, HasCallback<events::QLearningStep<STATE>> DERIVED>
        void on(const events::AgentEndEpisode<BODY> & /* event */, DERIVED && derived) {
            learningStepEvent.setEndOfEpisode();
            callback(learningStepEvent, derived);
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_QLEARNINGSTEPMIXIN_H
