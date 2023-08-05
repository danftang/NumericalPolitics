// Wrapper for bodies so that the last move is recorded
//
// Created by daniel on 03/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_MESSAGERECORDER_H
#define MULTIAGENTGOVERNMENT_MESSAGERECORDER_H

#include "../Body.h"

namespace abm::bodies {
    template<Body BODY>
    class MessageRecorder: public BODY {
    public:
        typedef BODY::message_type message_type;
        typedef BODY::action_type action_type;
        typedef BODY::action_mask action_mask;

        message_type actToMessage(action_type action) {
            lastOutgoingMessage = BODY::actToMessage(action);
            return lastOutgoingMessage;
        }

        double messageToReward(message_type incomingMessage) {
            lastIncomingMessage = incomingMessage;
            return BODY::messageToReward(incomingMessage);
        }

        // TODO: add int and vector casts

        message_type lastOutgoingMessage;
        message_type lastIncomingMessage;
    };

}


#endif //MULTIAGENTGOVERNMENT_MESSAGERECORDER_H
