//
// Created by daniel on 08/09/23.
//

#ifndef MULTIAGENTGOVERNMENT_ACTIONRESPONSEREWARD_H
#define MULTIAGENTGOVERNMENT_ACTIONRESPONSEREWARD_H

#include "../abm/Body.h"

namespace observations {

    template<abm::Body MYBODY, class OUTMESSAGE = MYBODY::out_message_type, class INMESSAGE=MYBODY::in_message_type>
    class ActionResponseReward {
    public:
        MYBODY startState;
        MYBODY::action_type action;
        OUTMESSAGE outgoingMessage;
        INMESSAGE  incomingMessage;
        const MYBODY &endState;
        double reward;
    };

}


#endif //MULTIAGENTGOVERNMENT_ACTIONRESPONSEREWARD_H
