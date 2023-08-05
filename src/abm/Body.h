//
// Created by daniel on 03/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_BODY_H
#define MULTIAGENTGOVERNMENT_BODY_H

#include <concepts>
#include "ActionMask.h"

namespace abm {

    template<class T>
    concept Body = requires(T body, T::action_type actFromMind, T::in_message_type messageFromEnvironment) {
        typename T::in_message_type;   // incoming message type
        typename T::action_type;    // incoming action type
        body.actToMessage(actFromMind);                 // no constraints on outgoing message
        body.messageToReward(messageFromEnvironment);   // returns a reward (no constraint until joined with mind)
        body.legalActs();                               // returns a mask of legal acts (no constraint until joined with mind)
        { body.isEndOfEpisode() } -> std::same_as<bool>; // can be called after actToMessage or messageToReward to signal terminal condition (how to get final reward?)
        body.endEpisode();                               // ends the episode, returning any outstanding reward from final act.
    };
}

#endif //MULTIAGENTGOVERNMENT_BODY_H
