//
// Created by daniel on 03/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_BODY_H
#define MULTIAGENTGOVERNMENT_BODY_H

#include <concepts>
#include "ActionMask.h"

namespace abm {

    template<class T, class ACTION, class MESSAGE>
    concept Body = requires(T body, ACTION actFromMind, MESSAGE messageFromEnvironment) {
//        body.actToMessageAndReward(actFromMind);                 // returns <outgoing message,reward> pair
//        body.messageToReward(messageFromEnvironment);           // receives incoming message and returns reward
        body.handleAct(actFromMind);
        { body.handleMessage(messageFromEnvironment) } -> std::same_as<double>;
        { body.legalActs() } -> ActionMask; // returns a mask of legal acts (no constraint until joined with mind)
    };
}

#endif //MULTIAGENTGOVERNMENT_BODY_H
