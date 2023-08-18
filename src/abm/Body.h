//
// Created by daniel on 03/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_BODY_H
#define MULTIAGENTGOVERNMENT_BODY_H

#include <concepts>
#include "ActionMask.h"

namespace abm {

//    template<class T>
//    concept IsOptional = requires(T obj) {
//        typename T::value_type;
//        { obj.has_value() } -> std::same_as<bool>;
//        obj.value();
//    };


    template<class T>
    concept Body = requires(T body, T::action_type actFromMind, T::in_message_type messageFromEnvironment) {
        typename T::in_message_type;   // incoming message type
        typename T::action_type;    // incoming action type
        body.actToMessage(actFromMind);                 // returns outgoing message of any type
        body.endEpisode();                               // ends the episode, returning any outstanding reward from final act.
        { body.messageToReward(messageFromEnvironment) } -> std::same_as<decltype(body.endEpisode())>;
        { body.legalActs() } -> ActionMask; // returns a mask of legal acts (no constraint until joined with mind)
//        { body.isEndOfEpisode() } -> std::same_as<bool>; // can be called after actToMessage or messageToReward to signal terminal condition (how to get final reward?)
    };

    template<Body BODY>
    class Traits {
    public:
        typedef BODY::in_message_type in_message_type;
        typedef BODY::action_type action_type;
        typedef decltype(std::declval<BODY>().actToMessage(std::declval<typename BODY::action_type>())) out_message_type;
        typedef decltype(std::declval<BODY>().endEpisode()) reward_type;
        typedef decltype(std::declval<BODY>().legalActs()) action_mask;
    };
}

#endif //MULTIAGENTGOVERNMENT_BODY_H
