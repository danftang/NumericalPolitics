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
    struct is_optional {
        static constexpr bool value = false;
    };

    template<class T>
    struct is_optional<std::optional<T>> {
        static constexpr bool value = true;
    };

    template<class T>
    constexpr bool is_optional_v = is_optional<T>::value;

    template<class T>
    concept IsOptional = is_optional_v<T>;

    // TODO: An agent may be able to handle multiple types of message, and may send multiple types of
    //  message. It may also communicate with more than one other agent/body (perhaps encoded by message type
    //  (compile time recipient identification) or message content (runtime recipient identification)).
    template<class T>
    concept Body = requires(T body, typename T::action_type actFromMind, typename T::in_message_type messageFromEnvironment) {
        body.actToMessageAndReward(actFromMind);                 // returns <outgoing message,reward> pair
        body.messageToReward(messageFromEnvironment);           // receives incoming message and returns reward
        { body.legalActs() } -> ActionMask; // returns a mask of legal acts (no constraint until joined with mind)
    };

    template<class T, class INMESSAGE>
    concept BodyAsAgent = requires(T body, INMESSAGE message) {
        handle(message); // handles an incoming message and returns a (possibly optional) outgoing message
    };

}

#endif //MULTIAGENTGOVERNMENT_BODY_H
