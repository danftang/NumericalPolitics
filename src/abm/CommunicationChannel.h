//
// Created by daniel on 08/04/23.
//

#ifndef MULTIAGENTGOVERNMENT_COMMUNICATIONCHANNEL_H
#define MULTIAGENTGOVERNMENT_COMMUNICATIONCHANNEL_H

#include <functional>
#include "Schedule.h"

namespace abm {

    // A channel object represents the sending end of a communication channel.
    //
    // To send a message down the channel, call the channel with a (message, sendTime)
    // or use the send() method. This returns a schedule that contains the event that
    // processes the receipt of the message.
    //
    // The default constructor creates a channel that isn't connected
    // (i.e. sending data down the channel creates a scheduled event that does nothing).
    // A connected channel can be created by constructing with a handler and a transit time
    // or by calling the connectTo(handler, transitTime) method.
    //
    // A Handler must be a callable that takes a (message, sendTime) and returns a schedule.
    // The schedule must have a SCHEDULE::time_type which defines the type used to measure time
    template<class HANDLER, class SCHEDULE, class MESSAGE>
    concept MessageHandler = requires(HANDLER handler, MESSAGE message) {
        typename SCHEDULE::time_type;
        { handler(message, std::declval<SCHEDULE::time_type>()) } -> std::convertible_to<SCHEDULE>;
    };

    template<class SCHEDULE, class MESSAGE>
    class CommunicationChannel : public std::function<SCHEDULE(MESSAGE, typename SCHEDULE::time_type)> {
    public:
        typedef MESSAGE message_type;
        typedef SCHEDULE schedule_type;
        typedef SCHEDULE::time_type time_type;
        typedef std::function<SCHEDULE(MESSAGE, typename SCHEDULE::time_type)> send_function_type;

        CommunicationChannel() {
            // unconnected channel
        }

        template<MessageHandler<SCHEDULE,MESSAGE> LAMBDA>
        CommunicationChannel(LAMBDA handler, time_type transitTime) {
            connectTo(handler, transitTime);
        }


//        template<MessageHandler<SCHEDULE,MESSAGE> LAMBDA>
//        void connectTo(LAMBDA handler, time_type transitTime) {
//            send_function_type::operator=([transitTime, handler](MESSAGE message, time_type sendTime) {
//                time_type deliveryTime = sendTime + transitTime;
//                return SCHEDULE{[message, handler, deliveryTime]() { return handler(message, deliveryTime); },
//                                deliveryTime};
//            });
//        }


        template<class AGENT>
        void connectTo(AGENT &agent, SCHEDULE(AGENT::*handlerMethod)(MESSAGE, time_type), time_type transitTime) {
            send_function_type::operator=([&agent, transitTime, handlerMethod](MESSAGE message, time_type sendTime) {
                time_type deliveryTime = sendTime + transitTime;
                return SCHEDULE(
                        [message, &agent, handlerMethod, deliveryTime]() {
                            return (agent.*handlerMethod)(message, deliveryTime);
                        },
                        deliveryTime);
            });
        }

        // Alternative way of sending messages down the channel (other than operator ())
        inline SCHEDULE send(MESSAGE message, time_type sendTime) {
            return (*this)(message, sendTime);
        }
    };


    // Specialisation for case of a "ping" message that has no data payload
    template<class SCHEDULE>
    class CommunicationChannel<SCHEDULE,void> : public std::function<SCHEDULE(typename SCHEDULE::time_type)> {
    public:
        typedef void message_type;
        typedef SCHEDULE schedule_type;
        typedef SCHEDULE::time_type time_type;
        typedef std::function<SCHEDULE(typename SCHEDULE::time_type)> send_function_type;

        template<class AGENT>
        void connectTo(AGENT &agent, SCHEDULE(AGENT::*handlerMethod)(time_type), time_type transitTime) {
            send_function_type::operator=([&agent, transitTime, handlerMethod](time_type sendTime) {
                time_type deliveryTime = sendTime + transitTime;
                return SCHEDULE(
                        [&agent, handlerMethod, deliveryTime]() {
                            return (agent.*handlerMethod)(deliveryTime);
                        },
                        deliveryTime);
            });
        }

        // Alternative way of sending messages down the channel (other than operator ())
        inline SCHEDULE send(time_type sendTime) {
            return (*this)(sendTime);
        }
    };

};

#endif //MULTIAGENTGOVERNMENT_COMMUNICATIONCHANNEL_H
