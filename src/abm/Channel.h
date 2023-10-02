//
// Created by daniel on 21/09/23.
//

#ifndef MULTIAGENTGOVERNMENT_CHANNEL_H
#define MULTIAGENTGOVERNMENT_CHANNEL_H

#include <deque>
#include <functional>

namespace abm {
    /** A ChannelRunner does the delivery of messages on all channels. Use the .run(.)
     * method to start a computational graph computation.
     *
     */
    class ChannelRunner {
        std::deque<std::function<void()>>  undeliveredMessages;

    public:
        template<std::invocable... JOBS>
        inline void submit(JOBS &&... jobs) {
            (undeliveredMessages.push_back(std::forward<JOBS>(jobs)),...);
        }

        template<std::invocable... INITIALMESSAGES>
        inline void run(INITIALMESSAGES... initialMessages) {
            submit(initialMessages...);
            while (!undeliveredMessages.empty()) {
                undeliveredMessages.front()();
                undeliveredMessages.pop_front();
            }
        }
    };

    ChannelRunner defaultChannelRunner = ChannelRunner{};

    template<auto HANDLERPOINTER> class Handler {};

    template<class NODE, class ARG, void(NODE::*handlerMethod)(ARG)>
    struct Handler<handlerMethod> {
        typedef NODE    node_type;
        typedef ARG     arg_type;
        typedef std::remove_cvref_t<ARG> message_type;
    };

    template<class NODE, class MESSAGE>
    concept CanHandle = requires(NODE node, MESSAGE message) { node.handle(message); };

    template<auto runner>
    concept Runner = requires() { runner.submit([](){}); };



    /** A channel is a way of sending messages to another object without having to worry about the
     * type of the destination object. The destination can also be changed at runtime.
     * A message can be sent by using the operator()(message) method. This submits the delivery
     * of the message to a runner (by default this is set to abm::defaultChannelRunner)
     *
     * @tparam MESSAGE  type of the messages that can be sent down this channel. This becomes
     *                  the argument when sending messages down the channel, but messages will
     *                  always be copied/moved into the channel.
     * @tparam runner   the channel runner to use to deliver the messages. This can be any static object
     *                  that implements submit(runnable).
     */
    template<class MESSAGE, auto &runner = defaultChannelRunner> requires Runner<runner>
    class Channel : public std::function<void(MESSAGE)> {
    public:

        Channel() = default;

        template<auto handlerMethod>
        Channel(typename Handler<handlerMethod>::node_type &destinationNode, Handler<handlerMethod> /* static handler identity */) {
            connectToHandler<handlerMethod>(destinationNode);
        }

        template<class NODE>
        Channel(NODE &destinationNode) {
            connectTo(destinationNode);
        }


        /** sets channel to point to the .handle(.) method of the destination node.
         *
         * @tparam handlerMethod the method in the destination node that
         * @param channel channel we want to connect
         * @param destinationNode destination we want to connect to
         */
        template<CanHandle<MESSAGE> NODE>
        void connectTo(NODE &destinationNode) {
            static_cast<std::function<void(MESSAGE)> &>(*this) = [&destinationNode](MESSAGE message) {
                runner.submit([msg = std::move(message), &destinationNode]() {
                    destinationNode.handle(msg);
                });
            };
        }

        /** sets channel to point to a named method in the destination node
         *
         * @tparam handlerMethod the method in the destination node that will be called by this channel
         * @param destinationNode the object we want to connect to
         */
        template<auto handlerMethod>
        void connectToHandler(typename Handler<handlerMethod>::node_type &destinationNode) {
            static_cast<std::function<void(MESSAGE)> &>(*this) = [&destinationNode](MESSAGE message) {
                runner.submit([msg = std::forward<MESSAGE>(message), &destinationNode]() {
                    (destinationNode.*handlerMethod)(msg);
                });
            };
        }

        static void run() { runner.run(); }
    };

    template<auto handlerMethod>
    Channel(typename Handler<handlerMethod>::node_type &destinationNode, Handler<handlerMethod>)
    -> Channel<typename Handler<handlerMethod>::message_type>;

}

#endif //MULTIAGENTGOVERNMENT_CHANNEL_H
