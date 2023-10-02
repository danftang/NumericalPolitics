//
// Created by daniel on 19/09/23.
//

#ifndef MULTIAGENTGOVERNMENT_STATICCOMPUTATIONALGRAPH_H
#define MULTIAGENTGOVERNMENT_STATICCOMPUTATIONALGRAPH_H

#include <cstdlib>
#include <utility>
#include <tuple>
#include <optional>
#include <future>

namespace abm {


    template<class NODE, class MESSAGE>
    concept CanHandle = requires(NODE node, MESSAGE message) { node.handle(message); };


    /** A wrapper for a message in order to add a compile-time node-id for the intended recipient.
     * A convenience function staticallyAddress<ID>(.) is provided to construct a StaticallyAddressedMessage
     * without having to specify the parameters.
     */
    template<size_t DESTINATION, class MESSAGE>
    struct StaticallyAddressedMessage {
        MESSAGE value;
    };

    template<size_t DESTINATION, class MESSAGE>
    StaticallyAddressedMessage<DESTINATION,typename std::remove_reference<MESSAGE>::type> staticallyAddress(MESSAGE &&message) {
        return { std::forward<MESSAGE>(message) };
    }


    /** A StaticComputationalGraph is a graph with a fixed number of nodes with unchanging edge connections.
     * Nodes are computational units that implement the handle(.) method to perform computations.
     * The handle(.) method is called with a message each time one is sent to this node from a neighbouring node.
     * The method should return one of:
     *   1) a StaticallyAddressedMessage
     *   2) a std::optional<StaticallyAddressedMessage>
     *   3) a tuple of some combination of the above
     * which specify a (set of) messages to be delivered to other nodes. Unset optionals mean no message.
     *
     * The structure of the graph is defined in the constructor, which should be sent a number of rvalue references
     * to the nodes of the graph. Each node is given an identifying "address" which is its zero-based order
     * in the call to the constructor. Nodes should be made statically aware of the addresses of the nodes on
     * their out-edges (e.g. by passing as size_t template parameters), which can then be used in the handle(.)
     * methods to address the outgoing messages.
     *
     * Computations are started by injecting initial messages into the graph using deliver(.). This will initiate
     * a cascade of messages to perform a computation.
     *
     * TODO: allow arrays (and vectors for dynamic graph) of nodes in the tuple?
     */
    template<class...NODES>
    class StaticComputationalGraph {
    public:
        std::tuple<NODES...> nodes;

        /** Construct with a set of rvalue references to the nodes of the graph
         * which will be moved into this.
         * The address of a node is taken from its zero-based order in this constructor.
         * You can then get lvalue references to the nodes using node<ADDRESS>().
         * Runtime indexing is not possible.
         *
         * @param nodes
         */
        StaticComputationalGraph(NODES &&...nodes) : nodes(std::move(nodes)...) {}

        /**
         *
         * @tparam ADDRESS address of the node to get, as given by its order during construction of this.
         * @return an lvalue reference to the node at the given address.
         */
        template<size_t ADDRESS> inline auto &node() { return std::get<ADDRESS>(nodes); }

        template<size_t DESTINATION, class MESSAGE>
        inline void deliver(const StaticallyAddressedMessage<DESTINATION,MESSAGE> &message) {
            auto &destinationNode = node<DESTINATION>();
            static_assert(CanHandle<decltype(destinationNode),MESSAGE>); // improve error messages a bit
            deliver(destinationNode.handle(message.value)); // N.B. tail-call optimisation will prevent stack overflow.
        }

        template<size_t DESTINATION, class MESSAGE>
        inline void deliver(const std::optional<StaticallyAddressedMessage<DESTINATION,MESSAGE>> &optMessage) {
            if(!optMessage.has_value()) return;
            deliver(*optMessage);
        }

        template<class... ADDRESSEDMESSAGES>
        inline void deliver(const std::tuple<ADDRESSEDMESSAGES...> &messageTuple) {
            std::apply([this](ADDRESSEDMESSAGES...messages) { (deliverAsync(messages),...); }, messageTuple);
        }

        /** Delivers a message asynchronously by submitting the delivery as a task to a thread-pool.
         */
        template<size_t DESTINATION, class MESSAGE>
        inline void deliverAsync(const StaticallyAddressedMessage<DESTINATION,MESSAGE> &message) {
            std::async([this,message]() { deliver(message); });
        }

        template<size_t DESTINATION, class MESSAGE>
        inline void deliverAsync(const std::optional<StaticallyAddressedMessage<DESTINATION,MESSAGE>> &optMessage) {
            if(!optMessage.has_value()) return;
            deliverAsync(*optMessage);
        }
    };

    template<class...NODES>
    class DynamicComputationalGraph {
    public:
        std::tuple<NODES...> nodes;

        /** Construct with a set of rvalue references to the nodes of the graph
         * which will be moved into this.
         * The address of a node is taken from its zero-based order in this constructor.
         * You can then get lvalue references to the nodes using node<ADDRESS>().
         * Runtime indexing is not possible.
         *
         * @param nodes
         */
        StaticComputationalGraph(NODES &&...nodes) : nodes(std::move(nodes)...) {}

        /**
         *
         * @tparam ADDRESS address of the node to get, as given by its order during construction of this.
         * @return an lvalue reference to the node at the given address.
         */
        template<size_t ADDRESS> inline auto &node() { return std::get<ADDRESS>(nodes); }

        template<size_t DESTINATION, class MESSAGE>
        inline void deliver(const StaticallyAddressedMessage<DESTINATION,MESSAGE> &message) {
            auto &destinationNode = node<DESTINATION>();
            static_assert(CanHandle<decltype(destinationNode),MESSAGE>); // improve error messages a bit
            deliver(destinationNode.handle(message.value)); // N.B. tail-call optimisation will prevent stack overflow.
        }

        template<size_t DESTINATION, class MESSAGE>
        inline void deliver(const std::optional<StaticallyAddressedMessage<DESTINATION,MESSAGE>> &optMessage) {
            if(!optMessage.has_value()) return;
            deliver(*optMessage);
        }

        template<class... ADDRESSEDMESSAGES>
        inline void deliver(const std::tuple<ADDRESSEDMESSAGES...> &messageTuple) {
            std::apply([this](ADDRESSEDMESSAGES...messages) { (deliverAsync(messages),...); }, messageTuple);
        }

        /** Delivers a message asynchronously by submitting the delivery as a task to a thread-pool.
         */
        template<size_t DESTINATION, class MESSAGE>
        inline void deliverAsync(const StaticallyAddressedMessage<DESTINATION,MESSAGE> &message) {
            std::async([this,message]() { deliver(message); });
        }

        template<size_t DESTINATION, class MESSAGE>
        inline void deliverAsync(const std::optional<StaticallyAddressedMessage<DESTINATION,MESSAGE>> &optMessage) {
            if(!optMessage.has_value()) return;
            deliverAsync(*optMessage);
        }
    };

}

#endif //MULTIAGENTGOVERNMENT_STATICCOMPUTATIONALGRAPH_H
