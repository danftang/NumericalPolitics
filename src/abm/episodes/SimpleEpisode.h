//
// Created by daniel on 14/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_SIMPLEEPISODE_H
#define MULTIAGENTGOVERNMENT_SIMPLEEPISODE_H

#include <functional>
#include <concepts>
#include <utility>

#include "../CallbackUtils.h"

namespace abm::events {
    template<class SOURCE, class DEST, class MESSAGE>
    struct Message {
        SOURCE &source;
        DEST &dest;
        MESSAGE &message;
    };
    template<class SOURCE, class DEST, class MESSAGE> Message(SOURCE &, DEST &,
                                                              MESSAGE &) -> Message<SOURCE, DEST, MESSAGE>;

    template<class AGENT1, class AGENT2>
    struct StartEpisode {
        AGENT1 &agent1;
        AGENT2 &agent2;
    };
    template<class AGENT1, class AGENT2> StartEpisode(AGENT1 &, AGENT2 &) -> StartEpisode<AGENT1, AGENT2>;

    template<class AGENT1, class AGENT2>
    struct EndEpisode {
        AGENT1 &agent1;
        AGENT2 &agent2;
    };
    template<class AGENT1, class AGENT2> EndEpisode(AGENT1 &, AGENT2 &) -> EndEpisode<AGENT1, AGENT2>;
}


namespace abm::callbacks {
    constexpr auto print = [](auto &agent) { std::cout << agent << std::endl; };

    /** Callback object for verbose logging of episodes
     */
    class Verbose {
    public:
        void *firstMover;

        template<class AGENT1, class AGENT2>
        void on(const events::StartEpisode<AGENT1, AGENT2> &event) {
            firstMover = &event.agent1;
            std::cout << "------- Starting episode -------" << std::endl;
            deselby::constexpr_if<deselby::IsStreamable<AGENT1>>(print, event.agent1);
            deselby::constexpr_if<deselby::IsStreamable<AGENT2>>(print, event.agent2);
        }

        template<class SOURCE, class DEST, class MESSAGE>
        void on(const events::Message<SOURCE, DEST, MESSAGE> &event) {
            bool isFromFirstMover = (&event.source == firstMover);
            std::cout << (isFromFirstMover ? "--> " : "<-- ") << event.message << std::endl;
        }

        template<class AGENT1, class AGENT2>
        void on(const events::EndEpisode<AGENT1, AGENT2> &event) {
            deselby::constexpr_if<deselby::IsStreamable<AGENT1>>(print, event.agent1);
            deselby::constexpr_if<deselby::IsStreamable<AGENT2>>(print, event.agent2);
            std::cout << "------- Ending episode -------" << std::endl;
        }

        // TODO: handle more message types...
    };

    class MessageCounter {
    public:
        size_t nMessages;
        template<class SOURCE, class DEST, class MESSAGE>
        void on(const events::Message<SOURCE,DEST,MESSAGE> & /* event */) {
            ++nMessages;
        }
    };
}

namespace abm::episodes {
    /** Runs episodic interactions between two agents of any type.
     * Agents should implement startEpisode() and handleMessage(.) methods.
     * which can return a message of any type or an optional of any type.
     * At the end of an episode, an EndEpisode event will be sent
     * to both agents
     * The supplied callbacks will be called on the following events:
     *   - StartEpisode
     *   - Message
     *   - EndEpisode
     * TODO: add synchronous episode running (for e.g. rock/paper/scissors)
     */
    template<class AGENT0, class AGENT1, class... CALLBACKS>
    class Runner {
    public:
        AGENT0                agent0;
        AGENT1                agent1;
        std::tuple<CALLBACKS...> callbacks;


        template<class FIRSTAGENT, class SECONDAGENT>
        Runner(FIRSTAGENT &&firstMover, SECONDAGENT &&secondMover, CALLBACKS &&...callbacks):
                agent0(std::forward<AGENT0>(firstMover)),
                agent1(std::forward<AGENT1>(secondMover)),
                callbacks(std::forward<CALLBACKS>(callbacks)...) { }

         /** Runs a number of episodes asynchronously (i.e. agents take turns to send messages)
          *
          * @param nEpisodes number of epsiodes to run
          */
        void runAsync() {
            callback(events::StartEpisode{agent0, agent1}, callbacks);
            passMessageRightAndRecurse(agent0.startEpisode());
            callback(events::EndEpisode{agent0, agent1}, agent0, agent1, callbacks);
        }

        /** Runs a number of episodes asynchronously (i.e. agents take turns to send messages)
         *
         * @param nEpisodes number of epsiodes to run
         */
        void runSync() {
            callback(events::StartEpisode{agent0, agent1}, callbacks);
            passMessagesSynchronously(agent1.startEpisode(), agent0.startEpisode());
            callback(events::EndEpisode{agent0, agent1}, agent0, agent1, callbacks);
        }

    protected:
        template<class MESSAGE>
        void passMessageRightAndRecurse(MESSAGE message) {
            if(isEmptyOptional(message)) return;
            callback(events::Message{agent0,agent1,valueIfOptional(message)},callbacks);
            passMessageLeftAndRecurse(agent1.handleMessage(std::move(valueIfOptional(message)))); // tail-recursion will be optimised out (if optimisation is on)
        }

        template<class MESSAGE>
        void passMessageLeftAndRecurse(MESSAGE message) {
            if(isEmptyOptional(message)) return;
            callback(events::Message{agent1,agent0,valueIfOptional(message)},callbacks);
            passMessageRightAndRecurse(agent0.handleMessage(std::move(valueIfOptional(message))));
        }

        template<class MESSAGE0, class MESSAGE1>
        void passMessagesSynchronously(MESSAGE0 messageFor0, MESSAGE1 messageFor1) {
            if(isEmptyOptional(messageFor0) || isEmptyOptional(messageFor1)) return;
            callback(events::Message{agent0,agent1,valueIfOptional(messageFor1)},callbacks);
            callback(events::Message{agent1,agent0,valueIfOptional(messageFor0)},callbacks);
            passMessagesSynchronously(
                    agent1.handleMessage(std::move(valueIfOptional(messageFor1))),
                    agent0.handleMessage(std::move(valueIfOptional(messageFor0))));
        }

        template<class T>
        inline static bool isEmptyOptional(const std::optional<T> &x) { return !x.has_value(); }

        template<class T>
        inline static bool isEmptyOptional(const T &x) { return false; }

        template<class T>
        inline static T &valueIfOptional(std::optional<T> &opt) { return opt.value(); }

        template<class T>
        inline static const T &valueIfOptional(const std::optional<T> &opt) { return opt.value(); }

        template<class T>
        inline static T valueIfOptional(T &&obj) { return std::forward<T>(obj); }

    };
    // Allow universal reference passing (runner will move rvalue refs and reference lvalue refs)
    template<class AGENT0, class AGENT1, class...CALLBACKS>
    Runner(AGENT0 &&agent0, AGENT1 &&agent1, CALLBACKS &&...callbacks) -> Runner<AGENT0,AGENT1,CALLBACKS...>;

    /** Execute a turns-based episode between two agents
     */
    template<class AGENT0, class AGENT1, class... CALLBACKS>
    inline void runAsync(AGENT0 &&agent0, AGENT1 &&agent1, CALLBACKS &&...callbacks) {
        Runner(std::forward<AGENT0>(agent0), std::forward<AGENT1>(agent1), std::forward<CALLBACKS>(callbacks)...).runAsync();
    }

    /** Execute a synchronous episode between two agents (i.e. agents swap messages at the same time)
     */
    template<class AGENT0, class AGENT1, class... CALLBACKS>
    inline void runSync(AGENT0 &&agent0, AGENT1 &&agent1, CALLBACKS &&...callbacks) {
        Runner(std::forward<AGENT0>(agent0), std::forward<AGENT1>(agent1), std::forward<CALLBACKS>(callbacks)...).runSync();
    }
}

#endif //MULTIAGENTGOVERNMENT_SIMPLEEPISODE_H
