// A runner for episodes between two agents.
// TODO: An episode should conisit of
//  AGENT1                          AGENT2
//  firstMoverStartEpisode()
//                                  secondMoverStartEpisode(Agent1body)
//                                  handleMessage(..)
//  handleMessage(..)               ...
//                                  handleMEssage(..)
//  otherPlayerEndedEpisode()
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

    template<class SOURCE, class DEST, class MESSAGE>
    struct LeftMessage : Message<SOURCE,DEST,MESSAGE> {
        friend std::ostream &operator <<(std::ostream &out, const LeftMessage<SOURCE,DEST,MESSAGE> &event) {
            out << "<-- ";
            deselby::invoke_if_invocable(deselby::streamoperator, out, event.message);
            return out;
        }
    };
    template<class SOURCE, class DEST, class MESSAGE> LeftMessage(SOURCE &, DEST &,
                                                              MESSAGE &) -> LeftMessage<SOURCE, DEST, MESSAGE>;

    template<class SOURCE, class DEST, class MESSAGE>
    struct RightMessage : Message<SOURCE,DEST,MESSAGE> {
        friend std::ostream &operator <<(std::ostream &out, const RightMessage<SOURCE,DEST,MESSAGE> &event) {
            out << "--> ";
            deselby::invoke_if_invocable(deselby::streamoperator, out, event.message);
            return out;
        }
    };
    template<class SOURCE, class DEST, class MESSAGE> RightMessage(SOURCE &, DEST &,
                                                                  MESSAGE &) -> RightMessage<SOURCE, DEST, MESSAGE>;

    template<class AGENT1, class AGENT2>
    struct StartEpisode {
        AGENT1 &agent1;
        AGENT2 &agent2;

        friend std::ostream &operator <<(std::ostream &out, const StartEpisode<AGENT1,AGENT2> &event) {
            out << "------- Starting episode -------" << std::endl;
            deselby::invoke_if_invocable(deselby::streamoperator, out, event.agent1);
            deselby::invoke_if_invocable(deselby::streamoperator, out, event.agent2);
            return out;
        }
    };
    template<class AGENT1, class AGENT2> StartEpisode(AGENT1 &, AGENT2 &) -> StartEpisode<AGENT1, AGENT2>;

    template<class AGENT1, class AGENT2>
    struct EndEpisode {
        AGENT1 &agent1;
        AGENT2 &agent2;

        friend std::ostream &operator <<(std::ostream &out, const EndEpisode<AGENT1,AGENT2> &event) {
            deselby::invoke_if_invocable(deselby::streamoperator, out, event.agent1);
            deselby::invoke_if_invocable(deselby::streamoperator, out, event.agent2);
            out << "-------    End episode   -------" << std::endl;
            return out;
        }
    };
    template<class AGENT1, class AGENT2> EndEpisode(AGENT1 &, AGENT2 &) -> EndEpisode<AGENT1, AGENT2>;
}


namespace abm::callbacks {
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
     *   - LeftMessage
     *   - RightMessage
     *   - EndEpisode
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


         /** Runs a number of episodes asynchronously (i.e. agents take turns to send messages) */
        void runAsync() {
            events::StartEpisode startEpisodeEvent(agent0,agent1);
            callback(startEpisodeEvent, agent0); // guarantee that agent0 gets message before agent1
            callback(startEpisodeEvent, agent1);
            callback(startEpisodeEvent, callbacks);
            passMessagesAsync(agent0.startEpisode());
            callback(events::EndEpisode{agent0, agent1}, agent0, agent1, callbacks);
        }


        /** Runs a number of episodes asynchronously (i.e. agents take turns to send messages) */
        void runSync() {
            events::StartEpisode startEpisodeEvent(agent0,agent1);
            callback(startEpisodeEvent, agent0); // guarantee that agent0 gets message before agent1
            callback(startEpisodeEvent, agent1);
            callback(startEpisodeEvent, callbacks);
            passMessagesSynchronously(agent1.startEpisode(), agent0.startEpisode());
            callback(events::EndEpisode{agent0, agent1}, agent0, agent1, callbacks);
        }


        template<class MESSAGE>
        void passMessagesAsync(MESSAGE messageFor1) {
            if(deselby::isEmptyOptional(messageFor1)) return;
            callback(events::RightMessage{agent0,agent1,deselby::valueIfOptional(messageFor1)},callbacks);
            auto messageFor0 = agent1.handleMessage(std::move(deselby::valueIfOptional(messageFor1)));
            if(deselby::isEmptyOptional(messageFor0)) return;
            callback(events::LeftMessage{agent1,agent0,deselby::valueIfOptional(messageFor0)},callbacks);
            // not exactly recursion as MESSAGE type may be different, but tail-call optimisation should prevent stack overflow
            passMessagesAsync(agent0.handleMessage(std::move(deselby::valueIfOptional(messageFor0))));
        }


        template<class MESSAGE0, class MESSAGE1>
        void passMessagesSynchronously(MESSAGE0 messageFor0, MESSAGE1 messageFor1) {
            if(deselby::isEmptyOptional(messageFor0) || deselby::isEmptyOptional(messageFor1)) return;
            callback(events::RightMessage{agent0,agent1,deselby::valueIfOptional(messageFor1)},callbacks);
            callback(events::LeftMessage{agent1,agent0,deselby::valueIfOptional(messageFor0)},callbacks);
            passMessagesSynchronously(
                    agent1.handleMessage(std::move(deselby::valueIfOptional(messageFor1))),
                    agent0.handleMessage(std::move(deselby::valueIfOptional(messageFor0)))); // tail-call will be optimised out (if optimisation is on)
        }

//        template<class T>
//        inline static bool isEmptyOptional(const std::optional<T> &x) { return !x.has_value(); }
//
//        template<class T>
//        inline static bool isEmptyOptional(const std::nullopt_t &x) { return true; }
//
//        template<class T>
//        inline static bool isEmptyOptional(const T &x) { return false; }
//
//        template<class T>
//        inline static T &valueIfOptional(std::optional<T> &opt) { return opt.value(); }
//
//        template<class T>
//        inline static const T &valueIfOptional(const std::optional<T> &opt) { return opt.value(); }
//
//        template<class T>
//        inline static T valueIfOptional(T &&obj) { return std::forward<T>(obj); }

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

    /** If this body/mind pair is a monad (i.e. body never sends outgoing messages) then the
     * agent can run a complete episode on its own */
    template<class AGENT, class... CALLBACKS>
    inline void runEpisode(AGENT &&agent, CALLBACKS &&... callbacks) {
        agent.runEpisode(callbacks...);
    }
}

#endif //MULTIAGENTGOVERNMENT_SIMPLEEPISODE_H
