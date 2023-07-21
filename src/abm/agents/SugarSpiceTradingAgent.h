//
// Created by daniel on 05/06/23.
//

#ifndef MULTIAGENTGOVERNMENT_SUGARSPICETRADINGAGENT_H
#define MULTIAGENTGOVERNMENT_SUGARSPICETRADINGAGENT_H

#include <deque>
#include <bitset>
#include "../abm.h"
#include "../../DeselbyStd/random.h"
#include "mlpack.hpp"

#include "SugarSpiceTradingBody.h"
#include "../QAgent.h"
#include "../DQN.h"
#include "../QTable.h"

namespace abm::agents {

//    template<bool HASLANGUAGE>
//    class SugarSpiceTradingAgent: public QAgent<SugarSpiceTradingBody<HASLANGUAGE>, SugarSpiceQFunction> {
//    public:
//        typedef QAgent<SugarSpiceTradingBody<HASLANGUAGE>, SugarSpiceQFunction>::body_type      body_type;
//        typedef QAgent<SugarSpiceTradingBody<HASLANGUAGE>, SugarSpiceQFunction>::intent_type   intent_type;
//
//
////        SugarSpiceTradingAgent():
////        QAgent<SugarSpiceTradingBody<HASLANGUAGE>, DQN>(DQN(body_type::dimension, 64, 32, intent_type::size,  32, 256, 1.0)) {}
//        SugarSpiceTradingAgent():
//                QAgent<SugarSpiceTradingBody<HASLANGUAGE>, QTable<body_type::nstates, intent_type::size>>(QTable<body_type::nstates, intent_type::size>()) {}
//    };

//        template<bool HASLANGUAGE>
//        class SugarSpiceTradingAgent {
//        public:
//            typedef uint time_type;
//            typedef abm::Schedule<time_type> schedule_type;
//            static inline bool verboseMode = false;
//
//            enum MessageEnum {
//                // Agent actions
//                GiveSugar,
//                GiveSpice,
//                WalkAway,
//                Fight,
//                // optional verbal acts
//                Say0,
//                Say1,
//                // Out of band comms: terminal states
//                YouLostFight,
//                YouWonFight,
//                Bandits
////                EndByMutualConsent
//            };
//
//
//            // message_type class so that this is a valid mlpack Environment
//            typedef abm::MlPackAction<MessageEnum, 4 + 2 * HASLANGUAGE> message_type;
//
//            class State {
//            public:
////                static const int conversationHistoryLength = 1;
//                static const int utilityOfPreferred = 10;      // utility of holding sugar/spice
//                static const int utilityOfNonPreferred = 1;
//                static const bool rememberOutgoingMessage = false;
//
//                // Agent state
//                arma::colvec netInput; // state in the form of one-hot vectors of action history, plus sugar, spice and preference
//
//                State() : netInput(dimension) {}
//
//                // perform a state transition given an act/response pair
//                // in a binary transaction
//                double transition(message_type myLastAction, message_type yourResponse);
//
//                std::bitset<message_type::size> legalIntents();
//
//                double &sugar() { return netInput[0]; }
//
//                double &spice() { return netInput[1]; }
//
//                double &prefersSugar() { return netInput[2]; }
//
//                double sugar() const { return netInput[0]; }
//
//                double spice() const { return netInput[1]; }
//
//                double prefersSugar() const { return netInput[2]; }
//
//                const arma::colvec &Encode() const { return netInput; }
//
//                // convert to integer giving the ordinal of this state
//                operator int() const {
//                    return netInput[0] + 2 * netInput[1] + 4 * netInput[2] + 8 * (getLastIncomingMessage() +
//                                                                                  (rememberOutgoingMessage ?
//                                                                                   message_type::size *
//                                                                                           getLastOutgoingMessage()
//                                                                                                           : 0));
//                }
//
////                void recordSpeechAct(double word, bool isOtherPlayer) {
////                    int bufferStart = isOtherPlayer?0:conversationHistoryLength;
////                    arma::colvec truncatedHistory = netInput.subvec(bufferStart, bufferStart + conversationHistoryLength-2);
////                    netInput.subvec(bufferStart + 1, bufferStart + conversationHistoryLength-1) = truncatedHistory;
////                    netInput[bufferStart] = word;
////                }
//
//                void recordIncomingMessage(MessageEnum message) {
//                    for (int i = 3; i < 3 + message_type::size; ++i) netInput[i] = 0.0;
//                    netInput[3 + message] = 1.0;
//                }
//
//                void recordOutgoingMessage(MessageEnum message) {
//                    if (rememberOutgoingMessage) {
//                        for (int i = 3 + message_type::size; i < 3 + 2 * message_type::size; ++i) netInput[i] = 0.0;
//                        netInput[3 + message_type::size + message] = 1.0;
//                    }
//                }
//
//                MessageEnum getLastIncomingMessage() const {
//                    int m = 0;
//                    while (netInput[3 + m] == 0.0) ++m;
//                    return static_cast<MessageEnum>(m);
//                }
//
//                MessageEnum getLastOutgoingMessage() const {
//                    assert(rememberOutgoingMessage);
//                    int m = 0;
//                    while (netInput[3 + message_type::size + m] == 0.0) ++m;
//                    return static_cast<MessageEnum>(m);
//                }
//
//
//                void reset(bool hasSugar, bool hasSpice, bool prefersSugar);
//
//                double utility() const;
//
//                friend std::ostream &operator<<(std::ostream &out, const SugarSpiceTradingAgent::State &state) {
////                    out << state.sugar() << state.spice() << state.prefersSugar();
//                    out << state.netInput.t();
//                    return out;
//                }
//
//                static constexpr size_t dimension = message_type::size * (1 + rememberOutgoingMessage) + 3;
//                static constexpr size_t nstates = 8 * message_type::size * (rememberOutgoingMessage ? message_type::size : 1);
//
//            };
//
////            static constexpr double initialCostOfVerbosity = 0.05;
////            static constexpr double deltaCostOfVerbosity = 0.25; // change in cost of verbosity per agent action
////            static constexpr double kappaCostOfVerbosity = 1.13; // change in cost of verbosity per agent action
////            double costOfVerbosity = initialCostOfVerbosity;
//            static constexpr double costOfFighting = 1.5;
//            static constexpr double costOfBanditAttack = 15.0;
//            inline static double pBanditAttack = 0.02; // probability of a bandit attack per message received
//
//            State state;
//            State stateBeforeLastAction;
//            std::optional<MessageEnum> myLastAction;
////            DQNPolicy<message_type::size> policy = DQNPolicy<message_type::size>(State::dimension, 64, 32, 32, 256, 1.0);
//            DQNPolicy<SugarSpiceTradingAgent<HASLANGUAGE>> policy = DQNPolicy<SugarSpiceTradingAgent<HASLANGUAGE>>(State::dimension, 64, 32, 32, 256, 1.0);
////            QTablePolicy<State::nstates, message_type::size> policy = QTablePolicy<State::nstates, message_type::size>(1.0, 0.5, std::pow(0.02, 1.0/500000.0), 0.01);
//
//            CommunicationChannel<Schedule<time_type>, MessageEnum> otherPlayer;
//
////            SugarSpiceTradingAgent():
////                    policy(State::dimension, 100,50,64,100000,1.0) {}
//
//            void connectTo(SugarSpiceTradingAgent &otherAgent) {
//                otherPlayer.connectTo(otherAgent, &SugarSpiceTradingAgent::handleTradingAct, 1);
//            }
//
//
//            Schedule<time_type> start(time_type time = 0) {
//                MessageEnum message = makeMove(false);
//                return otherPlayer.send(message, time);
//            }
//
//            void reset(bool hasSugar, bool hasSpice, bool prefersSugar) {
//                state.reset(hasSugar, hasSpice, prefersSugar);
//                myLastAction.reset();
////                costOfVerbosity = initialCostOfVerbosity;
//            }
//
//            // TODO: At the momemnt, the reward structure is hidden. Can we change the handler structure
//            //  to a standard form that brings rewards to the surface?
//            //  - there's the rewad function, which takes state-before-last-decision, action (outgoing channel/message)
//            //  and state-before-next-decision
//            //  - then there's state transition on receipt of a message on a channel
//            //  - then there's the policy from state to act, which decides next message to send
//            //  - then there's state transition on sending of a message
//            //  So, the cycle for an agent interaction on a channel:
//            //    outMessage = policy(state)
//            //    startState = state
//            //    state = outgoingMessageStateTransition(state, outMessage, channel)
//            //    send(outMessage)
//            //     ... [possible state transitions from other channels or episode is atomic?] ...
//            //    inMessage = handleMessage()
//            //    state = incomingMessageStateTransition(state, inMessage, channel)
//            //    reward = Reward(startState, outMessage, state)
//            //    train(startState, outMessage, reward, state)
//            Schedule<time_type> handleTradingAct(MessageEnum opponentMessage, time_type time) {
//                bool isEnd = false;
//                double costs = 0.0;
//                switch (opponentMessage) {
//                    case GiveSugar:
//                        if (verboseMode) std::cout << "Give sugar" << std::endl;
//                        state.sugar() += 1;
//                        state.recordIncomingMessage(GiveSugar);
//                        break;
//                    case GiveSpice:
//                        if (verboseMode) std::cout << "Give spice" << std::endl;
//                        state.spice() += 1;
//                        state.recordIncomingMessage(GiveSpice);
//                        break;
//                    case WalkAway:
//                        if (verboseMode) std::cout << "Walk away" << std::endl;
//                        state.recordIncomingMessage(WalkAway);
//                        break;
//                    case Say0:
//                        if (verboseMode) std::cout << "Say ogg" << std::endl;
//                        state.recordIncomingMessage(Say0);
//                        break;
//                    case Say1:
//                        if (verboseMode) std::cout << "Say igg" << std::endl;
//                        state.recordIncomingMessage(Say1);
//                        break;
//
//                    case YouLostFight:
//                        if (verboseMode) std::cout << "Fight, agressor wins" << std::endl;
//                        if (state.sugar() > 0) state.sugar() = 0;
//                        if (state.spice() > 0) state.spice() = 0;
//                        isEnd = true;
//                        costs = costOfFighting;
//                        break;
//                    case YouWonFight:
//                        if (verboseMode) std::cout << "Fight, agressor loses" << std::endl;
//                        state.sugar() = 1;
//                        state.spice() = 1;
//                        isEnd = true;
//                        costs = costOfFighting;
//                        break;
//                    case Bandits:
//                        if (verboseMode) std::cout << "Bandits!" << std::endl;
//                        if (state.sugar() > 0) state.sugar() = 0;
//                        if (state.spice() > 0) state.spice() = 0;
//                        isEnd = true;
//                        costs = costOfBanditAttack;
//                        break;
//                    case EndByMutualConsent:
//                        if (verboseMode) std::cout << "Walk away" << std::endl;
//                        isEnd = true;
//                        break;
//                }
//                if (myLastAction.has_value()) { // train on time from immediately before last action to immediately before next action
//                    double reward = state.utility() - stateBeforeLastAction.utility() - costs;
//                    policy.train(stateBeforeLastAction, myLastAction.value(), reward, state, isEnd);
//                }
//
//                if (isEnd) return Schedule<time_type>();
//
//                MessageEnum myMessage = makeMove(opponentMessage == WalkAway);
//                return otherPlayer.send(myMessage, time);
//            }
//
//
//            MessageEnum makeMove(bool opponentMessageIsWalkaway) {
//                MessageEnum message;
//                stateBeforeLastAction = state;
//                myLastAction = static_cast<MessageEnum>(policy.getAction(state));
//                if (deselby::Random::nextBool(pBanditAttack)) {
//                    message = Bandits;
//                } else if ((state.sugar() < 1.0 && myLastAction == GiveSugar) ||
//                           (state.spice() < 1.0 && myLastAction == GiveSpice)) {
//                    message = WalkAway;
//                } else if (myLastAction == Fight) {
//                    message = deselby::Random::nextBool(0.5) ? YouWonFight : YouLostFight;
//                } else {
//                    message = myLastAction.value();
//                }
//
//                if (opponentMessageIsWalkaway && message == WalkAway) message = EndByMutualConsent;
//
//                double costs = 0.0;
//                bool isEnd = false;
//                switch (message) {
//                    case GiveSugar:
//                        state.sugar() -= 1;
//                        break;
//                    case GiveSpice:
//                        state.spice() -= 1;
//                        break;
//                    case Say0:
//                        break;
//                    case Say1:
//                        break;
//                    case YouWonFight:
//                        // I started fight and other won
//                        state.sugar() = 0;
//                        state.spice() = 0;
//                        isEnd = true;
//                        costs = costOfFighting;
//                        break;
//                    case YouLostFight:
//                        // I started fight and other lost
//                        state.sugar() = 1;
//                        state.spice() = 1;
//                        isEnd = true;
//                        costs = costOfFighting;
//                        break;
//                    case Bandits:
//                        state.sugar() = 0;
//                        state.spice() = 0;
//                        isEnd = true;
//                        costs = costOfBanditAttack;
//                        break;
//                    case EndByMutualConsent:
//                        isEnd = true;
//                        break;
//                    default:
//                        break;
//                }
//                if (isEnd) {
//                    double reward = state.utility() - stateBeforeLastAction.utility() - costs;
//                    policy.train(stateBeforeLastAction, myLastAction.value(), reward, state, true);
//                }
//                state.recordOutgoingMessage(myLastAction.value());
//                return message;
//            };
//
//
//
//            static bool isTerminal(message_type act, message_type response) {
//                return (response == Bandits ||
//                        response == YouLostFight ||
//                        response == YouWonFight ||
//                        (act == WalkAway && response == WalkAway));
//            }
//        };
//
//        template<bool HASLANGUAGE>
//        std::bitset<SugarSpiceTradingAgent<HASLANGUAGE>::message_type::size> SugarSpiceTradingAgent<HASLANGUAGE>::State::legalIntents() {
//            std::bitset<SugarSpiceTradingAgent<HASLANGUAGE>::message_type::size> legalActs;
//            if(sugar() == 0) legalActs[GiveSugar] = false;
//            if(spice() == 0) legalActs[GiveSpice] = false;
//            if(getLastIncomingMessage() == Fight) {
//                legalActs = 0;
//                legalActs[YouLostFight] = true;
//                legalActs[YouWonFight] = true;
//            }
//            return legalActs;
//        }
//
//        template<bool HASLANGUAGE>
//        double SugarSpiceTradingAgent<HASLANGUAGE>::State::utility() const {
//            double utilityOfSugar = (prefersSugar() > 0.5 ? utilityOfPreferred : utilityOfNonPreferred);
//            double utilityOfSpice = (prefersSugar() > 0.5 ? utilityOfNonPreferred : utilityOfPreferred);
//            return sugar() * utilityOfSugar + spice() * utilityOfSpice;
//        }
//
//        template<bool HASLANGUAGE>
//        void SugarSpiceTradingAgent<HASLANGUAGE>::State::reset(bool hasSugar, bool hasSpice, bool prefersSugar) {
//            recordIncomingMessage(Fight); // use Fight to mean Empty as we don't use this
//            recordOutgoingMessage(Fight);
//            sugar() = hasSugar;
//            spice() = hasSpice;
//            this->prefersSugar() = prefersSugar;
//        }
//
//        template<bool HASLANGUAGE>
//        double SugarSpiceTradingAgent<HASLANGUAGE>::State::transition(SugarSpiceTradingAgent::message_type myLastAction,
//                                                                      SugarSpiceTradingAgent::message_type yourResponse) {
//            double initialUtility = utility();
//            double costs = 0.0;
//            recordOutgoingMessage(myLastAction);
//            switch (myLastAction) {
//                case GiveSugar:
//                    sugar() -= 1;
//                    break;
//                case GiveSpice:
//                    spice() -= 1;
//                    break;
//                case Fight:
//                    costs = costOfFighting;
//                    break;
//                case YouWonFight:
//                    // You started fight and won
//                    sugar() = 0;
//                    spice() = 0;
//                    break;
//                case YouLostFight:
//                    // You started fight and lost
//                    sugar() = 1;
//                    spice() = 1;
//                    break;
//                case Bandits:
//                    sugar() = 0;
//                    spice() = 0;
//                    costs = costOfBanditAttack;
//                    break;
//                default:
//                    break;
//            }
//
//            recordIncomingMessage(yourResponse);
//            switch (yourResponse) {
//                case GiveSugar:
//                    sugar() += 1;
//                    break;
//                case GiveSpice:
//                    spice() += 1;
//                    break;
//                case Fight:
//                    costs = costOfFighting;
//                    break;
//                case YouWonFight:
//                    // I started fight and won
//                    sugar() = 1;
//                    spice() = 1;
//                    break;
//                case YouLostFight:
//                    // I started fight and lost
//                    sugar() = 0;
//                    spice() = 0;
//                    break;
//                case Bandits:
//                    sugar() = 0;
//                    spice() = 0;
//                    costs += costOfBanditAttack;
//                    break;
//                default:
//                    break;
//            }
//            double reward = utility() - initialUtility - costs;
//            return reward;
//        }

}


#endif //MULTIAGENTGOVERNMENT_SUGARSPICETRADINGAGENT_H
