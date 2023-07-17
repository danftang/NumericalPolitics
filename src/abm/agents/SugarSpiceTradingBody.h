//
// Created by daniel on 17/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_SUGARSPICETRADINGBODY_H
#define MULTIAGENTGOVERNMENT_SUGARSPICETRADINGBODY_H

#include <deque>
#include <bitset>
#include "../abm.h"
#include "../../DeselbyStd/random.h"
#include "mlpack.hpp"

namespace abm::agents {

        template<bool HASLANGUAGE>
        class SugarSpiceTradingBody {
        public:
            static inline bool verboseMode = false;

            enum MessageEnum {
                // Agent actions
                GiveSugar,
                GiveSpice,
                WalkAway,
                YouWonFight = WalkAway,
                Fight,
                YouLostFight = Fight,
                Bandits,
                // optional verbal acts
                Say0,
                Say1,
                size  // marker for number of actions
            };


            // Action class so that this is a valid mlpack Environment
            class Action: public abm::MlPackAction<MessageEnum, MessageEnum::size - 2 * HASLANGUAGE> {
            public:
                static bool isTerminal(Action act, Action response) {
                    return (response == Bandits ||
                            act == Fight ||
                            (act == WalkAway && response == WalkAway));
                }

            };

            class State {
            public:
//                static const int conversationHistoryLength = 1;
                static const int utilityOfPreferred = 10;      // utility of holding sugar/spice
                static const int utilityOfNonPreferred = 1;
                static constexpr double costOfFighting = 1.5;
                static constexpr double costOfBanditAttack = 15.0;
                inline static double pBanditAttack = 0.02; // probability of a bandit attack per message received
                static const bool rememberOutgoingMessage = false;

                // Agent state
                arma::colvec netInput; // state in the form of one-hot vectors of action history, plus sugar, spice and preference

                State() : netInput(dimension) {}

                // perform a state transition given an act/response pair
                // in a binary transaction
                double transition(Action myLastAction, Action yourResponse);
                double terminalHalfTransition(Action yourAction, Action myResponse);

                std::bitset<Action::size> legalActions();

                double &sugar() { return netInput[0]; }

                double &spice() { return netInput[1]; }

                double &prefersSugar() { return netInput[2]; }

                double sugar() const { return netInput[0]; }

                double spice() const { return netInput[1]; }

                double prefersSugar() const { return netInput[2]; }

                const arma::colvec &Encode() const { return netInput; }

                // convert to integer giving the ordinal of this state
                operator int() const {
                    return netInput[0] + 2 * netInput[1] + 4 * netInput[2] + 8 * (getLastIncomingMessage() +
                                                                                  (rememberOutgoingMessage ?
                                                                                   Action::size *
                                                                                   getLastOutgoingMessage()
                                                                                                           : 0));
                }

                void recordIncomingMessage(MessageEnum message) {
                    for (int i = 3; i < 3 + Action::size; ++i) netInput[i] = 0.0;
                    netInput[3 + message] = 1.0;
                }

                void recordOutgoingMessage(MessageEnum message) {
                    if (rememberOutgoingMessage) {
                        for (int i = 3 + Action::size; i < 3 + 2 * Action::size; ++i) netInput[i] = 0.0;
                        netInput[3 + Action::size + message] = 1.0;
                    }
                }

                MessageEnum getLastIncomingMessage() const {
                    int m = 0;
                    while (netInput[3 + m] == 0.0) ++m;
                    return static_cast<MessageEnum>(m);
                }

                MessageEnum getLastOutgoingMessage() const {
                    assert(rememberOutgoingMessage);
                    int m = 0;
                    while (netInput[3 + Action::size + m] == 0.0) ++m;
                    return static_cast<MessageEnum>(m);
                }

                void reset(bool hasSugar, bool hasSpice, bool prefersSugar);

                double utility() const;

                friend std::ostream &operator<<(std::ostream &out, const SugarSpiceTradingBody::State &state) {
//                    out << state.sugar() << state.spice() << state.prefersSugar();
                    out << state.netInput.t();
                    return out;
                }

                static constexpr size_t dimension = Action::size * (1 + rememberOutgoingMessage) + 3;
                static constexpr size_t nstates = 8 * Action::size * (rememberOutgoingMessage ? Action::size : 1);

            };
        };

        template<bool HASLANGUAGE>
        std::bitset<SugarSpiceTradingBody<HASLANGUAGE>::Action::size> SugarSpiceTradingBody<HASLANGUAGE>::State::legalActions() {
            std::bitset<SugarSpiceTradingBody<HASLANGUAGE>::Action::size> legalActs;
            if(sugar() == 0) legalActs[GiveSugar] = false;
            if(spice() == 0) legalActs[GiveSpice] = false;
            if(getLastIncomingMessage() == Fight) {
                legalActs = 0;
                legalActs[YouLostFight] = true;
                legalActs[YouWonFight] = true;
            }
            return legalActs;
        }

        template<bool HASLANGUAGE>
        double SugarSpiceTradingBody<HASLANGUAGE>::State::utility() const {
            double utilityOfSugar = (prefersSugar() > 0.5 ? utilityOfPreferred : utilityOfNonPreferred);
            double utilityOfSpice = (prefersSugar() > 0.5 ? utilityOfNonPreferred : utilityOfPreferred);
            return sugar() * utilityOfSugar + spice() * utilityOfSpice;
        }

        template<bool HASLANGUAGE>
        void SugarSpiceTradingBody<HASLANGUAGE>::State::reset(bool hasSugar, bool hasSpice, bool prefersSugar) {
            recordIncomingMessage(Fight); // use Fight to mean Empty as we don't use this
            recordOutgoingMessage(Fight);
            sugar() = hasSugar;
            spice() = hasSpice;
            this->prefersSugar() = prefersSugar;
        }

        template<bool HASLANGUAGE>
        double SugarSpiceTradingBody<HASLANGUAGE>::State::transition(Action myLastAction, Action yourResponse) {
            if(yourResponse == -1) return 0; // must be start of episode, reward will be ignored
            double initialUtility = utility();
            double costs = 0.0;
            recordOutgoingMessage(myLastAction);
            switch (myLastAction) {
                case GiveSugar:
                    sugar() -= 1;
                    break;
                case GiveSpice:
                    spice() -= 1;
                    break;
                case Fight:
                    costs = costOfFighting;
                    break;
                default:
                    break;
            }

            recordIncomingMessage(yourResponse);
            switch (yourResponse) {
                case GiveSugar:
                    sugar() += 1;
                    break;
                case GiveSpice:
                    spice() += 1;
                    break;
                case Fight:
                    costs = costOfFighting;
                    break;
                case YouWonFight:
                    // I started fight and won
                    sugar() = 1;
                    spice() = 1;
                    break;
                case YouLostFight:
                    // I started fight and lost
                    sugar() = 0;
                    spice() = 0;
                    break;
                case Bandits:
                    sugar() = 0;
                    spice() = 0;
                    costs += costOfBanditAttack;
                    break;
                default:
                    break;
            }
            double reward = utility() - initialUtility - costs;
            return reward;
        }

        // I ended the episode with a terminating action
        template<bool HASLANGUAGE>
        double SugarSpiceTradingBody<HASLANGUAGE>::State::terminalHalfTransition(Action yourAction, Action myResponse) {
            double initialUtility = utility();
            double costs = 0.0;
                recordOutgoingMessage(myResponse); // just for the record!
            if(myResponse == Bandits) {
                sugar() = 0;
                spice() = 0;
                costs = costOfBanditAttack;
            } else if(yourAction == Fight) {
                if(myResponse == YouWonFight) {
                    sugar() = 0;
                    spice() = 0;
                } else {
                    sugar() = 1;
                    spice() = 1;
                }
            }
            double reward = utility() - initialUtility - costs;
            return reward;
        }
    }




#endif //MULTIAGENTGOVERNMENT_SUGARSPICETRADINGBODY_H
