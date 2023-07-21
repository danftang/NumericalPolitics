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
        // The intent of the agent, these are the decisions the brain can take
        enum intent_type {
            iGiveSugar,
            iGiveSpice,
            iWalkAway,
            iFight,
            iSay0,
            iSay1,
            size,  // marker for number of actions
        };

        // the tyoe of message passed between agents in response to an agent's intent
        enum message_type {
            // Agent actions
            GiveSugar,
            GiveSpice,
            WalkAway,
            Say0,
            Say1,
            Bandits,
            YouWonFight,
            YouLostFight,
            close = -1
        };

//        template<bool HasLanguage>
        friend std::ostream &operator <<(std::ostream &out, const typename abm::agents::SugarSpiceTradingBody<HASLANGUAGE>::message_type &message) {
            static const std::string messageNames[] = {
                    "GiveSugar",
                    "GiveSpice",
                    "WalkAway",
                    "Say0",
                    "Say1",
                    "Bandits",
                    "YouWonFight",
                    "YouLostFight"
            };
            if(message == abm::agents::SugarSpiceTradingBody<HASLANGUAGE>::message_type::close) {
                out << "close";
            } else {
                out << messageNames[message];
            }
            return out;
        }


        //                static const int conversationHistoryLength = 1;
        static const int utilityOfPreferred = 10;      // utility of holding sugar/spice
        static const int utilityOfNonPreferred = 1;
        static constexpr double costOfFighting = 1.5;
        static constexpr double costOfBanditAttack = 15.0;
        inline static double pBanditAttack = 0.02; // probability of a bandit attack per message received
        static const bool rememberOutgoingMessage = false;
        static const int nOneHotBitsForMessageEncode = 6;

        // Agent state
        arma::colvec netInput; // state in the form of one-hot vectors of action history, plus sugar, spice and preference
        bool isTerminal;

        SugarSpiceTradingBody() : netInput(dimension), isTerminal(false) {}

        void reset(int nSugar, int nSpice, bool agentPrefersSugar) {
            netInput.zeros();
            isTerminal = false;
            sugar() = nSugar;
            spice() = nSpice;
            prefersSugar() = agentPrefersSugar;
        }

        // perform a state transition given an act/response pair
        // in a binary transaction
        double transition(message_type myLastAction, message_type yourResponse);

        static message_type intentToMessage(int intent) {
            if (deselby::Random::nextBool(pBanditAttack)) return Bandits;
            switch (intent) {
                case iGiveSugar:
                    return GiveSugar;
                case iGiveSpice:
                    return GiveSpice;
                case iWalkAway:
                    return WalkAway;
                case iFight:
                    return deselby::Random::nextBool() ? YouWonFight : YouLostFight;
                case iSay0:
                    return Say0;
                case iSay1:
                    return Say1;
            }
            throw("Unrecognized intent");
        }

        std::bitset<intent_type::size> legalIntents();

        double &sugar() { return netInput[0]; }

        double &spice() { return netInput[1]; }

        double &prefersSugar() { return netInput[2]; }

        double sugar() const { return netInput[0]; }

        double spice() const { return netInput[1]; }

        double prefersSugar() const { return netInput[2]; }

        //const arma::mat &Encode() const { return netInput; }

        operator const arma::mat &() const {
            return netInput;
        }

        // convert to integer giving the ordinal of this state
        operator int() const {
            return netInput[0] + 2 * netInput[1] + 4 * netInput[2] + 8 * (getLastIncomingMessage() +
                                                                          (rememberOutgoingMessage ?
                                                                           nOneHotBitsForMessageEncode *
                                                                           getLastOutgoingMessage()
                                                                                                   : 0));
        }

        void recordIncomingMessage(message_type message) {
            assert(message <= nOneHotBitsForMessageEncode);
            for (int i = 3; i < 3 + nOneHotBitsForMessageEncode; ++i) netInput[i] = 0.0;
            netInput[3 + message] = 1.0;
        }

        void recordOutgoingMessage(message_type message) {
            if (rememberOutgoingMessage) {
                assert(message <= nOneHotBitsForMessageEncode);
                for (int i = 3 + nOneHotBitsForMessageEncode; i < 3 + 2 * nOneHotBitsForMessageEncode; ++i)
                    netInput[i] = 0.0;
                netInput[3 + nOneHotBitsForMessageEncode + message] = 1.0;
            }
        }

        message_type getLastIncomingMessage() const {
            int m = 0;
            while (netInput[3 + m] == 0.0) ++m;
            return static_cast<message_type>(m);
        }

        message_type getLastOutgoingMessage() const {
            assert(rememberOutgoingMessage);
            int m = 0;
            while (netInput[3 + nOneHotBitsForMessageEncode + m] == 0.0) ++m;
            return static_cast<message_type>(m);
        }

        void reset(bool hasSugar, bool hasSpice, bool prefersSugar);

        double utility() const;

        friend std::ostream &operator<<(std::ostream &out, const SugarSpiceTradingBody &state) {
//                    out << state.sugar() << state.spice() << state.prefersSugar();
            out << state.netInput.t();
            return out;
        }

        static constexpr size_t dimension = nOneHotBitsForMessageEncode * (1 + rememberOutgoingMessage) + 3;
        static constexpr size_t nstates =
                8 * nOneHotBitsForMessageEncode * (rememberOutgoingMessage ? nOneHotBitsForMessageEncode : 1);
    };

    template<bool HASLANGUAGE>
    std::bitset<SugarSpiceTradingBody<HASLANGUAGE>::intent_type::size>
    SugarSpiceTradingBody<HASLANGUAGE>::legalIntents() {
        std::bitset<SugarSpiceTradingBody<HASLANGUAGE>::intent_type::size> legalActs;
        if (isTerminal) {
            legalActs = 0;
        } else {
            for (int i = 0; i < legalActs.size(); ++i) legalActs[i] = true;
            if (sugar() == 0) legalActs[iGiveSugar] = false;
            if (spice() == 0) legalActs[iGiveSpice] = false;
            legalActs[iSay0] = HASLANGUAGE;
            legalActs[iSay1] = HASLANGUAGE;
        }
        return legalActs;
    }

    template<bool HASLANGUAGE>
    double SugarSpiceTradingBody<HASLANGUAGE>::utility() const {
        double utilityOfSugar = (prefersSugar() > 0.5 ? utilityOfPreferred : utilityOfNonPreferred);
        double utilityOfSpice = (prefersSugar() > 0.5 ? utilityOfNonPreferred : utilityOfPreferred);
        return sugar() * utilityOfSugar + spice() * utilityOfSpice;
    }

    template<bool HASLANGUAGE>
    void SugarSpiceTradingBody<HASLANGUAGE>::reset(bool hasSugar, bool hasSpice, bool prefersSugar) {
        recordIncomingMessage(Bandits); // use first terminal state to mean Empty as we don't use this
        recordOutgoingMessage(Bandits);
        sugar() = hasSugar;
        spice() = hasSpice;
        this->prefersSugar() = prefersSugar;
    }

    template<bool HASLANGUAGE>
    double SugarSpiceTradingBody<HASLANGUAGE>::transition(message_type myLastAction, message_type yourResponse) {
        if (myLastAction == close && yourResponse == close) {
            // must be start of episode, reward will be ignored
            isTerminal = false;
            return 0.0;
        }
        isTerminal = (
                yourResponse == Bandits ||
                yourResponse == YouWonFight ||
                yourResponse == YouLostFight ||
                yourResponse == close ||
                (myLastAction == WalkAway && yourResponse == WalkAway)
        );
        double initialUtility = utility();
        double costs = 0.0;
        if (!isTerminal) {
            recordOutgoingMessage(myLastAction);
            recordIncomingMessage(yourResponse);
        }
        switch (myLastAction) {
            case GiveSugar:
                sugar() -= 1;
                break;
            case GiveSpice:
                spice() -= 1;
                break;
            case YouWonFight:
                // I started fight but lost
                costs = costOfFighting;
                sugar() = 0;
                spice() = 0;
                break;
            case YouLostFight:
                // I started fight but won
                costs = costOfFighting;
                sugar() = 1;
                spice() = 1;
                break;
            case Bandits:
                costs += costOfBanditAttack;
                sugar() = 0;
                spice() = 0;
                break;
            default:
                break;
        }

        switch (yourResponse) {
            case GiveSugar:
                sugar() += 1;
                break;
            case GiveSpice:
                spice() += 1;
                break;
            case YouWonFight:
                // You started fight and I won
                costs = costOfFighting;
                sugar() = 1;
                spice() = 1;
                break;
            case YouLostFight:
                // You started fight and I lost
                costs = costOfFighting;
                sugar() = 0;
                spice() = 0;
                break;
            case Bandits:
                costs += costOfBanditAttack;
                sugar() = 0;
                spice() = 0;
                break;
            default:
                break;
        }
        double reward = utility() - initialUtility - costs;
        return reward;
    }


}






#endif //MULTIAGENTGOVERNMENT_SUGARSPICETRADINGBODY_H
