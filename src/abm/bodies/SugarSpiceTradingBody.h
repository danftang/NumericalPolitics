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
        enum action_type {
            iGiveSugar,
            iGiveSpice,
            iWalkAway,
            iFight,
            iSay0,
            iSay1,
            size  // marker for number of actions
        };

        // the tyoe of message passed between agents in response to an agent's intent
        enum message_type {
            close = -1,
            // Agent actions
            GiveSugar,
            GiveSpice,
            WalkAway,
            Say0,
            Say1,
            Bandits, // Terminal messages should all be at the end
            YouWonFight,
            YouLostFight,
            IndeterminateTerminalMessage = Bandits
        };

        static const int utilityOfPreferred = 10;      // utility of holding sugar/spice
        static const int utilityOfNonPreferred = 1;
        static constexpr double costOfFighting = 1.5;
        static constexpr double costOfBanditAttack = 15.0;
        inline static double pBanditAttack = 0.02; // probability of a bandit attack per message received
        static const bool encodeOutgoingMessage = false;
        static const int nOneHotBitsForMessageEncode = 6;
        static constexpr size_t dimension = nOneHotBitsForMessageEncode * (1 + encodeOutgoingMessage) + 3;
        static constexpr size_t nstates =
                8 * nOneHotBitsForMessageEncode * (encodeOutgoingMessage ? nOneHotBitsForMessageEncode : 1);

        // Agent state
        arma::mat::fixed<dimension, 1> netInput; // state in the form of one-hot vectors of action history, plus sugar, spice and preference
        bool isTerminal = false;
        double utilityBeforeLastAct = NAN;
        message_type lastOutgoingMessage;


        // ----- Body interface -----

        message_type actToMessage(int action);

        double messageToReward(message_type incomingMessage);

        std::bitset<action_type::size> legalActs();

        // ---- End of Body interface

//        double endEpisode() {
//            isTerminal = false;
//            return utility() - utilityBeforeLastAct;
//        }

        double &sugar() { return netInput[0]; }

        double &spice() { return netInput[1]; }

        double &prefersSugar() { return netInput[2]; }

        double sugar() const { return netInput[0]; }

        double spice() const { return netInput[1]; }

        double prefersSugar() const { return netInput[2]; }

        //const arma::mat &Encode() const { return netInput; }

        operator const arma::mat::fixed<dimension,1> &() const {
            return netInput;
        }

        // convert to integer giving the ordinal of this state
        operator int() const {
            return netInput[0] + 2 * netInput[1] + 4 * netInput[2] + 8 * (getLastIncomingMessage() +
                                                                          (encodeOutgoingMessage ?
                                                                           nOneHotBitsForMessageEncode *
                                                                           getLastOutgoingMessage()
                                                                                                 : 0));
        }

        void recordIncomingMessage(message_type message) {
            assert(message >= 0);
            if(message > IndeterminateTerminalMessage) message = IndeterminateTerminalMessage; // don't record type of terminal messages
            for (int i = 3; i < 3 + nOneHotBitsForMessageEncode; ++i) netInput[i] = 0.0;
            netInput[3 + message] = 1.0;
        }

        void recordOutgoingMessage(message_type message) {
            if (encodeOutgoingMessage) {
                assert(message >= 0);
                if(message > IndeterminateTerminalMessage) message = IndeterminateTerminalMessage;
                for (int i = 3 + nOneHotBitsForMessageEncode; i < 3 + 2 * nOneHotBitsForMessageEncode; ++i)
                    netInput[i] = 0.0;
                netInput[3 + nOneHotBitsForMessageEncode + message] = 1.0;
            } else {
                lastOutgoingMessage = message;
            }
        }

        message_type getLastIncomingMessage() const {
            int m = 0;
            while (netInput[3 + m] == 0.0) ++m;
            return static_cast<message_type>(m);
        }

        message_type getLastOutgoingMessage() const {
            if(encodeOutgoingMessage) {
                int m = 0;
                while (netInput[3 + nOneHotBitsForMessageEncode + m] == 0.0) ++m;
                return static_cast<message_type>(m);
            }
            return lastOutgoingMessage;
        }

        void reset(bool hasSugar, bool hasSpice, bool prefersSugar);

        double utility() const;

//        friend std::ostream &operator <<(std::ostream &out, const typename abm::agents::SugarSpiceTradingBody<HASLANGUAGE>::message_type &message);

        friend std::ostream &operator<<(std::ostream &out, const SugarSpiceTradingBody &state) {
//                    out << state.sugar() << state.spice() << state.prefersSugar();
            out << state.netInput.t();
            return out;
        }

        friend std::ostream &operator<<(std::ostream &out, const typename SugarSpiceTradingBody<HASLANGUAGE>::message_type &message) {
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

    };


    template<bool HASLANGUAGE>
    std::bitset<SugarSpiceTradingBody<HASLANGUAGE>::action_type::size>
    SugarSpiceTradingBody<HASLANGUAGE>::legalActs() {
        std::bitset<SugarSpiceTradingBody<HASLANGUAGE>::action_type::size> legalActs;
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
        recordIncomingMessage(IndeterminateTerminalMessage); // use first terminal state to mean Empty as we don't use this
        recordOutgoingMessage(IndeterminateTerminalMessage);
        sugar() = hasSugar;
        spice() = hasSpice;
        this->prefersSugar() = prefersSugar;
        isTerminal = false;
        utilityBeforeLastAct = utility();
    }


    template<bool HASLANGUAGE>
    SugarSpiceTradingBody<HASLANGUAGE>::message_type SugarSpiceTradingBody<HASLANGUAGE>::actToMessage(int action) {
//        if (myLastAction == close && yourResponse == close) {
//            // must be start of episode, reward will be ignored
//            isTerminal = false;
//            return 0.0;
//        }
//        isTerminal = (
//                yourResponse == Bandits ||
//                yourResponse == YouWonFight ||
//                yourResponse == YouLostFight ||
//                yourResponse == close ||
//                (myLastAction == WalkAway && yourResponse == WalkAway)
//        );
        utilityBeforeLastAct = utility();
        if (deselby::Random::nextBool(pBanditAttack)) {
            utilityBeforeLastAct += costOfBanditAttack;
            sugar() = 0;
            spice() = 0;
            return Bandits;
        }
        message_type outgoingMessage;
        switch (action) {
            case iGiveSugar:
                sugar() -= 1;
                outgoingMessage = GiveSugar;
                break;
            case iGiveSpice:
                spice() -= 1;
                outgoingMessage = GiveSpice;
                break;
            case iWalkAway:
                outgoingMessage = WalkAway;
                break;
            case iFight:
                // I started fight but lost
                utilityBeforeLastAct += costOfFighting;
                if(deselby::Random::nextBool()) {
                    outgoingMessage = YouWonFight;
                    sugar() = 0;
                    spice() = 0;
                } else {
                    outgoingMessage = YouLostFight;
                    sugar() = 1;
                    spice() = 1;
                }
                break;
            case iSay0:
                outgoingMessage = Say0;
                break;
            case iSay1:
                outgoingMessage = Say1;
            default:
                throw("Unrecognized act while handling act");
        }
        if (!isTerminal) recordOutgoingMessage(outgoingMessage);
        return outgoingMessage;
    }

    template<bool HASLANGUAGE>
    double SugarSpiceTradingBody<HASLANGUAGE>::messageToReward(SugarSpiceTradingBody::message_type incomingMessage) {
        isTerminal = (
                incomingMessage == Bandits ||
                incomingMessage == YouWonFight ||
                incomingMessage == YouLostFight ||
                incomingMessage == close ||
                (getLastOutgoingMessage() == WalkAway && incomingMessage == WalkAway));
        switch (incomingMessage) {
            case GiveSugar:
                sugar() += 1;
                break;
            case GiveSpice:
                spice() += 1;
                break;
            case YouWonFight:
                // You started fight and I won
                utilityBeforeLastAct += costOfFighting;
                sugar() = 1;
                spice() = 1;
                break;
            case YouLostFight:
                // You started fight and I lost
                utilityBeforeLastAct += costOfFighting;
                sugar() = 0;
                spice() = 0;
                break;
            case Bandits:
                utilityBeforeLastAct += costOfBanditAttack;
                sugar() = 0;
                spice() = 0;
                break;
            default:
                break;
        }
        recordIncomingMessage(incomingMessage);
        double reward = utility() - utilityBeforeLastAct;
        return reward;
    }


}






#endif //MULTIAGENTGOVERNMENT_SUGARSPICETRADINGBODY_H
