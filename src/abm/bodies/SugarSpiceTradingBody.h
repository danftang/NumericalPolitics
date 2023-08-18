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

namespace abm::bodies {

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
//            iSay1,
            size  // marker for number of actions
        };

        typedef std::bitset<action_type::size> action_mask;

        // the tyoe of message passed between agents in response to an agent's intent
        enum class message_type {
            // Agent actions
            GiveSugar,
            GiveSpice,
            WalkAway,
            Say0,
            Say1,
            Bandits, // Terminal messages should all be at the end
            YouWonFight,
            YouLostFight,
            size,
            IndeterminateTerminalMessage = Bandits
        };

        typedef message_type in_message_type;

        static const int utilityOfPreferred = 10;      // utility of holding sugar/spice
        static const int utilityOfNonPreferred = 1;
        static constexpr double costOfFighting = 1.5;
        static constexpr double costOfBanditAttack = 15.0;
        inline static double pBanditAttack = 0.02; // probability of a bandit attack per message received
        static const bool encodeOutgoingMessage = false;
        static const int nOneHotBitsForMessageEncode = static_cast<int>(message_type::IndeterminateTerminalMessage) + 1;
        static constexpr size_t dimension = nOneHotBitsForMessageEncode * (1 + encodeOutgoingMessage) + 3;
        static constexpr size_t nstates =
                8 * nOneHotBitsForMessageEncode * (encodeOutgoingMessage ? nOneHotBitsForMessageEncode : 1);

        // Agent state
        arma::mat::fixed<dimension, 1> netInput; // state in the form of one-hot vectors of action history, plus sugar, spice and preference
        bool isTerminal = false;
        double reward = 0.0;
        message_type lastOutgoingMessage;

        SugarSpiceTradingBody(): SugarSpiceTradingBody(false, false, false) { }

        SugarSpiceTradingBody(bool hasSugar, bool hasSpice, bool prefersSugar) {
            reset(hasSugar, hasSpice, prefersSugar);
        }


        // ----- Body interface -----

        message_type actToMessage(int action);
        double messageToReward(message_type incomingMessage);
        std::bitset<action_type::size> legalActs();
//        bool isEndOfEpisode();
        double endEpisode();

        // ---- End of Body interface

        double &sugar() { return netInput[0]; }

        double &spice() { return netInput[1]; }

        double &prefersSugar() { return netInput[2]; }

        [[nodiscard]] double sugar() const { return netInput[0]; }

        [[nodiscard]] double spice() const { return netInput[1]; }

        [[nodiscard]] double prefersSugar() const { return netInput[2]; }

        operator const arma::mat::fixed<dimension,1> &() const {
            return netInput;
        }

        // convert to integer giving the ordinal of this state
        operator int() const {
            return netInput[0] + 2 * netInput[1] + 4 * netInput[2] + 8 * (static_cast<int>(getLastIncomingMessage()) +
                                                                          (encodeOutgoingMessage ?
                                                                           nOneHotBitsForMessageEncode *
                                                                                   static_cast<int>(getLastOutgoingMessage())
                                                                                                 : 0));
        }

        void recordIncomingMessage(message_type message) {
            assert(static_cast<int>(message) >= 0);
            if(message > message_type::IndeterminateTerminalMessage) message = message_type::IndeterminateTerminalMessage; // don't record type of terminal messages
            for (int i = 3; i < 3 + nOneHotBitsForMessageEncode; ++i) netInput[i] = 0.0;
            assert(3 + static_cast<int>(message) < dimension);
            netInput[3 + static_cast<int>(message)] = 1.0;
        }

        void recordOutgoingMessage(message_type message) {
            if (encodeOutgoingMessage) {
                assert(static_cast<int>(message) >= 0);
                if(message > message_type::IndeterminateTerminalMessage) message = message_type::IndeterminateTerminalMessage;
                for (int i = 3 + nOneHotBitsForMessageEncode; i < 3 + 2 * nOneHotBitsForMessageEncode; ++i)
                    netInput[i] = 0.0;
                netInput[3 + nOneHotBitsForMessageEncode + static_cast<int>(message)] = 1.0;
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

        // [[nodiscard]] double utility() const;

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
            out << messageNames[static_cast<int>(message)];
            return out;
        }

    };

    template<bool HASLANGUAGE>
    double SugarSpiceTradingBody<HASLANGUAGE>::endEpisode() {
        double residualReward = reward;
        reward = 0.0;
        isTerminal = false;
        return residualReward;
    }

//    template<bool HASLANGUAGE>
//    bool SugarSpiceTradingBody<HASLANGUAGE>::isEndOfEpisode() {
//        return isTerminal;
//    }


    template<bool HASLANGUAGE>
    SugarSpiceTradingBody<HASLANGUAGE>::action_mask
    SugarSpiceTradingBody<HASLANGUAGE>::legalActs() {
        if(isTerminal) return 0;
        action_mask legalActs;
        for (int i = 0; i < legalActs.size(); ++i) legalActs[i] = true;
        if (sugar() == 0) legalActs[iGiveSugar] = false;
        if (spice() == 0) legalActs[iGiveSpice] = false;
        legalActs[iSay0] = HASLANGUAGE;
//        legalActs[iSay1] = HASLANGUAGE;
        return legalActs;
    }

//    template<bool HASLANGUAGE>
//    double SugarSpiceTradingBody<HASLANGUAGE>::utility() const {
//        double utilityOfSugar = (prefersSugar() > 0.5 ? utilityOfPreferred : utilityOfNonPreferred);
//        double utilityOfSpice = (prefersSugar() > 0.5 ? utilityOfNonPreferred : utilityOfPreferred);
//        return sugar() * utilityOfSugar + spice() * utilityOfSpice;
//    }

    template<bool HASLANGUAGE>
    void SugarSpiceTradingBody<HASLANGUAGE>::reset(bool hasSugar, bool hasSpice, bool prefersSugar) {
        recordIncomingMessage(message_type::IndeterminateTerminalMessage); // use first terminal state to mean Empty as we don't use this
        recordOutgoingMessage(message_type::IndeterminateTerminalMessage);
        sugar() = hasSugar;
        spice() = hasSpice;
        this->prefersSugar() = prefersSugar;
        reward = 0.0;
    }


    template<bool HASLANGUAGE>
    SugarSpiceTradingBody<HASLANGUAGE>::message_type SugarSpiceTradingBody<HASLANGUAGE>::actToMessage(int action) {
        if (deselby::Random::nextBool(pBanditAttack)) {
            reward -= costOfBanditAttack;
            if(sugar() == 1.0) reward -= (prefersSugar()?utilityOfPreferred:utilityOfNonPreferred);
            if(spice() == 1.0)  reward -= (prefersSugar()?utilityOfNonPreferred:utilityOfPreferred);
            sugar() = 0;
            spice() = 0;
            isTerminal = true;
            return message_type::Bandits;
        }
        message_type outgoingMessage;
        switch (action) {
            case iGiveSugar:
                assert(sugar() >= 1);
                sugar() -= 1;
                outgoingMessage = message_type::GiveSugar;
                reward -= (prefersSugar()?utilityOfPreferred:utilityOfNonPreferred);
                break;
            case iGiveSpice:
                assert(spice() >= 1);
                spice() -= 1;
                outgoingMessage = message_type::GiveSpice;
                reward -= (prefersSugar()?utilityOfNonPreferred:utilityOfPreferred);
                break;
            case iWalkAway:
                outgoingMessage = message_type::WalkAway;
                break;
            case iFight:
                // I started fight
                reward -= costOfFighting;
                if(deselby::Random::nextBool()) {
                    outgoingMessage = message_type::YouWonFight;
                    if(sugar() == 1.0) reward -= (prefersSugar()?utilityOfPreferred:utilityOfNonPreferred);
                    if(spice() == 1.0)  reward -= (prefersSugar()?utilityOfNonPreferred:utilityOfPreferred);
                    sugar() = 0;
                    spice() = 0;
                } else {
                    outgoingMessage = message_type::YouLostFight;
                    if(sugar() == 0.0) reward += (prefersSugar()?utilityOfPreferred:utilityOfNonPreferred);
                    if(spice() == 0.0)  reward += (prefersSugar()?utilityOfNonPreferred:utilityOfPreferred);
                    sugar() = 1;
                    spice() = 1;
                }
                break;
            case iSay0:
                outgoingMessage = message_type::Say0;
                break;
//            case iSay1:
//                outgoingMessage = message_type::Say1;
//                break;
            default:
                throw(std::out_of_range("Unrecognized act while handling act"));
        }
//        isTerminal =
//                (outgoingMessage == message_type::YouWonFight ||
//                outgoingMessage == message_type::YouLostFight ||
//                (outgoingMessage == message_type::WalkAway && getLastIncomingMessage() == message_type::WalkAway));
        recordOutgoingMessage(outgoingMessage);
        return outgoingMessage;
    }


    template<bool HASLANGUAGE>
    double SugarSpiceTradingBody<HASLANGUAGE>::messageToReward(SugarSpiceTradingBody::message_type incomingMessage) {
        switch (incomingMessage) {
            case message_type::GiveSugar:
                sugar() += 1;
                reward += (prefersSugar()?utilityOfPreferred:utilityOfNonPreferred);
                break;
            case message_type::GiveSpice:
                spice() += 1;
                reward += (prefersSugar()?utilityOfNonPreferred:utilityOfPreferred);
                break;
            case message_type::YouWonFight:
                // You started fight and I won
                reward -= costOfFighting;
                if(sugar() == 0.0) reward += (prefersSugar()?utilityOfPreferred:utilityOfNonPreferred);
                if(spice() == 0.0)  reward += (prefersSugar()?utilityOfNonPreferred:utilityOfPreferred);
                sugar() = 1;
                spice() = 1;
                break;
            case message_type::YouLostFight:
                // You started fight and I lost
                reward -= costOfFighting;
                if(sugar() == 1.0) reward -= (prefersSugar()?utilityOfPreferred:utilityOfNonPreferred);
                if(spice() == 1.0)  reward -= (prefersSugar()?utilityOfNonPreferred:utilityOfPreferred);
                sugar() = 0;
                spice() = 0;
                break;
            case message_type::Bandits:
                reward -= costOfBanditAttack;
                if(sugar() == 1.0) reward -= (prefersSugar()?utilityOfPreferred:utilityOfNonPreferred);
                if(spice() == 1.0)  reward -= (prefersSugar()?utilityOfNonPreferred:utilityOfPreferred);
                sugar() = 0;
                spice() = 0;
                break;
            default:
                break;
        }
        recordIncomingMessage(incomingMessage);
        double returnedReward = reward;
        reward = 0.0;
//        std::cout << "sending reward " << reward << std::endl;
        isTerminal =
                (incomingMessage == message_type::Bandits ||
                incomingMessage == message_type::YouWonFight ||
                incomingMessage == message_type::YouLostFight ||
                (incomingMessage == message_type::WalkAway && getLastOutgoingMessage() == message_type::WalkAway));
        return returnedReward;
    }
}

#endif //MULTIAGENTGOVERNMENT_SUGARSPICETRADINGBODY_H
