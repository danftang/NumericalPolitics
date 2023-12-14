//
// Created by daniel on 17/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_SUGARSPICETRADINGBODY_H
#define MULTIAGENTGOVERNMENT_SUGARSPICETRADINGBODY_H

#include <deque>
#include <bitset>
#include "../../DeselbyStd/random.h"
#include "mlpack.hpp"

#include "../Agent.h"
#include "../episodes/SimpleEpisode.h"

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
            HighestRecordedMessage = Bandits
        };

        typedef message_type in_message_type;
        typedef const SugarSpiceTradingBody<HASLANGUAGE> & init_type;

        static const int utilityOfPreferred = 10;      // utility of holding sugar/spice
        static const int utilityOfNonPreferred = 1;
        static constexpr double costOfFighting = 1.5;
        static constexpr double costOfBanditAttack = 15.0;
        inline static double pBanditAttack = 0.02; // probability of a bandit attack per message received
        static const bool encodeOutgoingMessage = false;
        static const int nOneHotBitsForMessageEncode = static_cast<int>(message_type::HighestRecordedMessage) + 1;
        static constexpr size_t dimension = nOneHotBitsForMessageEncode * (1 + encodeOutgoingMessage) + 3;
        static constexpr size_t nstates =
                8 * nOneHotBitsForMessageEncode * (encodeOutgoingMessage ? nOneHotBitsForMessageEncode : 1);

        // Agent state
//        arma::mat::fixed<dimension, 1> netInput; // state in the form of one-hot vectors of action history, plus sugar, spice and preference
//        arma::mat netInput = arma::mat(dimension,1); // state in the form of one-hot vectors of action history, plus sugar, spice and preference
//        bool isTerminal = false;
//        double reward = 0.0;
        bool bSugar;
        bool bSpice;
        bool bPrefersSugar;
        message_type lastOutgoingMessage;
        message_type lastIncomingMessage;
        action_mask legalMoves;

        SugarSpiceTradingBody(): SugarSpiceTradingBody(false, false, false) { }

        SugarSpiceTradingBody(bool hasSugar, bool hasSpice, bool prefersSugar) {
            resetLegalMoves();
            reset(hasSugar, hasSpice, prefersSugar);
        }


        // ----- Body interface -----

        events::OutgoingMessage<message_type> handleAct(int action);

        events::IncomingMessageResponse handleMessage(message_type incomingMessage);

        std::bitset<action_type::size> legalActs() const;

//        template<class A1, class A2> void on(const events::EndEpisode<A1,A2> &);

        action_type messageToAct(message_type message) const;

        // ---- End of Body interface

        void setSugar(int nSugar) {
            bSugar = nSugar;
            legalMoves[iGiveSugar] = nSugar>0;
        }

        void setSpice(int nSpice) {
            bSpice = nSpice;
            legalMoves[iGiveSpice] = nSpice>0;
        }

        void setPrefersSugar(bool prefersSugar) { bPrefersSugar = prefersSugar; }

//        [[nodiscard]] const double sugar() const { return bSugar; }
//
//        [[nodiscard]] const double spice() const { return bSpice; }

        [[nodiscard]] bool hasSugar() const { return bSugar; }

        [[nodiscard]] bool hasSpice() const { return bSpice; }

        [[nodiscard]] bool prefersSugar() const { return bPrefersSugar; }

        operator arma::mat () const {
            arma::mat netInput(dimension,1);
            netInput.zeros();
            netInput[0] = hasSugar();
            netInput[1] = hasSpice();
            netInput[2] = prefersSugar();
            int lastIncomingMessageInt = static_cast<int>(std::min(message_type::HighestRecordedMessage, lastIncomingMessage));
            int lastOutgoingMessageInt = static_cast<int>(std::min(message_type::HighestRecordedMessage, lastOutgoingMessage));
            netInput[3 + lastIncomingMessageInt] = 1.0;
            if (encodeOutgoingMessage) netInput[3 + nOneHotBitsForMessageEncode + lastOutgoingMessageInt] = 1.0;
            return netInput;
        }

        // convert to integer giving the ordinal of this state
        operator size_t() const {
            return hasSugar() + 2 * hasSpice() + 4 * prefersSugar() + 8 * (static_cast<int>(lastIncomingMessage) +
                                                                          (encodeOutgoingMessage ?
                                                                           nOneHotBitsForMessageEncode *
                                                                                   static_cast<int>(lastOutgoingMessage)
                                                                                                 : 0));
        }

        void recordIncomingMessage(message_type message) { lastIncomingMessage = message; }

        void recordOutgoingMessage(message_type message) { lastOutgoingMessage = message; }

        void reset(bool hasSugar, bool hasSpice, bool prefersSugar);

//        void onInit(const SugarSpiceTradingBody<HASLANGUAGE> &state) { reset(state.sugar(), state.spice(), state.prefersSugar()); }

        // [[nodiscard]] double utility() const;

//        friend std::ostream &operator <<(std::ostream &out, const typename abm::agents::SugarSpiceTradingBody<HASLANGUAGE>::message_type &message);

        friend std::ostream &operator<<(std::ostream &out, const SugarSpiceTradingBody &state) {
//                    out << state.sugar() << state.spice() << state.prefersSugar();
            out << state.hasSugar() << " " << state.hasSpice() << " " << state.prefersSugar() << " " << state.lastIncomingMessage << " " << state.lastOutgoingMessage;
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

        void resetLegalMoves();
    };

    /** sample from the posterior act probability given a message, assuming uniform prior over acts */
    template<bool HASLANGUAGE>
    SugarSpiceTradingBody<HASLANGUAGE>::action_type
    SugarSpiceTradingBody<HASLANGUAGE>::messageToAct(SugarSpiceTradingBody::message_type message) const {
            switch(message) {
                case message_type::GiveSugar:
                    return iGiveSugar;
                case message_type::GiveSpice:
                    return iGiveSpice;
                case message_type::WalkAway:
                    return iWalkAway;
                case message_type::Say0:
                    return iSay0;
//                case message_type::Say1:
//                    return iSay1;
                case message_type::Bandits:
                    return action_type(minds::ZeroIntelligence::sampleUniformly(legalActs()));
                case message_type::YouWonFight:
                case message_type::YouLostFight:
                    return iFight;
                default:
                    throw(std::runtime_error("Unrecognized message"));
            }
    }

//    template<bool HASLANGUAGE>
//    template<class A1, class A2>
//    void SugarSpiceTradingBody<HASLANGUAGE>::on(const events::EndEpisode<A1,A2> & /* event */) {
//        resetLegalMoves();
//    }

//    template<bool HASLANGUAGE>
//    bool SugarSpiceTradingBody<HASLANGUAGE>::isEndOfEpisode() {
//        return isTerminal;
//    }


    template<bool HASLANGUAGE>
    SugarSpiceTradingBody<HASLANGUAGE>::action_mask
    SugarSpiceTradingBody<HASLANGUAGE>::legalActs() const {
        return legalMoves;
    }

    template<bool HASLANGUAGE>
    void SugarSpiceTradingBody<HASLANGUAGE>::resetLegalMoves() {
        for (int i = 0; i < legalMoves.size(); ++i) legalMoves[i] = true;
        legalMoves[iSay0] = HASLANGUAGE;
//        legalActs[iSay1] = HASLANGUAGE;


    }
//    template<bool HASLANGUAGE>
//    double SugarSpiceTradingBody<HASLANGUAGE>::utility() const {
//        double utilityOfSugar = (prefersSugar() > 0.5 ? utilityOfPreferred : utilityOfNonPreferred);
//        double utilityOfSpice = (prefersSugar() > 0.5 ? utilityOfNonPreferred : utilityOfPreferred);
//        return sugar() * utilityOfSugar + spice() * utilityOfSpice;
//    }

    template<bool HASLANGUAGE>
    void SugarSpiceTradingBody<HASLANGUAGE>::reset(bool hasSugar, bool hasSpice, bool prefersSugar) {
        recordIncomingMessage(message_type::HighestRecordedMessage); // use first terminal state to mean Empty as we don't use this
        recordOutgoingMessage(message_type::HighestRecordedMessage);
        setSugar(hasSugar);
        setSpice(hasSpice);
        setPrefersSugar(prefersSugar);
    }


    template<bool HASLANGUAGE>
    abm::events::OutgoingMessage<typename SugarSpiceTradingBody<HASLANGUAGE>::message_type>
    SugarSpiceTradingBody<HASLANGUAGE>::handleAct(int action) {
        events::OutgoingMessage<message_type> response(message_type(), 0.0);
        if (deselby::random::Bernoulli(pBanditAttack)) {
            response.reward -= costOfBanditAttack;
            if(hasSugar()) response.reward -= (prefersSugar()?utilityOfPreferred:utilityOfNonPreferred);
            if(hasSpice()) response.reward -= (prefersSugar()?utilityOfNonPreferred:utilityOfPreferred);
            setSugar(0);
            setSpice(0);
            response.message = message_type::Bandits;
            return response;
        }
        switch (action) {
            case iGiveSugar:
                assert(hasSugar());
                setSugar(0);
                response.message = message_type::GiveSugar;
                response.reward -= (prefersSugar()?utilityOfPreferred:utilityOfNonPreferred);
                break;
            case iGiveSpice:
                assert(hasSpice());
                setSpice(0);
                response.message = message_type::GiveSpice;
                response.reward -= (prefersSugar()?utilityOfNonPreferred:utilityOfPreferred);
                break;
            case iWalkAway:
                response.message = message_type::WalkAway;
                break;
            case iFight:
                // I started fight
                response.reward -= costOfFighting;
                if(deselby::random::uniform<bool>()) {
                    response.message = message_type::YouWonFight; // I lost fight
                    if(hasSugar()) response.reward -= (prefersSugar()?utilityOfPreferred:utilityOfNonPreferred);
                    if(hasSpice()) response.reward -= (prefersSugar()?utilityOfNonPreferred:utilityOfPreferred);
                    setSugar(0);
                    setSpice(0);
                } else {
                    response.message = message_type::YouLostFight; // I won fight
                    if(!hasSugar()) response.reward += (prefersSugar()?utilityOfPreferred:utilityOfNonPreferred);
                    if(!hasSpice()) response.reward += (prefersSugar()?utilityOfNonPreferred:utilityOfPreferred);
                    setSugar(1);
                    setSpice(1);
                }
                break;
            case iSay0:
                response.message = message_type::Say0;
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
        recordOutgoingMessage(response.message);
//        std::cout << "Sending message " << response << std::endl;
        return response;
    }


    template<bool HASLANGUAGE>
    events::IncomingMessageResponse SugarSpiceTradingBody<HASLANGUAGE>::handleMessage(message_type incomingMessage) {
        events::IncomingMessageResponse response{
            0.0,
            incomingMessage == message_type::Bandits ||
            incomingMessage == message_type::YouWonFight ||
            incomingMessage == message_type::YouLostFight ||
            (incomingMessage == message_type::WalkAway && lastOutgoingMessage == message_type::WalkAway)
        };
        switch (incomingMessage) {
            case message_type::GiveSugar:
                setSugar(1);
                response.reward += (prefersSugar()?utilityOfPreferred:utilityOfNonPreferred);
                break;
            case message_type::GiveSpice:
                setSpice(1);
                response.reward += (prefersSugar()?utilityOfNonPreferred:utilityOfPreferred);
                break;
            case message_type::YouWonFight:
                // You started fight and I won
                response.reward -= costOfFighting;
                if(!hasSugar()) response.reward += (prefersSugar()?utilityOfPreferred:utilityOfNonPreferred);
                if(!hasSpice())  response.reward += (prefersSugar()?utilityOfNonPreferred:utilityOfPreferred);
                setSugar(1);
                setSpice(1);
                break;
            case message_type::YouLostFight:
                // You started fight and I lost
                response.reward -= costOfFighting;
                if(hasSugar()) response.reward -= (prefersSugar()?utilityOfPreferred:utilityOfNonPreferred);
                if(hasSpice()) response.reward -= (prefersSugar()?utilityOfNonPreferred:utilityOfPreferred);
                setSugar(0);
                setSpice(0);
                break;
            case message_type::Bandits:
                response.reward -= costOfBanditAttack;
                if(hasSugar()) response.reward -= (prefersSugar()?utilityOfPreferred:utilityOfNonPreferred);
                if(hasSpice()) response.reward -= (prefersSugar()?utilityOfNonPreferred:utilityOfPreferred);
                setSugar(0);
                setSpice(0);
                break;
            default:
                break;
        }
        recordIncomingMessage(incomingMessage);
//        std::cout << "Incoming message response " << response << std::endl;
        return response;
    }
}

#endif //MULTIAGENTGOVERNMENT_SUGARSPICETRADINGBODY_H
