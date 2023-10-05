//
// Created by daniel on 16/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_PRISONERSDILEMMABODY_H
#define MULTIAGENTGOVERNMENT_PRISONERSDILEMMABODY_H

#include <bitset>
#include "../../DeselbyStd/random.h"
#include "../Agent.h"

namespace abm::bodies {
    class PrisonersDilemmaBody {
    public:
        enum message_type {
            Cooperate,
            Defect,
            size
        };

        uint    myLastMove      = Cooperate;
        uint    yourLastMove    = Cooperate;
    private:
        double          pEndEpisode;
        static constexpr int moveId(uint myMove, uint yourMove) { return myMove*2 + yourMove; }

    public:

        PrisonersDilemmaBody(double pEndEpisode): pEndEpisode(pEndEpisode) { }

        events::MessageReward<std::optional<message_type>> handleAct(size_t actFromMind) {
            myLastMove = actFromMind;
            return {(deselby::Random::nextBool(pEndEpisode) ? std::nullopt : std::optional(static_cast<message_type>(myLastMove))), 0.0};
        };

        double handleMessage(uint incomingMessage) {
            yourLastMove = incomingMessage;
            switch(moveId(myLastMove, yourLastMove)) {
                case moveId(Cooperate, Cooperate):  return 3.0;
                case moveId(Cooperate, Defect):     return 0.0;
                case moveId(Defect,    Cooperate):  return 4.0;
                case moveId(Defect,    Defect):     return 1.0;
            }
            throw(std::out_of_range("unknown incomming message"));
        }

        operator size_t() const { return 2*myLastMove + yourLastMove; }

        static inline auto legalActs() { return std::bitset<2>(3); }
    };
}


#endif //MULTIAGENTGOVERNMENT_PRISONERSDILEMMABODY_H
