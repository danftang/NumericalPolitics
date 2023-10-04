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

        message_type    myLastMove;
    private:
        double          pEndEpisode;
        static constexpr int moveId(message_type myMove, message_type yourMove) { return myMove*2 + yourMove; }

    public:

        PrisonersDilemmaBody(double pEndEpisode): pEndEpisode(pEndEpisode) { }

        events::OutgoingMessage<message_type, PrisonersDilemmaBody> handleAct(uint actFromMind) {
            myLastMove = static_cast<message_type>(actFromMind);
            return {*this, deselby::Random::nextBool(pEndEpisode) ? std::nullopt : std::optional(myLastMove)};
        };

        double handleMessage(message_type incomingMessage) {
            switch(moveId(myLastMove, incomingMessage)) {
                case moveId(Cooperate, Cooperate):  return 3.0;
                case moveId(Cooperate, Defect):     return 0.0;
                case moveId(Defect,    Cooperate):  return 4.0;
                case moveId(Defect,    Defect):     return 1.0;
            }
            throw(std::out_of_range("unknown incomming message"));
        }

        static inline auto legalActs() { return std::bitset<2>(3); }
    };
}


#endif //MULTIAGENTGOVERNMENT_PRISONERSDILEMMABODY_H
