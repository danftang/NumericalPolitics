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
        double  pEndEpisode;
    private:
        static constexpr int moveId(uint myMove, uint yourMove) { return myMove*2 + yourMove; }

    public:

        PrisonersDilemmaBody(double pEndEpisode): pEndEpisode(pEndEpisode) { }

        events::OutgoingMessage<message_type> handleAct(size_t actFromMind) {
            myLastMove = actFromMind;
            return { message_type(myLastMove), 0.0 };
        };

        events::IncomingMessageResponse handleMessage(uint incomingMessage) {
            events::IncomingMessageResponse response {0.0, deselby::random::Bernoulli(pEndEpisode)};
            yourLastMove = incomingMessage;
            switch(moveId(myLastMove, yourLastMove)) {
                case moveId(Cooperate, Cooperate):  response.reward = 3.0; break;
                case moveId(Cooperate, Defect):     response.reward = 0.0; break;
                case moveId(Defect,    Cooperate):  response.reward = 4.0; break;
                case moveId(Defect,    Defect):     response.reward = 1.0; break;
                default: throw(std::out_of_range("unknown incomming message"));
            }
            return response;
        }

        operator size_t() const { return 2*myLastMove + yourLastMove; }

        static inline auto legalActs() { return std::bitset<2>(3); }
    };
}


#endif //MULTIAGENTGOVERNMENT_PRISONERSDILEMMABODY_H
