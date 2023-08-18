//
// Created by daniel on 16/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_PRISONERSDILEMMABODY_H
#define MULTIAGENTGOVERNMENT_PRISONERSDILEMMABODY_H

#include <bitset>
#include "../../DeselbyStd/random.h"

class PrisonersDilemmaBody {
public:
    enum message_type {
        Cooperate,
        Defect,
        size
    };

    typedef message_type action_type;   // incoming message type
    typedef message_type in_message_type;   // incoming message type

    action_type     myLastMove;
private:
    double          pEndEpisode;
    static constexpr int moveId(message_type myMove, message_type yourMove) { return myMove*2 + yourMove; }

public:

    PrisonersDilemmaBody(double pEndEpisode): pEndEpisode(pEndEpisode) { }

    message_type actToMessage(action_type actFromMind) {
        myLastMove = actFromMind;
        return actFromMind;
    };

    double messageToReward(in_message_type incomingMessage) {
        switch(moveId(myLastMove, incomingMessage)) {
            case moveId(Cooperate, Cooperate):  return 3.0;
            case moveId(Cooperate, Defect):     return 0.0;
            case moveId(Defect,    Cooperate):  return 4.0;
            case moveId(Defect,    Defect):     return 1.0;
        }
        throw(std::out_of_range("unknown incomming message"));
    }

    std::bitset<2> legalActs() {
        return deselby::Random::nextBool(pEndEpisode)?0:3;
    }

    double endEpisode() { return 0.0; }
};


#endif //MULTIAGENTGOVERNMENT_PRISONERSDILEMMABODY_H
