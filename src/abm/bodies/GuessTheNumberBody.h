// The Guess-the-number is an asynchronous game that proceeds as follows:
//   A first mover is chosen at random and initialised with an number in 1...N, unknown to the other agent.
//   The first mover must then pass a symbol in a language, L. The second mover must then guess which number the
//   first mover was given. If the second mover guesses correctly both agents get reward 1, otherwise both get no reward.
//
// This is a game to test an agent's ability to create a language.
//
// Created by daniel on 19/12/23.
//

#ifndef MULTIAGENTGOVERNMENT_GUESSTHENUMBERBODY_H
#define MULTIAGENTGOVERNMENT_GUESSTHENUMBERBODY_H

#include "../Agent.h"
#include "../../DeselbyStd/random.h"

namespace abm::bodies {
    class GuessTheNumberBody {
    public:
        enum action_type {
            // !iHavePlayed, !iAmGuesser -> iAmGuesser
            sayA,
            sayB,
            sayC,
            // !iHavePlayed, iAmGuesser -> !iAmGuesser
            guess1 = sayA,
            guess2 = sayB,
            guess3 = sayC,
            // iHavePlayed
            wrong = 0,
            right = 1,
            size
        };

        typedef action_type message_type;

        bool iAmGuesser;
        bool iHavePlayed;
        action_type state; // stores hidden number, (right/wrong) or received hint


        void init(bool isGuesser) {
            iHavePlayed = false;
            iAmGuesser = isGuesser;
            state = action_type(deselby::random::uniform<int>(action_type::size));
        }

        // ----- Body interface -----

        events::OutgoingMessage<message_type> handleAct(int action) {
            iHavePlayed = true;
            return {static_cast<message_type>(action), 0.0};
        }

        events::IncomingMessageResponse handleMessage(message_type incomingMessage) {
            double reward = 0.0;
            bool isEndEpisode = false;
            if(iAmGuesser) {
                if(iHavePlayed) {
                    switch(incomingMessage) {
                        case right: reward = 1.0; isEndEpisode = true; break;
                        case wrong: isEndEpisode = true; break;
                        default: throw(std::runtime_error("Illegal move"));
                    }
                } else {
                    state = incomingMessage;
                }
            } else {
                if(state == incomingMessage) {
                    reward = 1.0;
                    state = right;
                } else {
                    state = wrong;
                }
            }
            return events::IncomingMessageResponse(reward, isEndEpisode);
        }

        std::bitset<action_type::size> legalActs() const {
            if(iHavePlayed && state == right) return std::bitset<action_type::size>(2);
            if(iHavePlayed && state == wrong) return std::bitset<action_type::size>(1);
            return std::bitset<action_type::size>("111");
        }

        action_type messageToAct(message_type message) const { return message; }

        // ---- End of Body interface


    };
}

#endif //MULTIAGENTGOVERNMENT_GUESSTHENUMBERBODY_H
