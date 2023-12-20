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
            size,
            // !iHavePlayed, iAmGuesser -> !iAmGuesser
            guess1 = sayA,
            guess2 = sayB,
            guess3 = sayC,
            // iHavePlayed
            wrong = 0,
            right = 1
        };

        typedef action_type message_type;

        static constexpr size_t dimension = 5;

        bool iAmGuesser;
        bool iHavePlayed;
        action_type state; // stores hidden number, (right/wrong) or received hint


        void reset(bool isGuesser) {
            iHavePlayed = false;
            iAmGuesser = isGuesser;
            state = action_type(isGuesser?0:deselby::random::uniform<int>(action_type::size));
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
//            if(!iAmGuesser) return 1<<state; // TODO: TEST!!!
            return std::bitset<action_type::size>("111");
        }

        action_type messageToAct(message_type message) const { return message; }

        operator arma::mat () {
            arma::mat vecState(5,1);
            vecState(0,0) = iAmGuesser;
            vecState(1,0) = iHavePlayed;
            vecState(2,0) = state == sayA;
            vecState(3,0) = state == sayB;
            vecState(4,0) = state == sayC;
            return vecState;
        }

        // ---- End of Body interface

        friend std::ostream &operator<<(std::ostream &out, const GuessTheNumberBody &body) {
            out << body.iAmGuesser << " " << body.iHavePlayed << " " << body.state;
            return out;
        }

    };
}

#endif //MULTIAGENTGOVERNMENT_GUESSTHENUMBERBODY_H
