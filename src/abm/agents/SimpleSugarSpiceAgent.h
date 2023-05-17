// An agent that has just one bit of memory, holding the last encountered move
// (all agents are considered strangers)
//
// Created by daniel on 15/05/23.
//

#ifndef MULTIAGENTGOVERNMENT_SIMPLESUGARSPICEAGENT_H
#define MULTIAGENTGOVERNMENT_SIMPLESUGARSPICEAGENT_H

#include <cstdlib>
#include "../Schedule.h"
#include "../QTablePolicy.h"
#include "../CommunicationChannel.h"

namespace abm {
    namespace agents {

        class SimpleSugarSpiceAgent {
        public:
            static constexpr size_t NSTATES = 4; // cooperate/cooperate, cooperate/defect, defect/cooperate, defect/defect
            static constexpr size_t NACTIONS = 2;

            typedef ulong time_type;
            typedef QTablePolicy<NSTATES, NACTIONS> policy_type;
            typedef Schedule<time_type> schedule_type;

            static constexpr double REWARD[2][2] = {{3, 0},
                                                   {4, 1}};

            static constexpr double QMIN = 0;
            static constexpr double QMAX = REWARD[1][0] /
                                          (1.0 -
                                           policy_type::DEFAULT_DISCOUNT); // Value of Q if all future rewards are max reward

            policy_type policy;
            CommunicationChannel<abm::Schedule<time_type>, int> opponent;
            int lastGameOutcome;
            int myLastMove;

            void setPolicy(long policyID) { policy.setPolicy(policyID, QMIN, QMAX); }

            // Handler for receiving a message to handle a new opponent
            // Connect to the new opponent and sends it a trading action at the given time
            schedule_type handleNewOpponent(SimpleSugarSpiceAgent &newOpponent, time_type time) {
                opponent.connectTo(newOpponent, &SimpleSugarSpiceAgent::handleOpponentMove, 1);
                myLastMove = policy.getAction(lastGameOutcome);
                return opponent.send(myLastMove, time);
            }


            // handler for receiving opponents move.
            // No action needs to be taken, just need to train on the outcome/reward
            // and update lastGameOutcome
            schedule_type handleOpponentMove(int opponentsMove, time_type time) {
//        std::cout << this << " Handling " << myLastMove << ", " << opponentsMove << std::endl;
                int newGameOutcome = 2 * myLastMove + opponentsMove;
                double reward = REWARD[myLastMove][opponentsMove];
                policy.train(lastGameOutcome, myLastMove, reward, newGameOutcome);
                lastGameOutcome = newGameOutcome;
                return Schedule<time_type>();
            }

        };
    }
}

#endif //MULTIAGENTGOVERNMENT_SIMPLESUGARSPICEAGENT_H
