//
// Created by daniel on 15/05/23.
//

#ifndef MULTIAGENTGOVERNMENT_PRISONERSDILEMMAAGENT_H
#define MULTIAGENTGOVERNMENT_PRISONERSDILEMMAAGENT_H

#include <cstdlib>
#include "../Schedule.h"
#include "../QTablePolicy.h"
#include "../CommunicationChannel.h"

namespace abm {
    namespace agents {
        class PrisonersDilemmaAgent {
        public:
            static constexpr size_t NSTATES = 4; // number of possible outputs from the observation function
            static constexpr size_t NACTIONS = 2;
            static constexpr double REWARD[2][2] = {{3, 0},
                                                   {4, 1}}; // Agent reward for [ownMove][opponentMove]
            // where 0 = co-operate, 1 = defect

            typedef ulong time_type;
            typedef QTablePolicy<NSTATES, NACTIONS> policy_type;
            typedef Schedule<time_type> schedule_type;

            static constexpr double QMIN = 0;
            static constexpr double QMAX = REWARD[1][0] /
                                          (1.0 -
                                           policy_type::DEFAULT_DISCOUNT); // Value of Q if all future rewards are max reward

            policy_type policy;
            int myLastMove;
            CommunicationChannel<abm::Schedule<time_type>, int> opponent;

            void connectTo(PrisonersDilemmaAgent &opponentAgent) {
                opponent.connectTo(opponentAgent, &PrisonersDilemmaAgent::handleOpponentMove, 1);
            }

            schedule_type start() {
                myLastMove = 0;
                policy.trainingStepsSinceLastPolicyChange = 0;
                policy.pExplore = policy.DEFAULT_INITIAL_EXPLORATION;
                return opponent.send(myLastMove, 0);
            }

            void setPolicy(long policyID) {
                policy.setPolicy(policyID, QMIN, QMAX);
            }

            schedule_type handleOpponentMove(int opponentsMove, time_type time) {
//            std::cout << "Handling " << opponentsMove << ", " << myLastMove << std::endl;
                int newState = 2 * myLastMove + opponentsMove;
                myLastMove = policy.getActionAndTrain(newState, REWARD[myLastMove][opponentsMove]);
                return opponent.send(myLastMove, time);
            }
        };
    }
}

#endif //MULTIAGENTGOVERNMENT_PRISONERSDILEMMAAGENT_H
