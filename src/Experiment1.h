//
// Created by daniel on 27/02/23.
//
// Create a multi-agent society of Q-learning agents (with heterogeneous discount, possibly heterogeneous reward)
// who learn to play
// repeated prisoner's dilemma, in a round-robin fashion, where each interaction consists of
// two rounds of the game, and perception is the move history (i.e.
// Q table of size 10).
//
// Add a government that optimises a "social-welfare" function over some family of
// perturbations of the natural reward structure as a function of
// perceptual state.
// The family of perturbations represent the total effect on reward
// of a set of laws (accounting for people's intrinsic desire for
// freedom and any costs of enforcement).

#ifndef MULTIAGENTGOVERNMENT_EXPERIMENT1_H
#define MULTIAGENTGOVERNMENT_EXPERIMENT1_H

#include <random>
#include <array>
#include "abm/abm.h"
#include "abm/DiscreteEventSimulation.h"

namespace experiment1 {
    void QPrisonersDilemma();

    class PrisonersDilemmaAgent {
    public:
        static constexpr size_t NSTATES  = 4; // number of possible outputs from the observation function
        static constexpr size_t NACTIONS = 2;
        static constexpr float  REWARD[2][2] = {{3, 0},
                                                {4, 1}}; // Agent reward for [ownMove][opponentMove]
                                                         // where 0 = co-operate, 1 = defect
        typedef ulong time_type;
        typedef abm::QTablePolicy<NSTATES,NACTIONS> policy_type;
        typedef abm::Schedule<time_type> schedule_type;

        policy_type policy;
        int myLastMove;
        abm::CommunicationChannel<abm::Schedule<time_type>, int> opponent;

        void connectTo(PrisonersDilemmaAgent &opponentAgent) {
            opponent.connectTo(opponentAgent, &PrisonersDilemmaAgent::handleOpponentMove, 1);
        }

        schedule_type start() {
            myLastMove = 0;
            policy.trainingStepsSinceLastPolicyChange = 0;
            policy.pExplore = policy.DEFAULT_INITIAL_EXPLORATION;
            return opponent.send(myLastMove, 0);
        }

        schedule_type handleOpponentMove(int opponentsMove, time_type time) {
//            std::cout << "Handling " << opponentsMove << ", " << myLastMove << std::endl;
            int newState = 2*myLastMove + opponentsMove;
            myLastMove = policy.getActionAndTrain(newState, REWARD[myLastMove][opponentsMove]);
            return opponent.send(myLastMove, time);
        }
    };
}






#endif //MULTIAGENTGOVERNMENT_EXPERIMENT1_H
