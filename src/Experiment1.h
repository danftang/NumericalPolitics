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
//    std::mt19937 rand;

    void QPrisonersDilemma();

    // An agent has:
    //  * Any number of public interfaces (consisting of channels that the agent writes to and per-agent public constants)
    //  * Any number of read channels from which the agent gathers info about its surroundings (implemented as const pointers to channels)
    //  * Internal, private state
    //  * A step function that gathers info from read channels, updates write channels and internal state and returns
    //    a schedule of tasks for computation.
    //
    //  The step function may (optionally) be split into three parts:
    //      - an observation function which takes the agent's internal state, read channels and a time
    //        and generates an observation (by convention, an agent observes its own internal state and write channels at each step)
    //      - a policy which takes an observation and generates an action
    //      - an action function which takes an <internal state, action> pair and returns an update of the
    //        write-channels and internal state.
    //
    //  For online learning, there is additionally a reward function which takes the internal state, read channels and a time,
    //  and returns a reward. Both the observation and reward are passed to an online policy which updates its
    //  internal state and returns an action.
    //
    // For batch training, a "replay" object records a set of <observation, action> tuples which represent
    // trajectories of experience and the policy takes this batch of trajectories and updates its internal state.
    //
    // An "agent view" represents three things:
    //   * the public interfaces (consisting of communication channels which other agents can read)
    //   * its observation function
    //     (taking info from its read channels and internal state to generate an observation and reward for the policy)
    //   * its action function (taking an action from the policy and updating the write channels and internal state)
    //

    class PrisonersDilemmaAgent {
    public:
        static constexpr size_t NSTATES  = 4; // number of possible outputs from the observation function
        static constexpr size_t NACTIONS = 2;
        static constexpr float  REWARD[2][2] = {{3, 0},
                                                {4, 1}};
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
