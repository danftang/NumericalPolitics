//
// Created by daniel on 27/02/23.
//
// Background:
// A repeated prisoner's dilemma, where each agent remembers its own and other's last move
// with online Q-learning.
//
// At each round, an agent has a choice of two actions: co-operate or defect,
// so there are 4 agent states corresponding to the last moves of self and opponent.
// With 4 states, each with 2 possible actions, the Q-table of an agent has 8 entries.
// The "social state" is the 2 x 8 = 16 Q values in the two
// Q-tables of the agents, so the pair of agents has a 16 dimensional phase space in which
// we can consider the structure of the attractors (fixed point probability distribution).
//
// Questions:
//   * How many attractors are there?
//   * Are they all point attractors?
//
// Approach:
// There are only 16 possible agent policies (corresponding to the choice of cooperate or defect in
// each of the 4 states, which we number ) so only 256 possible policy pairs, so we can split
// the social space into 256 partitions (which we number ). For each partition, we start with a strong bias
// towards this policy pair and execute Q-learning until convergence. If an attractor exists
// in this partition, the agents should converge to it.
//
// We test for convergence by running until no policy changes occur for a "large" number
// of steps.
//
// Numbering convention:
//  Agent moves are numbered: 0 = co-operate, 1 = defect
//  Agent states are numbered 2*ownMove + opponentMove.
//  Agent policies are numbered sum_{s=0}^{N-1} 2^s m_s
//    where N is the number of states and m_s is the policy's move in state s
//  Social policy (partitions of the social phase space) is numbered 16*p_1 + p_2
//    where p_1 is the policy of agent 1 and p_2 is the policy of agent 2.
//
// Results:
// All start points eventually converge to a point attractor. However, convergence is quite slow.
// There are 5 attracting social policies:
//  1) 0x62/0x26
//  2) 0x66
//  3) 0xfe/0xef
//  4) 0xee
//  5) 0xff
//
// However, the mixed societies (0x62 and 0xfe) are artifacts of the slow convergence.
//
// Society 0xff immediately leads to mutual defection.
// Society 0xee has a stable equilibrium of mutual defection and an unstable equilibrium of mutual cooperation.
// Interestingly 0x66 quickly leads to mutual cooperation.
//
// Discussion:
// For a given pair of policies (Pa,Pb) there is the expected reward for agent A and agent B
// so we can think of A's expected reward, B's expected reward and social welfare plotted
// on the joint policy state space.
//
// A point attractor occurs when all Q-table entries equal the expected discounted reward
// of performing a given act in a given state (with the current policy). If the agent chooses the act with the highest
// expected reward, there is no policy that has better expected reward [does Q-learning always reach the global
// maximum, or does it sometimes reach a local maximum. Certainly the joint policy can get stuck in a local maximum.
// How is this affected in deep Q-learning, where the Q-table is approximated? For finite games, i think q-learning
// always finds the global max, but for infinite games perhaps not...does this matter? It affects the possible
// equilibria of a society with a large number of agents.
// For fixed environment (other agents have fixed policies) then a q-learning agent will find the optimal policy
// as long as it has exploration. However, more importantly, with all agents learning, a society will often not
// reach its nash equilibrium].
//
// A Nash equilibrium is a point attractor in joint policy space. i.e. for policy pair (Pa,Pb) on a point attractor
//   * If B has policy Pb then A's best policy is Pa
//   * If A has policy Pa then B's best policy is Pb
// In this case it makes no sense for A to deceive B into believing he has policy Pa while actually
// having a different policy since Pa would be A's best policy were B to believe A's deceit.
// Pa (and Pb) become deceit-free policies.
// Not only this, but if A adopts policy Pa then B cannot exploit A (where we define "B exploits A" as
// B increasing expected reward at the expense of A's expected reward when compared to the Nash equilibrium
// (Pa,Pb). So the policy pair (Pa,Pb) is also exploitation-free and either agent can unilaterally decide to
// take on a strategy on a Nash equilibrium without fear of exploitation (although there is still fear of
// irrationality of the other agent so that both are worse off, or equally hope that the other agent is irrationally
// self-exploiting)
//
// In the limit of an infinite number of agents all playing repeated prisoner's dilemma in randomly chosen pairs,
// the social state can be thought of as a probability distribution over individual agent policy. A stable society
// becomes a fixpoint in the space of probability distributions over the individual policy space.
// Clearly, the symmetrical 2-agent stable societies are also stable for a homogeneous society of any number
// of agents.
#ifndef MULTIAGENTGOVERNMENT_EXPERIMENT1_H
#define MULTIAGENTGOVERNMENT_EXPERIMENT1_H

#include <set>

#include "abm/abm.h"
#include "abm/agents/agents.h"
#include "DeselbyStd/stlstream.h"

inline void experiment1() {
    typedef abm::agents::PrisonersDilemmaAgent agent_type;

    constexpr int   NTIMESTEPS_TO_CONVERGENCE = 2000000; // number of timesteps without policy change to assume convergence
    constexpr int   MAXSTEPS =  40000000; // number of steps to give up looking for convergence

    std::set<long> stableSocieties;
    std::array<agent_type,2> agents;
    agents[0].connectTo(agents[1]);
    agents[1].connectTo(agents[0]);

    std::cout << "Initialised simulation" << std::endl;

    // scan through all possible policy pairs
    for(int policy0=0; policy0 < agent_type::policy_type::nPolicies(); ++policy0) {
        for(int policy1=0; policy1 < agent_type::policy_type::nPolicies(); ++policy1) {
//                std::cout << "Trying policy " << policy0 << ":" << policy1 << std::endl;
            agents[0].setPolicy(policy0);
            agents[1].setPolicy(policy1);
            long initialSociety = agents[0].policy.policyID() * agent_type::policy_type::nPolicies() + agents[1].policy.policyID();
            agent_type::schedule_type sim(agents[0].start() + agents[1].start());

            //// exec until no policy change for NTIMESTEPS or NTRANSITIONS policy transitions without convergence.
            sim.execUntil(
                    [&agents, &sim]() {
                        return (agents[0].policy.trainingStepsSinceLastPolicyChange > NTIMESTEPS_TO_CONVERGENCE &&
                                agents[1].policy.trainingStepsSinceLastPolicyChange > NTIMESTEPS_TO_CONVERGENCE) ||
                               sim.time() >= MAXSTEPS;
                    },
                    std::execution::seq);
//                std::cout << "Sim time = " << std::dec << sim.time() << std::endl;
            long finalSociety = agents[0].policy.policyID() * agent_type::policy_type::nPolicies() + agents[1].policy.policyID();
            if(sim.time() < MAXSTEPS) {
                std::cout << std::hex << initialSociety << " goes to point attractor :" << std::hex << finalSociety << std::endl;
                stableSocieties.insert(finalSociety);
            } else {
                std::cout << initialSociety <<  " goes to non-point attractor (or slow convergence)" << std::endl;
                long societyOnAttractor;
                std::cout << finalSociety << " -> ";
                do {
                    sim.execUntil([&agents]() {
                        return agents[0].policy.trainingStepsSinceLastPolicyChange == 0 || agents[1].policy.trainingStepsSinceLastPolicyChange == 0;
                    }, std::execution::seq);
                    societyOnAttractor = agents[0].policy.policyID() * agent_type::policy_type::nPolicies() + agents[1].policy.policyID();
                    std::cout << societyOnAttractor << " -> ";
                } while(societyOnAttractor != finalSociety);
                std::cout << std::endl;
            }
        }
    }
    std::cout << "The stable societies are: " << stableSocieties << std::endl;
}

#endif //MULTIAGENTGOVERNMENT_EXPERIMENT1_H
