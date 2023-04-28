//
// Created by daniel on 27/02/23.
//

#include "Experiment1.h"

#include <random>
#include "abm/abm.h"
#include "DeselbyStd/stlstream.h"

namespace experiment1 {

    // Background:
    // A repeated prisoner's dilemma, where each agent remembers its own and other's last move
    // with online Q-learning. The "social state" is the 2 x 8 = 16 Q values of the pair of
    // Q-table entries, so we can embed this into a 16 dimensional phase space and consider the
    // structure of the attractors (fixed point probability distribution) in this space.
    //
    //
    // Questions:
    //   * How many attractors are there?
    //   * Are they all point attractors?
    //
    // Approach:
    // There are only 16 possible agent policies so only 256 possible policy pairs, so we can split
    // the social space into 256 partitions. For each partition, we can start with a strong bias
    // towards this policy pair and execute Q-learning until convergence. If an attractor exists
    // in this partition, the agents should converge to it.
    //
    // We can test for convergence by running until no policy changes occur for a "large" number
    // of steps.
    //
    // Results:
    // All start points eventually converge to a point attractor. However, convergence is quite slow.
    // There are 5 attracting societies:
    //  1) 0x62/0x26
    //  2) 0x66
    //  3) 0xfe/0xef
    //  4) 0xee
    //  5) 0xff
    //
    // Societies 0xff and 0xfe quickly lead to mutual defection. It is questionable that 0xfe is truly stable.
    // Society 0xee has a stable equilibrium of mutual defection and an unstable equilibrium of mutual cooperation.
    // Interestingly 0x66 and 0x62 quickly lead to mutual cooperation.
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
    // equilibria of a society with a large number of agents].
    //
    // A point attractor is a Nash equilibrium with respect to policy. i.e. for policy pair (Pa,Pb) on a point attractor
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

    void QPrisonersDilemma() {
        constexpr int   NTIMESTEPS = 2000000; // number of timesteps without policy change to assume convergence
        constexpr int   MAXSTEPS =  40000000; // number of steps to give up looking for convergence
        const float     QMIN = 0;
        const float     QMAX =
                PrisonersDilemmaAgent::REWARD[1][0]/
                (1.0-abm::QTablePolicy<0,0>::DEFAULT_DISCOUNT); // Value of Q if all future rewards are max reward

        std::set<long> stableSocieties;
        std::array<PrisonersDilemmaAgent,2> agents;
        agents[0].connectTo(agents[1]);
        agents[1].connectTo(agents[0]);

        std::cout << "Initialised simulation" << std::endl;

        // scan through all possible policy pairs
        for(int policy0=0; policy0 < PrisonersDilemmaAgent::policy_type::nPolicies(); ++policy0) {
            for(int policy1=0; policy1 < PrisonersDilemmaAgent::policy_type::nPolicies(); ++policy1) {
//                std::cout << "Trying policy " << policy0 << ":" << policy1 << std::endl;
                agents[0].policy.setPolicy(policy0, QMIN, QMAX);
                agents[1].policy.setPolicy(policy1, QMIN, QMAX);
                long initialSociety = agents[0].policy.policyID() * PrisonersDilemmaAgent::policy_type::nPolicies() + agents[1].policy.policyID();
                PrisonersDilemmaAgent::schedule_type sim(agents[0].start() + agents[1].start());

                //// exec until no policy change for NTIMESTEPS or NTRANSITIONS policy transitions without convergence.
                sim.execUntil(
                        [&agents, &sim]() {
                            return (agents[0].policy.trainingStepsSinceLastPolicyChange > NTIMESTEPS &&
                                    agents[1].policy.trainingStepsSinceLastPolicyChange > NTIMESTEPS) ||
                            sim.time() >= MAXSTEPS;
                        },
                        std::execution::seq);
//                std::cout << "Sim time = " << std::dec << sim.time() << std::endl;
                long finalSociety = agents[0].policy.policyID() * PrisonersDilemmaAgent::policy_type::nPolicies() + agents[1].policy.policyID();
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
                        societyOnAttractor = agents[0].policy.policyID() * PrisonersDilemmaAgent::policy_type::nPolicies() + agents[1].policy.policyID();
                        std::cout << societyOnAttractor << " -> ";
                    } while(societyOnAttractor != finalSociety);
                    std::cout << std::endl;
                }
            }
        }
        std::cout << "The stable societies are: " << stableSocieties << std::endl;
    }

}
