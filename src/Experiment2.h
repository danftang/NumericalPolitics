// Background:
// When we expand the simulation to many agents stable societies
// can be defined as populations of policies. If communication channels aren't fixed, then a
// society may only be stable under a certain set of initial channel configurations. These can be
// specified, or we can just say that there exists some non-empty set under which the policy
// population is stable.
//
// A "random encounter" society is one where agents encounter other agents at random
// drawn from a fixed distribution of agent pairs. In this case each agent is effectively in a two-agent
// game with an opponent who has a policy that is a probabilistic mix of policies (i.e. a policy is chosen at
// the beginning of each "encounter" and held fixed until the end of the encounter). If each agent's policy
// is stable with respect to the mixed policy of its encounters then the society is stable (and vice-versa).
//
// A "uniform random encounter" society is a special case of a random encounter society where all agent pairs
// are equally probable. In this case, when the number of agents is large, each agent's opponent's mixed policy
// is approximately the same, so all policies in the society must be stable with respect to this mixed policy.
// So, we need to find mixed policies such that all members in the mix are stable against the mix.
//
// Suppose we have a trading society, "sugar and spice scape", where agents collect sugar and spice, but some
// agents can only collect sugar and some only spice. Each agent derives a reward
// by eating a certain combination of sugar and spice, let's say 50/50 is max reward. On encountering another agent,
// they can choose to offer to swap sugar for spice or try to steal the other's sugar/spice if offered,
// thus creating a prisoner's dilemma. Suppose half of agents can collect sugar and half collect spice.
//
// If an encounter involves only one iteration of prisoner's dilemma and agents are indistinguishable then
// the best strategy is to defect irrespective of the strategy of other agents, so there is only one
// attractor: the world where everyone always defects. So, what can we change about the world in order to
// make it better?
//
// Questions:
//
//  * If agents can identify n other agents and remember the last n encounters, then in the limit of infinite
//    agents, we have the single encounter situation and a descent into total defection, while in the limit
//    of only two agents we have repeated prisoner's dilemma and a possible escape. So, stable societies
//    can be sensitive to population. How does the population change the stability with respect to the agent's
//    memory size and/or lifespan? [i.e. probability of encountering a stranger...so we can distinguish
//    between tight-knit communities and loose-knit communities. ] Also, does experience with known agents
//    influence behaviour with strangers (i.e. can the agents learn to treat strangers in a way that will make
//    them into cooperating "friends")
//
//  * If each encounter involves three agents: two traders and a mediator, can the mediator learn to punish
//    defection in order to change the game to a pure mutual interest game (where the mediator is subsequently
//    a player)?
//      - If so, can agents learn to trade only when there is a mediator present? In a spatial simulation
//        does this result in the emergence of "market towns" if players can also be mediators of other games?
//
//  * If agents can choose whether to spray other agents red/white after an encounter, can they collectively learn
//    to spray defectors while at the same time learning to distrust red agents?
//
//  * If agents can communicate information about other agents, can they learn to warn other agents about
//    defectors?
//
// Approach:
//
// Starting with a 2-agent society, add agents two at a time and form a uniform, random pairing
// between agents for trading, observing social equilibrium after each addition.
// Try this with different agent abilities (memory of other agents, memory of past
// experience with strangers, with/without parental teaching)
//
// Created by daniel on 07/04/23.
//

#ifndef MULTIAGENTGOVERNMENT_EXPERIMENT2_H
#define MULTIAGENTGOVERNMENT_EXPERIMENT2_H

#include <cstdlib>
#include "abm/abm.h"
#include "abm/agents/agents.h"
#include <random>

template<class AGENT> void experiment2(int startPopulation, int endPopulation, int burninTimesteps, int simTimesteps);

// Growing society with agents that can only remember their last encounter
// Results:
// Even at a population of 3, the likelihood that a given agent will have
// n consecutive encounters with the same agent is 2^(1-n), so society
// eventually breaks down into defection (though somewhat slowly).
inline void experiment2a() {
    experiment2<abm::agents::SimpleSugarSpiceAgent>(2, 3, 200000000, 10000000);
}

// Test agents that can identify and remember the last encounter with the 3 most
// recently met agents.
// Results:
// Up to 3, we have simple repeated prisoner's dilemma and all agents reach policy 6.
// However, since strangers become friends (for the next 3 turns) in a small society
// it pays to cooperate, so cooperation persists beyond the memory size of the agents.
// However, as the probability that a remembered agent will be re-encounter reduces, so we're
// back to single-shot prisoner's dilemma.
inline void experiment2b() {
    experiment2<abm::agents::SugarSpiceAgentWithFriends>(2, 12, 50000000, 10000000);
}

//

template<class AGENT>
void experiment2(int startPopulatinon, int endPopulation, int burninTimesteps, int simTimesteps) {

//        typedef abm::agents::SugarSpiceAgent1 agent_type;

    abm::agents::SequentialPairingAgent<AGENT> rootAgent(startPopulatinon);
    while (rootAgent.agents.size() <= endPopulation) {
        std::cout << "Population of " << std::dec << rootAgent.agents.size() << " agents:" << std::endl;
        typename AGENT::schedule_type sim = rootAgent.start();
        sim.execUntil([&sim, &rootAgent, burninTimesteps]() {
            return sim.time() >= burninTimesteps;
        });
        std::cout << "  Policy population at end of burnin: " << std::hex << rootAgent.getPopulationByPolicy() << std::endl;
        rootAgent.resetAllTrainingStats();
        sim.execUntil([&sim, &rootAgent, maxTimesteps = burninTimesteps + simTimesteps]() {
            return sim.time() >= maxTimesteps;
        });

        std::pair<double, double> wellbeingMeanSD = rootAgent.getRewardMeanAndSD();
        std::cout << "  Wellbeing " << wellbeingMeanSD.first << " +- " << wellbeingMeanSD.second << std::endl;
        std::cout << "  Policy population: " << std::hex << rootAgent.getPopulationByPolicy() << std::endl;
//            if(sim.time() < MAX_TIMESTEPS) {
//                std::cout << "converged to " << std::hex << sentinel.getPopulationByPolicy() << std::endl;
//            } else {
//                std::cout << "did not converge. Current population: " << std::hex << sentinel.getPopulationByPolicy() << std::endl;
//            }
        rootAgent.addAgent();
    }
}


#endif //MULTIAGENTGOVERNMENT_EXPERIMENT2_H
