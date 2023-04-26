//
// Created by daniel on 07/04/23.
//

#ifndef MULTIAGENTGOVERNMENT_EXPERIMENT2_H
#define MULTIAGENTGOVERNMENT_EXPERIMENT2_H

#include <cstdlib>
#include "abm/abm.h"
#include <random>

namespace experiment2 {
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
    //  * If agents can identify other agents and remember previous encounters, then in the limit of infinite
    //    agents, we have the single encounter situation and a descent into total defection, while in the limit
    //    of only two agents we have repeated prisoner's dilemma and a possible escape. So, stable societies
    //    are sensitive to population. How does the population change the stability with respect to the agent's
    //    memory size and/or lifespan? [i.e. probability of encountering a stranger...so we can distinguish
    //    between tight-knit communities and loose-knit communities. ]
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
    typedef ulong                       time_type;
    typedef abm::Schedule<time_type>    schedule_type;

    class SugarSpiceAgent1 {
    public:
        static constexpr size_t NSTATES  = 5; // {stranger, cooperate/cooperate, cooperate/defect, defect/cooperate, defect,defect}
        static constexpr size_t NACTIONS = 2;
        static constexpr float  REWARD[2][2] = {{3, 0},
                                                {4, 1}};
        static constexpr int MEMORY_SIZE = 10;

//        typedef ulong                               time_type;
        typedef abm::QTablePolicy<NSTATES,NACTIONS> policy_type;
//        typedef abm::Schedule<time_type>            schedule_type;

        std::map<SugarSpiceAgent1 *, int>          memory; // map from previously encountered agent to last game play
        int myLastMove;
        int lastState;
        int lastAction;
        std::map<SugarSpiceAgent1 *,int>::iterator currentOpponentIt;

        policy_type     policy;
        abm::CommunicationChannel<abm::Schedule<time_type>, int> opponent;

        void connectTo(SugarSpiceAgent1 &opponentAgent) {
            opponent.connectTo(opponentAgent, &SugarSpiceAgent1::handleOpponentMove, 1);
        }

        schedule_type start() {
            myLastMove = 0;
            return opponent.send(myLastMove, 0);
        }

        schedule_type handleNewOpponent(SugarSpiceAgent1 &newOpponent, time_type time) {
            connectTo(newOpponent);
            currentOpponentIt = memory.find(&newOpponent);
            if(currentOpponentIt == memory.end()) {
                lastState = 4;
                if(memory.size() < MEMORY_SIZE) currentOpponentIt = memory.insert(std::pair(&newOpponent, 4)).first;
            } else {
                lastState = currentOpponentIt->second;
            }
            lastAction = policy.getAction(lastState);
            return opponent.send(lastAction, time);
        }


        schedule_type handleOpponentMove(int opponentsMove, time_type time) {
//            std::cout << "Handling " << opponentsMove << ", " << myLastMove << std::endl;
            int newState = 2*myLastMove + opponentsMove;
            policy.train(lastState, lastAction, REWARD[myLastMove][opponentsMove], newState);
            if(currentOpponentIt != memory.end()) currentOpponentIt->second = newState;
            return abm::Schedule<time_type>();
        }
    };


    // generates random 1-to-1 mappings between agents
    // and calls the agent's handleNewOpponent method
    class AgentPairer {
    public:
        static constexpr int NTIMESTEPS_TO_CONVERGENCE = 200000;

        typedef SugarSpiceAgent1    agent_type;

        std::vector<agent_type>             agents;
        std::vector<int>                    agentOrdering;
        abm::CommunicationChannel<schedule_type, void> selfLoop;

        AgentPairer(int nAgentsDiv2): agents(nAgentsDiv2*2), agentOrdering(nAgentsDiv2*2) {
            selfLoop.connectTo(*this, &AgentPairer::handlePairing, 2);
            for(int i=0; i<nAgentsDiv2*2; ++i) {
                agentOrdering[i] = i;
            }
        }

        schedule_type start() {
            return handlePairing(0);
        }

        schedule_type handlePairing(time_type time) {
            std::random_shuffle(agentOrdering.begin(), agentOrdering.end());
            auto agentp = agentOrdering.begin();
            schedule_type schedule;
            while(agentp != agentOrdering.end()) {
                agent_type &agent1 = agents[*agentp];
                ++agentp;
                agent_type &agent2 = agents[*agentp];
                ++agentp;
                schedule += agent1.handleNewOpponent(agent2, time);
                schedule += agent2.handleNewOpponent(agent1, time);
            }
            schedule += selfLoop(time);
            return schedule;
        }

        bool hasConverged() {
            for(const agent_type &agent: agents) {
                if(agent.policy.stepsSinceLastPolicyChange < NTIMESTEPS_TO_CONVERGENCE) return false;
            }
            return true;
        }

        std::map<long, int> getPopulationByPolicy() {
            std::map<long,int> population;
            for(const agent_type &agent: agents) {
                population[agent.policy.policyID()] += 1;
            }
            return population;
        }
    };



};


#endif //MULTIAGENTGOVERNMENT_EXPERIMENT2_H
