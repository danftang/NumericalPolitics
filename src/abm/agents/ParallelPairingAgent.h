// An agent that generates random 1-to-1 mappings between agents
// and calls the agent's handleNewOpponent method.
// Note that for a 1-to-1 mapping there must be an even number of agents
//
// Created by daniel on 15/05/23.
//

#ifndef MULTIAGENTGOVERNMENT_AGENTPAIRINGAGENT_H
#define MULTIAGENTGOVERNMENT_AGENTPAIRINGAGENT_H

#include <vector>
#include "../abm.h"

namespace abm {
    namespace agents {

        template<class AGENT>
        class ParallelPairingAgent {
        public:
            static constexpr int NTIMESTEPS_TO_CONVERGENCE = 2000000;

            typedef AGENT::time_type time_type;
            typedef AGENT::schedule_type schedule_type;

            std::vector<AGENT> agents;
            std::vector<int> agentOrdering;
            CommunicationChannel<schedule_type, void> selfLoop;

            ParallelPairingAgent(int nAgentsDiv2, long initialPolicyId) : agents(nAgentsDiv2 * 2),
                                                                          agentOrdering(nAgentsDiv2 * 2) {
                selfLoop.connectTo(*this, &ParallelPairingAgent<AGENT>::handlePairing, 2);
                for (int i = 0; i < nAgentsDiv2 * 2; ++i) {
                    agentOrdering[i] = i;
                    agents[i].setPolicy(initialPolicyId);
                }
            }

            schedule_type start() {
                return handlePairing(0);
            }

            schedule_type handlePairing(time_type time) {
                std::random_shuffle(agentOrdering.begin(), agentOrdering.end());
                auto agentp = agentOrdering.begin();
                schedule_type schedule;
                while (agentp != agentOrdering.end()) {
                    AGENT &agent1 = agents[*agentp];
                    ++agentp;
                    AGENT &agent2 = agents[*agentp];
                    ++agentp;
                    schedule += agent1.handleNewOpponent(agent2, time);
                    schedule += agent2.handleNewOpponent(agent1, time);
                }
                schedule += selfLoop(time);
                return schedule;
            }

            void add2MoreAgents(int initialPolicyId) {
                for (int i = 0; i < 2; ++i) {
                    agents.emplace_back();
                    agents.back().setPolicy(initialPolicyId);
                    agentOrdering.push_back(agentOrdering.size());
                }
            }

            bool hasConverged() {
                for (const AGENT &agent: agents) {
                    if (agent.policy.trainingStepsSinceLastPolicyChange < NTIMESTEPS_TO_CONVERGENCE) return false;
                }
                return true;
            }

            std::map<long, int> getPopulationByPolicy() {
                std::map<long, int> population;
                for (const AGENT &agent: agents) {
                    population[agent.policy.policyID()] += 1;
                }
                return population;
            }

            // sum (m + d)(m + d) = sum m^2 + 2md_i + d_i^2
            // = Nm^2 + sum d_i^2
            std::pair<double, double> getRewardMeanAndSD(time_type time) {
                double sumReward = 0.0;
                double sumRewardSq = 0.0;
                double NrewardSamples = time / 2.0;
                for (const AGENT &agent: agents) {
                    double meanReward = agent.totalReward / NrewardSamples;
                    sumReward += meanReward;
                    sumRewardSq += meanReward * meanReward;
                }
                return {sumReward / agents.size(),
                        sqrt((sumRewardSq - sumReward * sumReward / agents.size()) / agents.size())};
            }

            void resetTotalRewardCounts() {
                for (AGENT &agent: agents) {
                    agent.totalReward = 0.0;
                }
            }
        };
    }
}

#endif //MULTIAGENTGOVERNMENT_AGENTPAIRINGAGENT_H
