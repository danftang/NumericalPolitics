// An agent that randomly pairs agents for binary interaction one at a time
// so that there is one interaction per unit time.
// Unlike the ParallelPairingAgent, this can create pairs between any number
// of agents (not just even numbers).
//
// Created by daniel on 15/05/23.
//

#ifndef MULTIAGENTGOVERNMENT_SEQUENTIALPAIRINGAGENT_H
#define MULTIAGENTGOVERNMENT_SEQUENTIALPAIRINGAGENT_H

#include <vector>
#include "../CommunicationChannel.h"
#include "../../DeselbyStd/random.h"

namespace abm {
    namespace agents {

        template<class AGENT>
        class SequentialPairingAgent {
        public:
            static constexpr int NTIMESTEPS_TO_CONVERGENCE = 2000000;

            typedef AGENT::time_type time_type;
            typedef AGENT::schedule_type schedule_type;

            std::vector<AGENT> agents;
            CommunicationChannel<schedule_type, void> selfLoop;

            SequentialPairingAgent(int nAgents) : agents(nAgents) {
                selfLoop.connectTo(*this, &SequentialPairingAgent<AGENT>::handlePairing, 2);
            }

            schedule_type start() {
                return handlePairing(0);
            }

            schedule_type handlePairing(time_type time) {
                int agent1Index = deselby::Random::nextInt(0, agents.size());
                int agent2Index = deselby::Random::nextInt(0, agents.size() - 1);
                if (agent2Index >= agent1Index) agent2Index += 1;

                AGENT &agent1 = agents[agent1Index];
                AGENT &agent2 = agents[agent2Index];
                schedule_type schedule;
                schedule += agent1.handleNewOpponent(agent2, time);
                schedule += agent2.handleNewOpponent(agent1, time);
                schedule += selfLoop(time);
                return schedule;
            }

            void addAgent() {
                agents.emplace_back();
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
            std::pair<double, double> getRewardMeanAndSD() {
                double sumReward = 0.0;
                double sumRewardSq = 0.0;
                for (const AGENT &agent: agents) {
                    double meanReward = agent.policy.getMeanReward();
                    sumReward += meanReward;
                    sumRewardSq += meanReward * meanReward;
                }
                return {sumReward / agents.size(),
                        sqrt((sumRewardSq - sumReward * sumReward / agents.size()) / agents.size())};
            }

            void resetAllTrainingStats() {
                for (AGENT &agent: agents) {
                    agent.policy.resetTrainingStats();
                }
            }
        };
    }
}

#endif //MULTIAGENTGOVERNMENT_SEQUENTIALPAIRINGAGENT_H
