//
// Created by daniel on 04/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_PRISONERSDILEMMA_H
#define MULTIAGENTGOVERNMENT_PRISONERSDILEMMA_H

#include "../abm/minds/ZeroIntelligence.h"
#include "../abm/bodies/PrisonersDilemmaBody.h"
#include "../abm/Agent.h"
#include "../abm/episodes/SimpleEpisode.h"
#include "../abm/minds/QMind.h"
#include "../abm/minds/qLearning/QTable.h"
#include "../abm/minds/qLearning/GreedyPolicy.h"
#include "../abm/minds/LambdaMind.h"

namespace tests {
    void zeroIntelligencePrisonersDilemma() {
        double pEndEpisode = 0.01;
        auto agent1 = abm::Agent(abm::bodies::PrisonersDilemmaBody(pEndEpisode), abm::minds::ZeroIntelligence{});
        auto agent2 = abm::Agent(abm::bodies::PrisonersDilemmaBody(pEndEpisode), abm::minds::ZeroIntelligence{});
        abm::episodes::runSync(agent1, agent2, abm::callbacks::Verbose{});
    }

    void tabularQMindPrisonersDilemma() {
        double pEndEpisode = 0.0001;
        auto agent1 = abm::Agent(
                abm::bodies::PrisonersDilemmaBody(pEndEpisode),
                abm::minds::QMind(
                        abm::minds::QTable<4,2>(1.0),
                        abm::minds::GreedyPolicy(abm::explorationStrategies::LinearDecay(0.25, 20000, 0.05))
                        )
                );
        auto agent2 = abm::Agent(
                abm::bodies::PrisonersDilemmaBody(pEndEpisode),
                abm::minds::QMind(
                        abm::minds::QTable<4,2>(1.0),
                        abm::minds::GreedyPolicy(abm::explorationStrategies::LinearDecay(0.25, 20000, 0.05))
                )
//                abm::minds::LambdaMind([](auto body) -> size_t { return 0; })
        );
        auto verboseCallback = abm::callbacks::Verbose{};
        for(int i=0; i < 10; ++i) {
            abm::episodes::runSync(agent1, agent2, verboseCallback);
        }
    }


}

#endif //MULTIAGENTGOVERNMENT_PRISONERSDILEMMA_H
