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

namespace tests {
    void zeroIntelligencePrisonersDilemma() {
        double pEndEpisode = 0.01;
        auto agent1 = abm::Agent(abm::bodies::PrisonersDilemmaBody(pEndEpisode), abm::minds::ZeroIntelligence{});
        auto agent2 = abm::Agent(abm::bodies::PrisonersDilemmaBody(pEndEpisode), abm::minds::ZeroIntelligence{});
        abm::episodes::runSync(agent1, agent2, abm::callbacks::Verbose{});
    }


}

#endif //MULTIAGENTGOVERNMENT_PRISONERSDILEMMA_H
