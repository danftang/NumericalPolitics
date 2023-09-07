//
// Created by daniel on 29/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_DEEPIIMCTS_H
#define MULTIAGENTGOVERNMENT_DEEPIIMCTS_H

#include "../abm/minds/IncompleteInformationMCTS.h"
#include "../abm/minds/QMind.h"
#include "../abm/DQN.h"
#include "../abm/bodies/SugarSpiceTradingBody.h"

namespace tests {
    void deepIIMCTS() {
        auto dqnMind = 0;
        auto deepTreeMind = abm::minds::IncompleteInformationMCTS<abm::bodies::SugarSpiceTradingBody<true>,>();
    }
}

#endif //MULTIAGENTGOVERNMENT_DEEPIIMCTS_H
