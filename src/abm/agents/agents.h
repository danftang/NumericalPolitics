//
// Created by daniel on 15/05/23.
//

#ifndef MULTIAGENTGOVERNMENT_AGENTS_H
#define MULTIAGENTGOVERNMENT_AGENTS_H

#include <vector>
#include "../abm.h"

namespace abm {
    namespace agents {
        #include "PrisonersDilemmaAgent.h"
        #include "SimpleSugarSpiceAgent.h"
        #include "SugarSpiceAgentWithFriends.h"
        #include "ParallelPairingAgent.h"
        #include "SequentialPairingAgent.h"
    };
};

#endif //MULTIAGENTGOVERNMENT_AGENTS_H
