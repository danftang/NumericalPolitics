//
// Created by daniel on 18/07/23.
//

#include "../abm/societies/RandomEncounterSociety.h"
#include "../abm/agents/PingPongAgent.h"
#include "tests.h"

void pingPongTest() {
    abm::societies::RandomEncounterSociety<abm::agents::PingPongAgent> soc(2);
    soc.verbose = true;
    soc.episode();
}