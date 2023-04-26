//
// Created by daniel on 07/04/23.
//

#include "Experiment2.h"
#include "DeselbyStd/stlstream.h"

namespace experiment2 {
    // execute trading communities of different sizes until convergence
    // to see at what point cooperation becomes impossible.
    //
    // This can also be thought of as a single growing community. As the
    // community grows, od previously stable, cooperative societies become unstable
    // and descend into defection.
    void tradingCommunity() {
        constexpr int MAX_TIMESTEPS = 40000000; // number of timesteps to give up searching for convergence
        const float     QMIN = 0;
        const float     QMAX = SugarSpiceAgent1::REWARD[1][0]/
                (1.0-abm::QTablePolicy<0,0>::DEFAULT_DISCOUNT); // Value of Q if all future rewards are max reward

        schedule_type sim;

        for(int nPairs = 10; nPairs < 100; ++nPairs) {
            AgentPairer sentinel(nPairs);
            // start with cooperating policy with distrust of strangers[?]
            for(SugarSpiceAgent1 &agent: sentinel.agents) {
                agent.policy.setPolicy(0x06, QMIN, QMAX);
            }

            sim = sentinel.start();
            sim.execUntil([&sim, &sentinel]() {
                return sentinel.hasConverged() || sim.time() >= MAX_TIMESTEPS;
            });
            std::cout << nPairs << " pairs ";
            if(sim.time() < MAX_TIMESTEPS) {
                std::cout << " converged to " << sentinel.getPopulationByPolicy() << std::endl;
            } else {
                std::cout << " did not converge " << std::endl;
            }
        }

    }
};