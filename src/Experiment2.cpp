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
    // community grows, previously stable, cooperative societies become unstable
    // and descend into defection.
    void tradingCommunity() {
        constexpr int MAX_TIMESTEPS = 40000000; // number of timesteps to give up searching for convergence
        constexpr int BURNIN_TIMESTEPS = 5000000;
        constexpr int SIM_TIMESTEPS = 5000000;
        constexpr int STARTPOPULATION = 2;
        constexpr int ENDPOPULATION = 1000;

        AgentPairer sentinel(STARTPOPULATION/2, 0x16);
        while(sentinel.agents.size() < ENDPOPULATION) {
            schedule_type sim = sentinel.start();
            sim.execUntil([&sim, &sentinel]() {
                return sim.time() >= BURNIN_TIMESTEPS;
            });
            sentinel.resetTotalRewardCounts();
            sim.execUntil([&sim, &sentinel]() {
                return sim.time() >= BURNIN_TIMESTEPS + SIM_TIMESTEPS;
            });

            std::pair<double,double> wellbeingMeanSD = sentinel.getRewardMeanAndSD(SIM_TIMESTEPS);
            std::cout << "Population of " << std::dec << sentinel.agents.size() << " agents ";
            std::cout << "wellbeing " << wellbeingMeanSD.first << " +- " << wellbeingMeanSD.second << std::endl;
//            if(sim.time() < MAX_TIMESTEPS) {
//                std::cout << "converged to " << std::hex << sentinel.getPopulationByPolicy() << std::endl;
//            } else {
//                std::cout << "did not converge. Current population: " << std::hex << sentinel.getPopulationByPolicy() << std::endl;
//            }
            sentinel.add2MoreAgents(0x16);
        }

    }
};