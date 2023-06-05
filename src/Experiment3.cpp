//
// Created by daniel on 12/04/23.
//

#include "Experiment3.h"

namespace experiment3 {

    void QPrisonersDilemmaWithPunishingObserver() {
//        constexpr int NTIMESTEPS = 2000000;
//        constexpr int MAXPOLICYTRANSITIONS = 100000000;
////        constexpr int   NSAMPLES = 1000; // number of samples of policy space
//        const double QMIN = 0;
//        const double QMAX =
//                ObservedPrisonersDilemmaInterface::REWARD[1][0] /
//                (1.0 - abm::QTablePolicy<ObservedPrisonersDilemmaInterface>::DEFAULT_LEARNING_RATE);
//
//        std::set<long> stableSocieties;
//
//        // setup agents
//        std::array<ObservedPrisonersDilemmaAgent, 3> agents;
//        agents[0].interface.observerPhase = 0;
//        agents[0].interface.leftNeighbourMoves = &agents[2].interface.ownMoves;
//        agents[0].interface.rightNeighbourMoves = &agents[1].interface.ownMoves;
//        agents[1].interface.observerPhase = 1;
//        agents[1].interface.leftNeighbourMoves = &agents[0].interface.ownMoves;
//        agents[1].interface.rightNeighbourMoves = &agents[2].interface.ownMoves;
//        agents[2].interface.observerPhase = 2;
//        agents[2].interface.leftNeighbourMoves = &agents[1].interface.ownMoves;
//        agents[2].interface.rightNeighbourMoves = &agents[0].interface.ownMoves;
//
//        // sanity check what should be a stable society (always defect, never punish)
//        long alwaysDefectNeverPunish = 0b1111;
//        agents[0].policy.setPolicy(alwaysDefectNeverPunish, QMIN, QMAX);
//        agents[1].policy.setPolicy(alwaysDefectNeverPunish, QMIN, QMAX);
//        agents[2].policy.setPolicy(alwaysDefectNeverPunish, QMIN, QMAX);
//        Schedule <ObservedPrisonersDilemmaAgent::time_type> sim(agents);
//        sim.execUntil([&agents, &sim]() {
//            return agents[0].policyChangedLastStep || agents[1].policyChangedLastStep ||
//                   agents[2].policyChangedLastStep || sim.time() > NTIMESTEPS;
//        });
//        bool hasBeenPolicyChange =
//                agents[0].policyChangedLastStep || agents[1].policyChangedLastStep || agents[2].policyChangedLastStep;
//        std::cout << (hasBeenPolicyChange ? "Always defect never punish is not stable" : "Done sanity check")
//                  << std::endl;
//        std::cout << std::hex << agents[0].policy.policyID() << ":"
//                  << std::hex << agents[1].policy.policyID() << ":"
//                  << std::hex << agents[2].policy.policyID() << std::endl;
//        // check homogeneous societies for stability
//        // (the space of heterogeneous societies is too large)
//        long nPolicies = pow(ObservedPrisonersDilemmaAgent::interface_type::NACTIONS,
//                             ObservedPrisonersDilemmaAgent::interface_type::NSTATES);
////        for(int policyId=0; policyId < nPolicies; ++policyId) {
////            sim[0].policy.setPolicy(policyId, QMIN, QMAX);
////            sim[1].policy.setPolicy(policyId, QMIN, QMAX);
////            sim[2].policy.setPolicy(policyId, QMIN, QMAX);
////
////            //// find the policy after NTIMESTEPS of learning, and ensure that it has converged
////            bool hasBeenPolicyChange = sim.execUntilPolicyChange(NTIMESTEPS);
////
//////            long finalSociety = (sim[0].policy.policyID() * nPolicies + sim[1].policy.policyID())*nPolicies + sim[2].policy.policyID();
////            if (!hasBeenPolicyChange) {
////                std::cout << "Found point attractor :" << std::hex << policyId << std::endl;
////                stableSocieties.insert(policyId);
////            } else {
////                std::cout << policyId << " is not on a point attractor" << std::endl;
////            }
////        }
//        std::cout << "The stable societies are: " << stableSocieties << std::endl;
    }




//    void run() {
//        constexpr int NTIMESTEPS = 10;
//        constexpr int NSTATES = 10;
//        constexpr int NACTS = 2;
//        typedef RoundRobinPrisonersDilemmaAgent<NTIMESTEPS> QAgent;
//
//        const int NAGENTS = 1000;
//        std::uniform_real_distribution<double> discount(0.25,1.0);
//
//        //// populate ABM
//        abm::AgentBasedModel model;
//        // set up a round-robin playing schedule
//        QAgent *lastAagent = nullptr;
//        QAgent *lastBagent = nullptr;
//        for(int agentPair=0; agentPair<NAGENTS/2; ++agentPair) {
//            QAgent *agentA = new QAgent(true);
//            QAgent *agentB = new QAgent(true);
//            agentA->opponent = agentB;
//            agentB->opponent =  agentA;
//            if(lastAagent != nullptr) lastAagent->nextAgent = agentA;
//            if(lastBagent == nullptr) lastBagent = agentA;
//            agentB->nextAgent = lastBagent;
//            lastAagent = agentA;
//            lastBagent = agentB;
//            model.insert({agentA, agentB});
//        }
//        lastAagent->nextAgent = lastBagent;
//
//        //// Train agents to equilibrium with no law
//
//
//        // start with no perturbation
//        // train agents to equilibrium
//    }
}