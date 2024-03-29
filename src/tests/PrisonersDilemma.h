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
        double pEndEpisode = 0.001;
        auto agent1 = abm::Agent(
                abm::bodies::PrisonersDilemmaBody(pEndEpisode),
                abm::minds::QMind(
                        abm::minds::QTable<4,2>(1.0),
                        abm::minds::GreedyPolicy(abm::explorationStrategies::LinearDecay(0.25, 20000, 0.05))
                        )
                );
        auto agent2 = agent1;

        // train
        std::cout << "Training..." << std::endl;
        for(int i=0; i < 10000; ++i) {
            abm::episodes::runSync(agent1, agent2);
        }

        // show
        auto verboseCallback = abm::callbacks::Verbose{};
        abm::episodes::runSync(agent1, agent2, verboseCallback);
    }


//    void neuralQMindPrisonersDilemma() {
//        double pEndEpisode = 0.001;
//        auto agent1 = abm::Agent(
//                abm::bodies::PrisonersDilemmaBody(pEndEpisode),
//                abm::minds::QMind(
//                        abm::minds::LambdaMind(
//                                approximators::FeedForwardNeuralNet(
//                                        {new mlpack::Linear(100),
//                                        new mlpack::ReLU()})
//                                ),
//                        abm::minds::GreedyPolicy(abm::explorationStrategies::LinearDecay(0.25, 20000, 0.05))
//                )
//        );
//        auto agent2 = agent1;
//
//        // train
//        std::cout << "Training..." << std::endl;
//        for(int i=0; i < 10000; ++i) {
//            abm::episodes::runSync(agent1, agent2);
//        }
//
//        // show
//        auto verboseCallback = abm::callbacks::Verbose{};
//        abm::episodes::runSync(agent1, agent2, verboseCallback);
//    }


}

#endif //MULTIAGENTGOVERNMENT_PRISONERSDILEMMA_H
