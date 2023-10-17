//
// Created by daniel on 25/07/23.
//
//#include "../abm/DQN.h"
//#include "../abm/Agent.h"
//#include "../abm/minds/qLearning/GreedyPolicy.h"
//#include "../abm/minds/QMind.h"
//#include "../tests/CartPole.h"

#include "mlpack.hpp"
#include "../abm/approximators/AdaptiveFunction.h"
#include "../abm/approximators/FNN.h"

namespace tests {
    void DQNCartPole() {
//        auto mind = abm::minds::QMind {
//
//                abm::DQN<tests::CartPoleEnvironment::dimension, tests::CartPoleEnvironment::action_type::size>(
//                        mlpack::SimpleDQN<mlpack::MeanSquaredError, mlpack::HeInitialization>(100,50, tests::CartPoleEnvironment::action_type::size),
//                        abm::RandomReplay(64, 100000, tests::CartPoleEnvironment::dimension),
//                        5,
//                        1.0),
//
//                abm::minds::GreedyPolicy(
//                        abm::explorationStrategies::LinearDecay(
//                            0.5, // 0.1,
//                            50000, //20000,
//                            0.01)
//                )
//        };
//
//        tests::CartPoleEnvironment cartPole;
//
//        int episodes = 0;
////        int steps = 0;
//        double meanEpisodeLength = 0.0;
//        double meanDecay = 0.98;
//        while(meanEpisodeLength < 198.99) {
//            int episodeLength = cartPole.episode(mind);
//            meanEpisodeLength = meanEpisodeLength*meanDecay + (1.0-meanDecay)*episodeLength;
//            std::cout << meanEpisodeLength << std::endl;
//            ++episodes;
//        }
//        std::cout << "Learned to balance a pole in " << episodes << " episodes" << std::endl;

        using namespace abm::approximators;

        FNN fnn(20, new mlpack::Linear(20));
        DifferentiableOptimisableFunction net(fnn, ens::AdamUpdate(), 0.001);
    }
}

