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
#include "../abm/lossFunctions/QLearningLoss.h"
#include "../abm/bodies/CartPole.h"

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

        using namespace abm;

        const size_t bufferSize = 100000;
        const size_t batchSize = 64;
        const double discount = 1.0;
        const size_t endSatateFnnUpdateInterval = 5;
        const size_t explorationBurnin = 100;
        const double initialExploration = 0.1;
        const double finalExploration = 0.01;
        const size_t nExplorationSteps = 20000;
        const double updateStepSize = 0.001;

        approximators::FNN fnn(mlpack::GaussianInitialization(),
                abm::bodies::CartPole::dimension,
                mlpack::Linear(100),
                mlpack::ReLU(),
                mlpack::Linear(50),
                mlpack::ReLU(),
                mlpack::Linear(abm::bodies::CartPole::action_type::size)
                );

        auto burnInThenTrainEveryStep = [burnin = 2]<class BODY>(const events::PreActBodyState<BODY> & /* event */) mutable {
            if(burnin > 0) --burnin;
            return burnin == 0;
        };

        approximators::DifferentialTrainingPolicy trainingPolicy(
                ens::AdamUpdate(),
                updateStepSize,
                fnn.parameters().n_rows,
                fnn.parameters().n_cols,
                burnInThenTrainEveryStep);

        lossFunctions::QLearningLoss loss(
                bufferSize,
                abm::bodies::CartPole::dimension,
                batchSize,
                discount,
                fnn,
                endSatateFnnUpdateInterval);

        auto mind = abm::minds::QMind(
                approximators::AdaptiveFunction(
                        std::move(fnn),
                        std::move(trainingPolicy),
                        std::move(loss)),
                abm::minds::GreedyPolicy(
                        abm::explorationStrategies::BurninThenLinearDecay(
                                explorationBurnin,
                                initialExploration,
                                nExplorationSteps,
                                finalExploration))
                );

//        abm::Agent agent(abm::bodies::CartPole(), std::move(mind));
        abm::Agent agent(abm::bodies::CartPole(), mind);

        abm::callbacks::RewardPerEpisode rewardCallback;

        double smoothedReward = 0.0;
        for(int episode = 0; episode < 500; ++episode) {
            auto startTimer = std::chrono::system_clock::now();
            agent.runEpisode(rewardCallback);
            auto endTimer = std::chrono::system_clock::now();
            smoothedReward = 0.95*smoothedReward + 0.05*rewardCallback.rewardThisEpisode;
            std::cout << episode << " " << smoothedReward << " " << rewardCallback.rewardThisEpisode << " " << endTimer-startTimer << std::endl;
        }

    }
}

