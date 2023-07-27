//
// Created by daniel on 25/07/23.
//
#include "../abm/DQN.h"
#include "../abm/Agent.h"
#include "../abm/GreedyPolicy.h"
#include "../abm/minds/QMind.h"
#include "../abm/bodies/CartPole.h"


namespace tests {
    void DQNCartPole() {
//        const int nEpisodes = 1000;

        typedef abm::bodies::CartPoleEnvironment body_type;

        auto mind = abm::minds::QMind {

                abm::DQN<body_type::dimension, body_type::action_type::size>(
                        mlpack::SimpleDQN<mlpack::MeanSquaredError, mlpack::HeInitialization>(100,50, body_type::action_type::size),
                        abm::RandomReplay(64, 100000, body_type::dimension),
                        5,
                        1.0),

                abm::GreedyPolicy(
                        0.1,
                        20000,
                        0.01)
        };


        auto agent = abm::Agent(body_type(),mind);

        double smoothedReturn = 20.0;
        int episodes = 0;
        int steps = 0;
        while(smoothedReturn < 198.99) {
            body_type::message_type message = agent.startEpisode();
            while(message != body_type::message_type::close) {
                message = agent.handleMessage(message);
                ++steps;
            }
            smoothedReturn = 0.95*smoothedReturn + 0.05*agent.rewardSinceLastEpisodeStart;
            std::cout << smoothedReturn << std::endl;
            ++episodes;
        }
        std::cout << "Learned to balance a pole in " << episodes << " episodes and " << steps << " steps" << std::endl;
    }
}

