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
        const int nEpisodes = 1000;

        typedef abm::bodies::CartPoleEnvironment body_type;
        auto mind = abm::minds::QMind {

                abm::DQN<body_type::dimension, body_type::action_type::size>(
                        mlpack::SimpleDQN<mlpack::MeanSquaredError, mlpack::HeInitialization>(64,32, body_type::action_type::size),
                        abm::RandomReplay(16, 128, body_type::dimension),
                        1.0),

                abm::GreedyPolicy(
                        0.5,
                        0.9999,
                        0.01)
        };


        auto agent = abm::Agent(body_type(),mind);

        for(int episode = 0; episode < nEpisodes; ++episode) {
            body_type::message_type message = agent.startEpisode();
            while(message != body_type::message_type::close) {
                message = agent.handleMessage(message);
            }
        }
    }
}

