// Question:
//   Under what conditions can agents form a shared language? i.e. attach a shared semantics to an arbitrary
//   set of symbols.
// Hypothesis:
//    - Pairs of Q-learning agents can form a language, but that language may not be shared (each agent
//      must learn the semantics of the other).
//    - Q-learning agents in a random-encounter society can form a shared language, and the existence of
//      an established society allows new-born agents to learn the shared language faster than in the
//      two-agent case.
//    - Pure Monte-Carlo tree search agents cannot reliably form a shared language in either paired or
//      social situations.
//    - Agents who have the ability to copy the behaviour of other agents can form a shared language faster
//      than Q-learning agents in both the two-agent and random-encounter society scenarios.
//
// Method:
//   Agents play a turns-based game of "guess the number". A first mover is chosen at random
//   and initialised with an number in 1...N, unknown to the other agent. The first mover must then pass
//   a symbol in a language, L. The second mover must then guess which number the first mover was given.
//   If the second mover guesses correctly both agents get reward 1, otherwise both get no reward.
//
// Discussion:
//   If there are N symbols in the language, there are N!^2 optimal 2-agent strategies (each agent can have one of N!
//   semantic interpretations of the language) but only N! of these are classed as languages (i.e. both agents share
//   the same semantics). Pure Monte-Carlo tree search would allow each agent to
//   find a joint optimal strategy, but two Monte-Carlo tree search agents wouldn't be able to align their strategies,
//   let alone find a shared langauge.
//   Q-learners would find a joint optimal strategy, but not necessarily a shared language.
//
// Created by daniel on 19/12/23.
//

#ifndef MULTIAGENTGOVERNMENT_EXPERIMENT6_H
#define MULTIAGENTGOVERNMENT_EXPERIMENT6_H

#include <cstddef>
#include "abm/lossFunctions/QLearningLoss.h"
#include "abm/Agent.h"
#include "abm/bodies/GuessTheNumberBody.h"
#include "abm/minds/QMind.h"
#include "abm/approximators/FNN.h"
#include "abm/approximators/AdaptiveFunction.h"
#include "abm/minds/qLearning/GreedyPolicy.h"

namespace experiment6 {
    typedef abm::bodies::GuessTheNumberBody body_type;

    class GameLogger {
    public:
        static constexpr uint N = body_type::action_type::size; // language/state size
        uint hintStats[N][N];
        uint guessStats[N][N];
        uint nWins;
        uint nGames;

        GameLogger() {
            for(int i=0; i<N; ++i) {
                for(int j=0; j<N; ++j) {
                    hintStats[i][j] = 0;
                    guessStats[i][j] = 0;
                }
            }
            nWins = 0;
            nGames = 0;
        }

        template<class M1, class M2>
        void on(const abm::events::Message<abm::Agent<body_type,M1>, abm::Agent<body_type,M2>, body_type::message_type> &event) {
            if(event.source.body.iAmGuesser) {
                ++guessStats[event.source.body.state][event.message];
            } else {
                if(event.dest.body.iHavePlayed) {
                    assert(event.message < 2);
                    ++nGames;
                    if(event.message == 1) ++nWins;
                } else {
                    ++hintStats[event.source.body.state][event.message];
                }
            }
        }

        friend std::ostream &operator <<(std::ostream &out, const GameLogger &logger) {
            out << "Hint stats:" << std::endl;
            for(int i=0; i<N; ++i) {
                for(int j=0; j<N; ++j) {
                    out << static_cast<double>(logger.hintStats[i][j])/logger.nGames << '\t';
                }
                out << std::endl;
            }
            out << "Guess stats:" << std::endl;
            for(int i=0; i<N; ++i) {
                for(int j=0; j<N; ++j) {
                    out << static_cast<double>(logger.guessStats[i][j])/logger.nGames << '\t';
                }
                out << std::endl;
            }
            out << "Win stats:" << logger.nWins*1.0/logger.nGames << std::endl;
            return out;
        }
    };

    void qLearningGuessTheNumber() {
        const int NTRAININGEPISODES = 200000; // 4000000;
        const double updateStepSize = 0.001;
        const size_t bufferSize = 128;
        const size_t batchSize = 16;
        const double discount = 1.0;
        const size_t endStateFnnUpdateInterval = 2;

        auto burnInThenTrainEveryStep = [burnin = 2]<class BODY>(const abm::events::PreActBodyState<BODY> & /* event */) mutable {
            if(burnin > 0) --burnin;
            return burnin == 0;
        };

        abm::approximators::FNN approximatorFunction(
                body_type::dimension,
                mlpack::Linear(6),
                mlpack::ReLU(),
                mlpack::Linear(6),
                mlpack::ReLU(),
                mlpack::Linear(body_type::action_type::size)
        );

        abm::lossFunctions::QLearningLoss loss(
                bufferSize,
                body_type::dimension,
                batchSize,
                discount,
                approximatorFunction,
                endStateFnnUpdateInterval);

        auto mind1 = abm::minds::QMind(
                abm::approximators::DifferentiableAdaptiveFunction(
                        std::move(approximatorFunction),
                        std::move(loss),
                        ens::AdamUpdate(),
                        updateStepSize,
                        burnInThenTrainEveryStep),
                abm::minds::GreedyPolicy(
                        abm::explorationStrategies::ExponentialDecay(1.0, NTRAININGEPISODES, 0.005)
                )
        );

        auto mind2 = mind1;

        abm::Agent agent1(body_type(), std::move(mind1));
        abm::Agent agent2(body_type(), std::move(mind2));

        // ------ train ------

        int nTrainingEpisodes = 50000;
        deselby::random::gen.seed(1235);
        std::cout << "Starting " << nTrainingEpisodes << " training episodes" << std::endl;
        auto logger = GameLogger();
        while(nTrainingEpisodes-- > 0) {
            bool agent1IsGuesser = deselby::random::uniform<bool>();
            agent1.body.reset(agent1IsGuesser);
            agent2.body.reset(!agent1IsGuesser);
            if(agent1IsGuesser) {
                abm::episodes::runAsync(agent2, agent1, logger);
            } else {
                abm::episodes::runAsync(agent1, agent2, logger);
            }
        }
        std::cout << logger;
        std::cout << "\nAgent 1's Q-value semantics:\n";
        std::cout << agent1.mind(body_type(true, false, body_type::sayA)).t();
        std::cout << agent1.mind(body_type(true, false, body_type::sayB)).t();
        std::cout << agent1.mind(body_type(true, false, body_type::sayC)).t();
        std::cout << "\nAgent 2's Q-value semantics:\n";
        std::cout << agent2.mind(body_type(true, false, body_type::sayA)).t();
        std::cout << agent2.mind(body_type(true, false, body_type::sayB)).t();
        std::cout << agent2.mind(body_type(true, false, body_type::sayC)).t();
    }
}

#endif //MULTIAGENTGOVERNMENT_EXPERIMENT6_H
