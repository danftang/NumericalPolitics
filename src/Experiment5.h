// Large society (lots of strangers) with violence and language.
// Agents encounter each other randomly, and enter into an extended
// turns-based interaction that either ends by mutual agreement or in a fight. A randomly
// chosen player begins the interaction.
//
// Experiment 1a
// All agents are strangers, so state is simply internal state consisting of
// inventory of sugar/spice and a history of the interaction so far.
//
// MessageEnum are:
//  - say 0
//  - say 1
//  - give 1 unit of sugar to the other player
//  - give 1 unit of spice to the other player
//  - start a fight
//  - end interaction (say bye)
//
//  If either agent starts a fight, a uniformly chosen agent gets all the
//  inventory of the other agent and the interaction ends. However, both agents get some -ve reward for injury.
//
//  Reward can either occur at each change of inventory, or just at the end of the interaction. Communication has
//  no intrinsic reward. Giving something you don't have gets an immediate -ve reward [or perhaps just drops out
//  of the available actions]
//
// [Should an agent see the other agent's inventory? - no need if we always start with 1 sugar and 1 spice as
// we can infer other's inventory from one's own]
//
// How do things change if we change the take actions to give actions?... or allow both? [or add offer/accept/take by force]
// What if we make fighting back optional? With definite loss of inventory but no injury
//
// The size of the dialogue memory could be reduced by having distinct negotiate/swap/fight-or-flight phases?
// We could also restrict ourselves to initial inventory of one agent has 1 sugar and one has 1 spice
// and each agent prefers the other's inventory by a factor of 3. Do they manage to swap?
//
// Without fighting or talking, with just taking, it makes sense for the first player to take, then the second player
// to take his preferred (i.e. a swap), but then the first player to take back his less preferred, etc...
//
// With giving and no fighting or talking, it makes sense for all agents to just immediately say goodbye (if player 1
// gives, player 2 just needs to say goodbye and player 1 has no other option but to say goodbye).
//
// With giving and fighting, player 1 can start a fight if player 2 doesn't reciprocate, but it makes more sense for
// player 2 to reciprocate. So, language never becomes necessary.
//
// Language could be useful when the other's inventory is uncertain, i.e. to say "no need to fight me, I have nothing"
// but what motivation is there to be honest? There needs to be information that certainly improves both agent's reward.
// A piece of information must not be a lie if, given that I believe it and act accordingly, the best outcome for the
// speaker comes if the information is true.
// Maybe more to the point is "if you lie to me, I'll beat you up".
// How about "my preference is for sugar". Would an agent lie about this? Esp if it had spice. [although it could be a
// signal that the speaker is willing to fight]
//
// So, we should have it that there are interactions with all preference pairs, and agents don't know the preference
// of the other agent.
//
// Parameterised Agents
// --------------------
//
// Created by daniel on 30/05/23.

//
// Two agents that repeatedly have a sustained, turns-based interaction.
// At the start of each interaction, resources of 1 unit of spice and 1 unit of sugar are randomly distributed
// between them, and each agent is randomly assigned a preference of sugar or spice. At each turn an agent can:
//
//  - Give a unit of sugar to the other player
//  - Give a unit of spice to the other player
//  - Declare a wish to end the game
//  - Start a fight (winner takes all, winner chosen at random)
//  - say 0
//  - say 1
//
//  An agent state consists of the inventory of self and other, one's own preference (but not that of the other) and
//  the game history.

#ifndef MULTIAGENTGOVERNMENT_EXPERIMENT5_H
#define MULTIAGENTGOVERNMENT_EXPERIMENT5_H

#include "DeselbyStd/random.h"
#include "abm/societies/RandomEncounterSociety.h"
#include "abm/Agent.h"
#include "abm/minds/QMind.h"
#include "abm/minds/IncompleteInformationMCTS.h"
#include "abm/episodes/SimpleEpisode.h"
#include "abm/bodies/SugarSpiceTradingBody.h"
#include "abm/DQN.h"
#include "abm/RandomReplay.h"
#include "abm/MeanRewardMindWrapper.h"

namespace experiment5 {
    const bool HASLANGUAGE = true;

    typedef abm::bodies::SugarSpiceTradingBody<HASLANGUAGE> body_type;


    std::function<body_type()> bodyHiddenStateSampler(bool hasSugar, bool hasSpice) {
        return [hasSugar, hasSpice]() {
            return body_type(hasSugar, hasSpice, deselby::Random::nextBool());
        };
    }


    auto makeEpisode(bool firstMoverHasSugar, bool firstMoverHasSpice) {
        return abm::episodes::SimpleEpisode(bodyHiddenStateSampler(firstMoverHasSugar, firstMoverHasSpice),
                                            bodyHiddenStateSampler(!firstMoverHasSugar, !firstMoverHasSpice));
    }


    template<class AGENT>
    void train(std::vector<AGENT> &agents, size_t nTrainingEpisodes) {
        bool verbose = false;
        for(int iterations = 0; iterations < nTrainingEpisodes; ++iterations) {
            if(iterations%100 == 0) {
                std::cout << iterations << " "
                          << agents[0].mind.meanReward << " "
                          << agents[1].mind.meanReward << " "
                          << agents[0].mind.meanReward + agents[1].mind.meanReward
                          << std::endl;
            }
            // set random initial state
            bool firstMoverHasSugar = deselby::Random::nextBool();
            bool firstMoverHasSpice = deselby::Random::nextBool();
            auto episode = makeEpisode(firstMoverHasSugar, firstMoverHasSpice);
            int firstMoverIndex = deselby::Random::nextBool();
            episode.run(agents[firstMoverIndex], agents[firstMoverIndex^1], verbose);
        }
    }

    template<class AGENT>
    void showBehaviour(std::vector<AGENT> &agents) {
        bool verbose = true;
        for(int state = 0; state < 32; ++state) {
            // set random initial state
            int firstMoverIndex = state & 1;
            bool firstMoverHasSugar = state & 2;
            bool firstMoverHasSpice = state & 4;
            bool firstMoverPrefersSugar = state & 8;
            bool secondMoverPrefersSugar = state & 16;
            auto episode = makeEpisode(firstMoverHasSugar, firstMoverHasSpice);

            episode.run(agents[firstMoverIndex], agents[firstMoverIndex^1],
                        body_type(firstMoverHasSugar, firstMoverHasSpice, firstMoverPrefersSugar),
                        body_type(!firstMoverHasSugar, !firstMoverHasSpice, secondMoverPrefersSugar),
                        verbose);
        }
    }


    /** Test a QTable on binary, repeated SugarSpiceTrading
     *
     */
    void runA() {
        const int NTRAININGEPISODES = 200000; // 4000000;

        auto mind = abm::MeanRewardMindWrapper(
                0.99,
                abm::minds::QMind(
                        abm::QTable<body_type::nstates, body_type::action_type::size>(1.0, 0.9999),
                        abm::GreedyPolicy<body_type::action_type>(
                                abm::ExponentialDecay(1.0, NTRAININGEPISODES, 0.005)
                        )
                )
        );

        std::vector agents = {abm::Agent(body_type(), mind), abm::Agent(body_type(), mind)};
        train(agents,NTRAININGEPISODES);
        body_type::pBanditAttack = 0.002;
        agents[0].mind.policy.explorationStrategy = abm::NoExploration();
        agents[1].mind.policy.explorationStrategy = abm::NoExploration();
        showBehaviour(agents);
    }

    /** Test a Deep Q-network on binary, repeated SugarSpiceTrading
 * (spoiler: doesn't learn to cooperate)
 */
    void runB() {
        const int NTRAININGEPISODES = 200000; // 4000000;

        auto mind = abm::MeanRewardMindWrapper(
                0.99,
                abm::minds::QMind(
                        abm::DQN<body_type::dimension, body_type::action_type::size>(
                                mlpack::SimpleDQN<mlpack::MeanSquaredError, mlpack::HeInitialization>(
                                        50, 25, body_type::action_type::size),
                                abm::RandomReplay(16, 128, body_type::dimension),
                                2,
                                1.0),
                        abm::GreedyPolicy<body_type::action_type>(
                                abm::ExponentialDecay(1.0, NTRAININGEPISODES, 0.005)
                        )
                )
        );

        std::vector agents = {abm::Agent(body_type(), mind), abm::Agent(body_type(), mind)};
        train(agents,NTRAININGEPISODES);
        body_type::pBanditAttack = 0.002;
        agents[0].mind.policy.explorationStrategy = abm::NoExploration();
        agents[1].mind.policy.explorationStrategy = abm::NoExploration();
        showBehaviour(agents);
    }

    /** Test a Deep Q-network on binary, repeated SugarSpiceTrading
    * (spoiler: doesn't learn to cooperate)
    */
    void runC() {
        std::cout << "starting experiment" << std::endl;
        auto mind = abm::MeanRewardMindWrapper(
                0.99,
//                abm::minds::IncompleteInformationMCTS<body_type>(9000, 1.0)
        abm::minds::IncompleteInformationMCTS<body_type>(500000, 1.0)
        );
        std::vector agents = {abm::Agent(body_type(), mind), abm::Agent(body_type(), mind)};

        showBehaviour(agents);

//        bool hasSugar = true;
//        bool hasSpice = false;
//        bool prefersSugar0 = true;
//        bool prefersSugar1 = false;
//        deselby::Random::gen.seed(4568);
//        auto episode = makeEpisode(hasSugar, hasSpice);
//        episode.run(agents[0], agents[1],
//                    body_type(hasSugar, hasSpice, prefersSugar0),
//                    body_type(!hasSugar, !hasSpice, prefersSugar1),
//                    true);


    }

}










// Can 4 agents with no way to distinguish between each other generate a shared langauge?
//void experiment5b() {
//    const int NTRAININGITERATIONS = 8000000;
//    const int NPERFORMINGITERATIONS = 100;
//    const bool HASLANGUAGE = true;
//    abm::agents::SugarSpiceTradingAgent<HASLANGUAGE> agents[4];
//
//    // train
//    for(int iterations = 0; iterations < NTRAININGITERATIONS; ++iterations) {
//        if(iterations%10000 == 0) std::cout << iterations << std::endl;
//        // Choose agents to play
//        int agent0Index = deselby::Random::nextInt(0, 4);
//        int agent1Index = deselby::Random::nextInt(0, 3);
//        if (agent1Index >= agent0Index) agent1Index += 1;
//        agents[agent0Index].connectTo(agents[agent1Index]);
//        agents[agent1Index].connectTo(agents[agent0Index]);
//
//        // set initial state
//        bool agent0HasSugar = deselby::Random::nextBool();
//        bool agent0HasSpice = deselby::Random::nextBool();
//        agents[agent0Index].reset(agent0HasSugar, agent0HasSpice, deselby::Random::nextBool());
//        agents[agent1Index].reset(!agent0HasSugar, !agent0HasSpice, deselby::Random::nextBool());
//
//        agents[agent0Index].start().exec();
//    }
//
//    // perform
//    abm::agents::SugarSpiceTradingAgent<HASLANGUAGE>::verboseMode = true;
//    abm::agents::SugarSpiceTradingAgent<HASLANGUAGE>::pBanditAttack = 0.0;
//    for(int agent = 0; agent<4; ++agent) {
//        agents[agent].policy.setExploration(0.0);
//    }
//    for(int state = 0; state < 32; ++state) {
//        bool agent0PrefersSugar = (state >> 1) & 1;
//        bool agent1PrefersSugar = (state >> 2) & 1;
//        bool agent0HasSugar = (state >> 3) & 1;
//        bool agent0HasSpice = (state >> 4) & 1;
//        for(int agent0Index = 0; agent0Index < 4; agent0Index++) {
//            for(int r=0; r<3; ++r) {
//                int agent1Index = r;
//                if (agent1Index >= agent0Index) agent1Index += 1;
//                agents[agent0Index].connectTo(agents[agent1Index]);
//                agents[agent1Index].connectTo(agents[agent0Index]);
//
//                agents[agent0Index].reset(agent0HasSugar, agent0HasSpice, agent0PrefersSugar);
//                agents[agent1Index].reset(!agent0HasSugar, !agent0HasSpice, agent1PrefersSugar);
//
//                std::cout << "------- Starting game -------" << std::endl << agents[agent0Index].state << agents[agent1Index].state;
//                agents[agent0Index].start().exec();
//                std::cout << agents[agent0Index].state << agents[agent1Index].state;
//                std::cout << std::endl << std::endl;
//
//            }
//        }
//
//
//    }
//}

#endif //MULTIAGENTGOVERNMENT_EXPERIMENT5_H
