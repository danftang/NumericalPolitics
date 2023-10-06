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
#include "abm/minds/qLearning/QTable.h"
#include "abm/minds/IncompleteInformationMCTS.h"
#include "abm/episodes/SimpleEpisode.h"
#include "abm/bodies/SugarSpiceTradingBody.h"
#include "abm/DQN.h"
#include "abm/RandomReplay.h"
#include "abm/minds/qLearning/GreedyPolicy.h"

namespace experiment5 {
    const bool HASLANGUAGE = true;

    typedef abm::bodies::SugarSpiceTradingBody<HASLANGUAGE> body_type;


    /** Sets an agent pair to a given joint body state by index.
     * The total amount of sugar and spice is 1 sugar and 1 spice.
     * @param stateID binary number in 0..1111 that identifies the state
     */
    template<class MIND1, class MIND2>
    void
    setStartState(abm::Agent<body_type,MIND1> &firstAgent, abm::Agent<body_type,MIND2> &secondAgent, std::bitset<4> startState) {
        firstAgent.body.sugar() = startState[0];
        secondAgent.body.sugar() = 1.0 - firstAgent.body.sugar();
        firstAgent.body.spice() = startState[1];
        secondAgent.body.spice() = 1.0 - firstAgent.body.spice();
        firstAgent.body.prefersSugar() = startState[2];
        secondAgent.body.prefersSugar() = startState[3];
    }


    /** trains agents by running nTrainingEpisodes with random start state and random first-mover */
    template<class AGENT1, class AGENT2>
    void train(AGENT1 &agent1, AGENT2 &agent2, size_t nTrainingEpisodes) {
        while(nTrainingEpisodes-- > 0) {
            setStartState(agent1, agent2, deselby::Random::nextInt(32));
            if(deselby::Random::nextBool()) {
                abm::episodes::runAsync(agent1, agent2);
            } else {
                abm::episodes::runAsync(agent2, agent1);
            }
        }
    }


    /** Iterates through all 64 games between two agents (32 joint start states times two possible first movers) */
    template<class AGENT1, class AGENT2>
    void showBehaviour(AGENT1 &agent1, AGENT2 &agent2) {
        for(int startState = 0; startState < 32; ++startState) {
            setStartState(agent1, agent2, startState);
            abm::episodes::runAsync(agent1, agent2, abm::episodes::callbacks::Verbose());
            abm::episodes::runAsync(agent2, agent1, abm::episodes::callbacks::Verbose());
        }
    }


    /** Test a QTable on binary, repeated SugarSpiceTrading */
    void runA() {
        const int NTRAININGEPISODES = 200000; // 4000000;

        auto mind = abm::minds::MeanRewardMindWrapper(
                0.99,
                abm::minds::QMind(
                        abm::QTable<body_type::nstates, body_type::action_type::size>(1.0, 0.9999),
                        abm::GreedyPolicy<body_type::action_type>(
                                abm::explorationStrategies::ExponentialDecay(1.0, NTRAININGEPISODES, 0.005)
                        )
                )
        );

        auto agent1 = abm::Agent(body_type{}, mind);
        auto agent2 = abm::Agent(body_type{}, mind);
        train(agent1,agent2,NTRAININGEPISODES);
        body_type::pBanditAttack = 0.002;
        agent1.mind.policy.explorationStrategy = abm::explorationStrategies::NoExploration();
        agent2.mind.policy.explorationStrategy = abm::explorationStrategies::NoExploration();
        showBehaviour(agent1,agent2);
    }

    /** Test a Deep Q-network on binary, repeated SugarSpiceTrading
    * (spoiler: doesn't learn to cooperate)
    */
    void runB() {
        const int NTRAININGEPISODES = 200000; // 4000000;

        auto mind = abm::minds::MeanRewardMindWrapper(
                0.99,
                abm::minds::QMind(
                        abm::DQN<body_type::dimension, body_type::action_type::size>(
                                mlpack::SimpleDQN<mlpack::MeanSquaredError, mlpack::HeInitialization>(
                                        50, 25, body_type::action_type::size),
                                abm::RandomReplay(16, 128, body_type::dimension),
                                2,
                                1.0),
                        abm::GreedyPolicy<body_type::action_type>(
                                abm::explorationStrategies::ExponentialDecay(1.0, NTRAININGEPISODES, 0.005)
                        )
                )
        );

        std::vector agents = {abm::Agent(body_type(), mind), abm::Agent(body_type(), mind)};
        train(agents,NTRAININGEPISODES);
        body_type::pBanditAttack = 0.002;
        agents[0].mind.policy.explorationStrategy = abm::explorationStrategies::NoExploration();
        agents[1].mind.policy.explorationStrategy = abm::explorationStrategies::NoExploration();
        showBehaviour(agents);
    }


    /** Test a Incomplete Information Monte-Carlo tree search on binary, repeated SugarSpiceTrading
    */
    void runC() {
        std::cout << "starting experiment" << std::endl;
        typedef approximators::FeedForwardNeuralNet<body_type::dimension, body_type::action_type::size, mlpack::SimpleDQN<mlpack::MeanSquaredError, mlpack::HeInitialization>> qFunction;

        auto network = mlpack::FFN(mlpack::MeanSquaredError(), mlpack::HeInitialization());
        network.Add(new mlpack::Linear(100));
        network.Add(new mlpack::ReLU());
        network.Add(new mlpack::Linear(50));
        network.Add(new mlpack::ReLU());
        network.Add(new mlpack::Linear(body_type::action_type::size));

        ens::Adam adam;
        adam.MaxIterations()

        auto mind =  abm::minds::IncompleteInformationMCTS<body_type,decltype(network)>(200000, 1.0, network);

        std::vector agents = {abm::Agent(body_type(), mind), abm::Agent(body_type(), mind)};

        showBehaviour(agents);

//        static_assert(abm::HasOneParamInitCallback<abm::minds::IncompleteInformationMCTS<body_type>>);
//        static_assert(abm::HasOneParamInitCallback<body_type &>);


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
