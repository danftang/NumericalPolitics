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
// #include "abm/DQN.h"
// #include "abm/RandomQStepReplay.h"
#include "abm/minds/qLearning/GreedyPolicy.h"
#include "abm/approximators/FNN.h"
#include "abm/approximators/AdaptiveFunction.h"
#include "abm/lossFunctions/QLearningLoss.h"


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
        firstAgent.body.reset(startState[0], startState[1], startState[2]);
        secondAgent.body.reset(!firstAgent.body.hasSugar(), !firstAgent.body.hasSpice(), startState[3]);
    }


    /** trains agents by running nTrainingEpisodes with random start state and random first-mover */
    template<class AGENT1, class AGENT2>
    void train(AGENT1 &agent1, AGENT2 &agent2, size_t nTrainingEpisodes) {
        std::cout << "Starting " << nTrainingEpisodes << " training episodes" << std::endl;
        while(nTrainingEpisodes-- > 0) {
            setStartState(agent1, agent2, deselby::random::uniform(32));
            if(deselby::random::uniform<bool>()) {
                abm::episodes::runAsync(agent1, agent2);
            } else {
                abm::episodes::runAsync(agent2, agent1);
            }
        }
    }


    /** Iterates through all 64 games between two agents (32 joint start states times two possible first movers) */
    template<class AGENT1, class AGENT2>
    void showBehaviour(AGENT1 &agent1, AGENT2 &agent2) {
        abm::callbacks::MeanRewardPerEpisode agent1MeanReward;
        abm::callbacks::MeanRewardPerEpisode agent2MeanReward;

        for(int startState = 0; startState < 32; ++startState) {
            setStartState(agent1, agent2, startState);
            abm::episodes::runAsync(agent1, agent2, abm::callbacks::Verbose(), agent1MeanReward);
            setStartState(agent1, agent2, startState);
            abm::episodes::runAsync(agent2, agent1, abm::callbacks::Verbose(), agent2MeanReward);
        }
        std::cout << "Agent1 mean reward per episode = " << agent1MeanReward.mean() << std::endl;
        std::cout << "Agent2 mean reward per episode = " << agent2MeanReward.mean() << std::endl;
    }


    template<class MIND1, class MIND2>
    void trainAndShow(MIND1 &&mind1, MIND2 &&mind2, const int NTRAININGEPISODES) {
        abm::Agent agent1(body_type(), std::move(mind1));
        abm::Agent agent2(body_type(), std::move(mind2));
        train(agent1, agent2, NTRAININGEPISODES);
        body_type::pBanditAttack = 0.002;
        agent1.mind.policy.explorationStrategy = abm::explorationStrategies::NoExploration();
        agent2.mind.policy.explorationStrategy = abm::explorationStrategies::NoExploration();
        showBehaviour(agent1, agent2);
    }


    template<class APPROXIMATOR>
    void approximatorSugarSpice(APPROXIMATOR &&approximatorFunction) {
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

        abm::approximators::DifferentialTrainingPolicy trainingPolicy(
                ens::AdamUpdate(),
                updateStepSize,
                approximatorFunction.parameters().n_rows,
                approximatorFunction.parameters().n_cols,
                burnInThenTrainEveryStep);

        abm::lossFunctions::QLearningLoss loss(
                bufferSize,
                body_type::dimension,
                batchSize,
                discount,
                approximatorFunction,
                endStateFnnUpdateInterval);

        auto mind1 = abm::minds::QMind(
                abm::approximators::AdaptiveFunction(
                        std::move(approximatorFunction),
                        std::move(trainingPolicy),
                        std::move(loss)),
                abm::minds::GreedyPolicy(
                        abm::explorationStrategies::ExponentialDecay(1.0, NTRAININGEPISODES, 0.005)
                )
        );
        auto mind2 = mind1;
        trainAndShow(std::move(mind1), std::move(mind2), NTRAININGEPISODES);
    }


    /** Test a QTable on binary, repeated SugarSpiceTrading */
    void qTableSugarSpice() {
        const int NTRAININGEPISODES = 200000; // 4000000;

        auto mind1 = abm::minds::QMind(
                        abm::minds::QTable<body_type::nstates, body_type::action_type::size>(1.0),
                        abm::minds::GreedyPolicy(
                                abm::explorationStrategies::ExponentialDecay(1.0, NTRAININGEPISODES, 0.005)
                        )
                    );
        auto mind2 = mind1;
        trainAndShow(std::move(mind1), std::move(mind2), NTRAININGEPISODES);
    }



    /** Test a Deep Q-network on binary, repeated SugarSpiceTrading
    * (spoiler: doesn't learn to cooperate)
    */
    void dqnSugarSpice() {
        approximatorSugarSpice(abm::approximators::FNN(
                mlpack::GaussianInitialization(),
                body_type::dimension,
                mlpack::Linear(50),
                mlpack::ReLU(),
                mlpack::Linear(25),
                mlpack::ReLU(),
                mlpack::Linear(body_type::action_type::size)
        ));

    }


    /** Test a Incomplete Information Monte-Carlo tree search on binary, repeated SugarSpiceTrading
    */
    void iimctsSugarSpice() {
        const double discount = 1.0;
        const size_t nSamplesInATree = 1000000;
        const size_t nTrainingEpisodes = 1;

//        auto offTreeApproximator = abm::approximators::FNN(
//                mlpack::HeInitialization(),
//                body_type::dimension,
//                mlpack::Linear(50),
//                mlpack::ReLU(),
//                mlpack::Linear(25),
//                mlpack::ReLU(),
//                mlpack::Linear(body_type::action_type::size)
//        );

        auto offTreeApproximator = [](const body_type &body) {
            arma::mat qVec(body_type::action_type::size,1);
            qVec.randu();
//            std::cout << "Offtree " << qVec.t() << std::endl;
            return qVec;
        };


//        auto selfStateSampler = [](const body_type &myTrueState) {
//            return body_type(myTrueState.hasSugar(), myTrueState.hasSpice(), deselby::random::uniform<bool>());
//        };

        std::function<body_type(const body_type &)> bodyStateSampler = [](const body_type &myTrueState) {
            return body_type(!myTrueState.hasSugar(), !myTrueState.hasSpice(), deselby::random::uniform<bool>());
        };

        auto mind1 = abm::minds::QMind(
                abm::minds::IncompleteInformationMCTS(
                        offTreeApproximator,
                        bodyStateSampler,
                        bodyStateSampler,
                        discount,
                        nSamplesInATree
                        ),
                abm::minds::GreedyPolicy(abm::explorationStrategies::NoExploration()));

        auto mind2 = mind1;

        abm::Agent agent1(body_type(), std::move(mind1));
        abm::Agent agent2(body_type(), std::move(mind2));

//        agent1.body.reset(false, true, true);
//        agent1.mind.on(abm::events::AgentStartEpisode(agent1.body, true));
//        std::cout << "QVec = " << agent1.mind(agent1.body) << std::endl;
//        std::cout << "act = " << agent1.mind.act(agent1.body) << std::endl;
//        std::cout << "QVec = " << agent1.mind(agent1.body) << std::endl;
        //        train(agent1, agent2, 1);

        showBehaviour(agent1,agent2);

        //        trainAndShow(std::move(mind1), std::move(mind2), nTrainingEpisodes);
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
