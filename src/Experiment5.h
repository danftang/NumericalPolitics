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
    template<class AGENT1, class AGENT2, class... CALLBACKS>
    void train(AGENT1 &agent1, AGENT2 &agent2, size_t nTrainingEpisodes, CALLBACKS... callbacks) {
        std::cout << "Starting " << nTrainingEpisodes << " training episodes" << std::endl;
        while(nTrainingEpisodes-- > 0) {
            setStartState(agent1, agent2, deselby::random::uniform(16));
            if(deselby::random::uniform<bool>()) {
                abm::episodes::runAsync(agent1, agent2, callbacks...);
            } else {
                abm::episodes::runAsync(agent2, agent1, callbacks...);
            }
        }
    }


    /** Iterates through all 64 games between two agents (32 joint start states times two possible first movers) */
    template<class AGENT1, class AGENT2>
    void showBehaviour(AGENT1 &agent1, AGENT2 &agent2) {
        std::cout << "Showing behaviour..." << std::endl;
        auto callback = abm::callbacks::Verbose();
        for(int startState = 0; startState < 16; ++startState) {
            setStartState(agent1, agent2, startState);
            abm::episodes::runAsync(agent1, agent2, callback);
            setStartState(agent1, agent2, startState);
            abm::episodes::runAsync(agent2, agent1, callback);
        }
    }


    template<class MIND1, class MIND2>
    void trainAndShow(MIND1 &&mind1, MIND2 &&mind2, const int NTRAININGEPISODES) {
        abm::Agent agent1(body_type(), std::move(mind1));
        abm::Agent agent2(body_type(), std::move(mind2));
        train(agent1, agent2, NTRAININGEPISODES, abm::callbacks::Verbose());
        body_type::pBanditAttack = 0.005;
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


    /** Test an Incomplete Information Monte-Carlo tree search on binary, repeated SugarSpiceTrading
    */
    void iimctsSugarSpice() {
        const double discount = 1.0;
        const size_t nSamplesInATree = 50000;//200000;
        const size_t nTrainingEpisodes = 100;

        auto offTreeApproximator = abm::approximators::FNN(
                mlpack::HeInitialization(),
                body_type::dimension,
                mlpack::Linear(50),
                mlpack::ReLU(),
                mlpack::Linear(25),
                mlpack::ReLU(),
                mlpack::Linear(body_type::action_type::size)
        );

//        auto offTreeApproximator = [](const body_type &body) {
//            arma::mat qVec(body_type::action_type::size,1);
//            qVec.randu();
//            return qVec;
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

        trainAndShow(std::move(mind1), std::move(mind2), nTrainingEpisodes);
    }

}

#endif //MULTIAGENTGOVERNMENT_EXPERIMENT5_H
