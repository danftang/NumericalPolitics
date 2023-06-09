// Large society (lots of strangers) with violence and language.
// Agents encounter each other randomly, and enter into an extended
// turns-based interaction that either ends by mutual agreement or in a fight. A randomly
// chosen player begins the interaction.
//
// Experiment 1a
// All agents are strangers, so state is simply internal state consisting of
// inventory of sugar/spice and a history of the interaction so far.
//
// ActionEnum are:
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

#ifndef MULTIAGENTGOVERNMENT_EXPERIMENT5_H
#define MULTIAGENTGOVERNMENT_EXPERIMENT5_H

#include "DeselbyStd/random.h"
#include "abm/abm.h"
#include "abm/agents/agents.h"

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

void experiment5a() {
    const int NTRAININGITERATIONS = 4000000;
    const int NPERFORMINGITERATIONS = 100;
    abm::agents::SugarSpiceTradingAgent agents[2];

    agents[0].connectTo(agents[1]);
    agents[1].connectTo(agents[0]);

    // train
    for(int iterations = 0; iterations < NTRAININGITERATIONS; ++iterations) {
        // set random initial state
        bool agent0HasSugar = deselby::Random::nextBool();
        bool agent0HasSpice = deselby::Random::nextBool();
        agents[0].reset(agent0HasSugar, agent0HasSpice, deselby::Random::nextBool());
        agents[1].reset(!agent0HasSugar, !agent0HasSpice, deselby::Random::nextBool());

        // random agent starts
        int agentToStart = deselby::Random::nextInt(0,2);
        agents[agentToStart].start().exec();
        std::cout << std::endl;
    }

    // perform
    abm::agents::SugarSpiceTradingAgent::verboseMode = true;
    agents[0].policy.pExplore = 0.0;
    agents[1].policy.pExplore = 0.0;
    for(int state = 0; state < 16; ++state) {
        // set random initial state
        int agentToStart = state & 1;
        bool otherAgentPrefersSugar = (state >> 1) & 1;
        bool startAgentPrefersSugar = (state >> 2) & 1;
        bool startAgentHasSugar = (state >> 3) & 1;
        bool startAgentHasSpice = (state >> 4) & 1;
        agents[agentToStart].reset(startAgentHasSugar, startAgentHasSpice, startAgentPrefersSugar);
        agents[agentToStart^1].reset(!startAgentHasSugar, !startAgentHasSpice, otherAgentPrefersSugar);

        // random agent starts
        std::cout << "------- Starting game -------" << std::endl << agents[agentToStart].state << agents[agentToStart^1].state;
        agents[agentToStart].start().exec();
        std::cout << agents[agentToStart].state << agents[agentToStart^1].state;
        std::cout << std::endl << std::endl;
    }


}

#endif //MULTIAGENTGOVERNMENT_EXPERIMENT5_H
