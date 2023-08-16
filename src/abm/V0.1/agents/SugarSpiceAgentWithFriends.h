// An agent that classifies people as "friend" or "stranger"
// and can identify and remember the last move of a fixed number
// of agents (always the last "n" agents encountered).
//
// TODO: deal properly with forgetting
// When the agent forgets a friend, it's like saying that the previous step in-fact led
// to either re-encountering as a stranger or just the "end game" with Q-value 0.
// So, we can either do the forgetting at the end of a game, for some kind of statistical
// forgetting (although then it's hard to limit the number of remembered agents) or remember
// whole interaction histories with agents and train only at the forgetting stage. [Does it
// make a difference whether a forgotten agent is classed as a stranger or as a zero-quality
// end-game? Don't think so, it just multiplies everything by 1/(1-decay) and prob
// makes eveything less stable]
//
// Created by daniel on 15/05/23.
//

#ifndef MULTIAGENTGOVERNMENT_SUGARSPICEAGENT1_H
#define MULTIAGENTGOVERNMENT_SUGARSPICEAGENT1_H


#include <cstdlib>
#include <deque>
#include "../V0.1/QTablePolicy.h"
#include "../Schedule.h"
#include "../CommunicationChannel.h"
#include "../../DeselbyStd/stlstream.h"

namespace abm {
    namespace agents {

        class SugarSpiceAgentWithFriends {
        public:
            static constexpr size_t NSTATES = 5; // cooperate/cooperate, cooperate/defect, defect/cooperate, defect,defect, isStranger
            static constexpr size_t NACTIONS = 2;
            static constexpr int NFRIENDS = 3;
            static constexpr int STRANGER_STATE = 4;

            typedef ulong time_type;
            typedef QTablePolicy<NSTATES, NACTIONS> policy_type;
            typedef Schedule<time_type> schedule_type;

            static constexpr double REWARD[4] = {3, 0,
                                                4, 1}; // indexed by 2*myLastMove + opponentLastMove

            class InteractionHistory {
            public:
                SugarSpiceAgentWithFriends *agent;
                int lastStartState;
                int lastInteraction;
            };

            std::array<InteractionHistory, NFRIENDS> friends; // map from previously encountered agent to outcome of last game
            policy_type policy;
            CommunicationChannel<Schedule<time_type>, int> opponent;
            int nextFriendIndexToForget = 0;
            int currentFriendIndex = 0;
            int myLastMove;

            // Handler for receiving a message to handle a new opponent
            // Connect to the new opponent and sends it a trading action at the given time
            schedule_type handleNewOpponent(SugarSpiceAgentWithFriends &newOpponent, time_type time) {
                opponent.connectTo(newOpponent, &SugarSpiceAgentWithFriends::handleOpponentMove, 1);
                currentFriendIndex = 0;
                while (currentFriendIndex < friends.size() && friends[currentFriendIndex].agent != &newOpponent) {
                    ++currentFriendIndex;
                }
                if (currentFriendIndex == friends.size()) {
                    // opponent is a stranger, so eldest friend in history becomes a stanger and ends game, and new acquaintance
                    // is in STRANGER_STATE
                    if(friends[nextFriendIndexToForget].agent != nullptr) {
                        const int startState = friends[nextFriendIndexToForget].lastStartState;
                        const int lastInteraction = friends[nextFriendIndexToForget].lastInteraction;
                        const int myMove = lastInteraction>>1;
                        policy.train(startState, myMove, REWARD[lastInteraction], STRANGER_STATE, true);
                    }
                    currentFriendIndex = nextFriendIndexToForget;
                    nextFriendIndexToForget = (nextFriendIndexToForget + 1) % NFRIENDS;
                    friends[currentFriendIndex].agent = &newOpponent;
                    friends[currentFriendIndex].lastStartState = STRANGER_STATE;
                    myLastMove = policy.getAction(STRANGER_STATE);
                } else {
                    // seen this opponent before, so we're in the state of the last interaction
                    const int startState = friends[currentFriendIndex].lastStartState;
                    const int endState = friends[currentFriendIndex].lastInteraction;
                    const int myMove = endState>>1;
                    policy.train(startState, myMove, REWARD[endState], endState, false);
                    friends[currentFriendIndex].lastStartState = endState;
                    myLastMove = policy.getAction(endState);
                }
                return opponent.send(myLastMove, time);
            }


            // handler for receiving move. Remember new state for the current friend.
            // No future actions.
            schedule_type handleOpponentMove(int opponentsMove, time_type time) {
//            std::cout << "Handling " << opponentsMove << ", " << myLastMove << std::endl;
                friends[currentFriendIndex].lastInteraction = 2 * myLastMove + opponentsMove;
                return Schedule<time_type>();
            }

        protected:
//            void trainOnInteractionHistory(InteractionHistory &forgottenAgentHistory) {
//                std::cout << "Training on " << forgottenAgentHistory.agent << " " << forgottenAgentHistory.interactionHistory << std::endl;
//                assert(forgottenAgentHistory.interactionHistory.size() >= 1);
//                int stateBeforeLastAction = STRANGER_STATE;
//                while (forgottenAgentHistory.interactionHistory.size() > 1) {
//                    const int newState = forgottenAgentHistory.interactionHistory.front();
//                    const int myMove = newState >> 1;
//                    policy.train(stateBeforeLastAction, myMove, REWARD[newState], newState);
//                    stateBeforeLastAction = newState;
//                    forgottenAgentHistory.interactionHistory.pop_front();
//                }
//                const int lastInteraction = forgottenAgentHistory.interactionHistory.front();
//                const int myMove = lastInteraction >> 1;
//                policy.train(stateBeforeLastAction, myMove, REWARD[lastInteraction], policy_type::ENDGAME_STATE);
//                forgottenAgentHistory.interactionHistory.clear();
//            }

        };

    }
}

#endif //MULTIAGENTGOVERNMENT_SUGARSPICEAGENT1_H
