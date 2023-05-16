// An agent that classifies people as "friend" or "stranger"
// and can identify and remember the last move of a fixed number
// of agents (always the last "n" agents encountered).
//
// TODO: deal properly with forgetting
// When the agent forgets a friend, it's like saying that the previous step in-fact led
// to either re-encountering as a stranger or just the "end game" with Q-value 0.
// So, we can either do the forgetting at the end of a game, for some kind of statistical
// forgetting (although then it's hard to limit the number of remembered agents) or remember
// whole interaction histories with agents and train only at the forgetting stage.
//
// Created by daniel on 15/05/23.
//

#ifndef MULTIAGENTGOVERNMENT_SUGARSPICEAGENT1_H
#define MULTIAGENTGOVERNMENT_SUGARSPICEAGENT1_H


class SugarSpiceAgentWithFriends {
public:
    static constexpr size_t NSTATES = 5; // cooperate/cooperate, cooperate/defect, defect/cooperate, defect,defect, isStranger
    static constexpr size_t NACTIONS = 2;
    static constexpr int NFRIENDS = 3;

    typedef ulong time_type;
    typedef abm::QTablePolicy<NSTATES, NACTIONS> policy_type;
    typedef abm::Schedule<time_type> schedule_type;

    static constexpr float REWARD[2][2] = {{3, 0},
                                           {4, 1}};
    static constexpr int MEMORY_SIZE = 2; // Number of other agents this agent can remember

    std::array<std::pair<SugarSpiceAgentWithFriends *, int>, NFRIENDS> friends; // map from previously encountered agent to outcome of last game
    policy_type policy;
    abm::CommunicationChannel<abm::Schedule<time_type>, int> opponent;
    int nextFriendIndex = 0;
    int currentFriendIndex = 0;
    int myLastMove;

    // Handler for receiving a message to handle a new opponent
    // Connect to the new opponent and sends it a trading action at the given time
    schedule_type handleNewOpponent(SugarSpiceAgentWithFriends &newOpponent, time_type time) {
        opponent.connectTo(newOpponent, &SugarSpiceAgentWithFriends::handleOpponentMove, 1);
        currentFriendIndex = 0;
        while(currentFriendIndex < friends.size() && friends[currentFriendIndex].first != &newOpponent) {
            ++currentFriendIndex;
        }
        if(currentFriendIndex == friends.size()) {
            // opponent is a stranger
            currentFriendIndex = nextFriendIndex;
            nextFriendIndex = (nextFriendIndex + 1)%NFRIENDS;
            friends[currentFriendIndex] = std::pair(&newOpponent, 4);
        }
        myLastMove = policy.getAction(friends[currentFriendIndex].second);
        return opponent.send(myLastMove, time);
    }


    // handler for receiving
    schedule_type handleOpponentMove(int opponentsMove, time_type time) {
//            std::cout << "Handling " << opponentsMove << ", " << myLastMove << std::endl;
        int newState = 2 * myLastMove + opponentsMove;
        float reward = REWARD[myLastMove][opponentsMove];
        policy.train(friends[currentFriendIndex].second, myLastMove, reward, newState);
        friends[currentFriendIndex].second = newState;
        return abm::Schedule<time_type>();
    }
};


#endif //MULTIAGENTGOVERNMENT_SUGARSPICEAGENT1_H
