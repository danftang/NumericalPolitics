//
// Created by daniel on 15/05/23.
//

#ifndef MULTIAGENTGOVERNMENT_SUGARSPICEAGENT1_H
#define MULTIAGENTGOVERNMENT_SUGARSPICEAGENT1_H


class SugarSpiceAgent1 {
public:
    static constexpr size_t NSTATES = 5; // cooperate/cooperate, cooperate/defect, defect/cooperate, defect,defect, isStranger
    static constexpr size_t NACTIONS = 2;

    typedef ulong time_type;
    typedef abm::QTablePolicy<NSTATES, NACTIONS> policy_type;
    typedef abm::Schedule<time_type> schedule_type;

    static constexpr float REWARD[2][2] = {{3, 0},
                                           {4, 1}};
    static constexpr int MEMORY_SIZE = 3; // Number of other agents this agent can remember
    static constexpr float QMIN = 0;
    static constexpr float QMAX = REWARD[1][0] /
                                  (1.0 -
                                   policy_type::DEFAULT_DISCOUNT); // Value of Q if all future rewards are max reward

//        struct MemoryItem {
//            int otherLastPlay;
//            int myLastPlay;
//        };

    std::map<SugarSpiceAgent1 *, int> mem; // map from previously encountered agent to state of last game
    policy_type policy;
    abm::CommunicationChannel<abm::Schedule<time_type>, int> opponent;
    std::map<SugarSpiceAgent1 *, int>::iterator currentOpponentIt;
    int currentState;
    int myLastMove;
    float totalReward = 0;

    void setPolicy(long policyID) { policy.setPolicy(policyID, QMIN, QMAX); }

//            // Connect this agent's communication channel to the opponent's handler.
//            void connectTo(SugarSpiceAgent1 &opponentAgent) {
//            opponent.connectTo(opponentAgent, &SugarSpiceAgent1::handleOpponentMove, 1);
//            }

    // Handler for receiving a message to handle a new opponent
    // Connect to the new opponent and sends it a trading action at the given time
    schedule_type handleNewOpponent(SugarSpiceAgent1 &newOpponent, time_type time) {
        opponent.connectTo(newOpponent, &SugarSpiceAgent1::handleOpponentMove, 1);
        currentOpponentIt = mem.find(&newOpponent);
        if (currentOpponentIt == mem.end()) { // unknown new opponent
            currentState = 4; // is a stranger
            if (mem.size() < MEMORY_SIZE) currentOpponentIt = mem.insert(std::pair(&newOpponent, 4)).first;
        } else {
            currentState = currentOpponentIt->second;
        }
        myLastMove = policy.getAction(currentState);
        return opponent.send(myLastMove, time);
    }


    // handler for receiving
    schedule_type handleOpponentMove(int opponentsMove, time_type time) {
//            std::cout << "Handling " << opponentsMove << ", " << myLastMove << std::endl;
        int newState = 2 * myLastMove + opponentsMove;
        float reward = REWARD[myLastMove][opponentsMove];
        policy.train(currentState, myLastMove, reward, newState);
        totalReward += reward;
        if (currentOpponentIt != mem.end()) currentOpponentIt->second = newState;
        return abm::Schedule<time_type>();
    }
};


#endif //MULTIAGENTGOVERNMENT_SUGARSPICEAGENT1_H
