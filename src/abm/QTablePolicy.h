//
// Created by daniel on 13/03/23.
//

#ifndef MULTIAGENTGOVERNMENT_QTABLEPOLICY_H
#define MULTIAGENTGOVERNMENT_QTABLEPOLICY_H


template<int NSTATES, int NACTIONS>
class QTablePolicy {
public:
    static constexpr float DEFAULT_DISCOUNT = 0.99;
//        static constexpr float DEFAULT_EXPLORATION = 0.01;
    static constexpr float DEFAULT_INITIAL_EXPLORATION = 0.25;
    static constexpr float DEFAULT_EXPLORATION_DECAY = 0.9999999;
    static constexpr float DEFAULT_INITIALQ = 1.0;
    static constexpr float DEFAULT_LEARNING_RATE = 0.01;
    static constexpr long NPOLICIES = std::pow(NACTIONS, NSTATES);

    std::array<std::array<float, NACTIONS>, NSTATES> Qtable;
    std::array<int, NSTATES> bestAction;
    float discount;     // exponential decay factor of future reward
    float learningRate;
    float pExplore; // probability of exploring
    float exploreDecay; // rate of exploration decay
    int lastAction;
    int lastState = -1;
    uint trainingStepsSinceLastPolicyChange = 0;
    std::uniform_real_distribution<float> uniformReal;
    std::uniform_int_distribution<int> randomActionChooser;
//        std::function<void(const TabularQPolicy<INTERFACE> &)> policyChangeHook;
    double totalReward;
    uint   nTrainingSteps;

    QTablePolicy(
            float discount = DEFAULT_DISCOUNT,
            float exploration = DEFAULT_INITIAL_EXPLORATION, // probability of not taking the best action
            float explorationDecay = DEFAULT_EXPLORATION_DECAY,
            float initialQ = DEFAULT_INITIALQ,
            float learningRate = DEFAULT_LEARNING_RATE
    ) :
            discount(discount), learningRate(learningRate),
            pExplore(exploration), exploreDecay(explorationDecay),
            uniformReal(0.0, 1.0), randomActionChooser(0, NACTIONS - 1),
            totalReward(0.0f), nTrainingSteps(0) {
        std::uniform_int_distribution<int> actionDist(0, NACTIONS - 1);

        // initialise Q values and set random bestAction
        for (int state = 0; state < NSTATES; ++state) {
            for (int action = 0; action < NACTIONS; ++action) Qtable[state][action] = initialQ;
            bestAction[state] = actionDist(deselby::Random::gen);
        }
    }


    // Sets the policy in the following way:
    // 'policyID' is a VIEW::NSTATES digit number in base VIEW::NACTIONS
    // such that the n'th digit identifies the bast action of the n'th state
    // For each state, the best action gets a Q-value of qMax and all other
    // actions get a Q-value of qMin.
    //
    // In a cartesian Q-value phase space, this point is the furthest point
    // from the policy boundaries while still being within the [qMin:qMax]
    // hypercube. The idea being that if we start at this point, then if
    // there is an attractor within this policy partition, then this point
    // should be within its basin of attraction.
    void setPolicy(long policyID, float qMin, float qMax) {
        for (int state = 0; state < NSTATES; ++state) {
            bestAction[state] = policyID % NACTIONS;
            for (int act = 0; act < NACTIONS; ++act) {
                Qtable[state][act] = (act == bestAction[state]) ? qMax : qMin;
            }
            policyID /= NACTIONS;
        }
    }

    void setRandomPolicy(float qMin, float qMax) {
        setPolicy(deselby::Random::nextLong(0, nPolicies()), qMin, qMax);
    }

    // calculates the ID of the current policy
    long policyID() const {
        long id = 0;
        for (int state = 0; state < NSTATES; ++state) {
            id += pow(NACTIONS, state) * bestAction[state];
        }
        return id;

    }

    static long nPolicies() {
        return std::pow(NACTIONS, NSTATES);
    }

    // version that doesn't require interface
    int getActionAndTrain(int newState, float reward) {
        if (lastState != -1) train(lastState, lastAction, reward, newState);
        lastState = newState;
        lastAction = getAction(newState);
        return lastAction;
    }


    int getAction(int state) {
        pExplore *= exploreDecay;
        return (uniformReal(deselby::Random::gen) <= pExplore) ? randomActionChooser(deselby::Random::gen)
                                                               : bestAction[state];
    }

//        std::pair<int,bool> getActionAndTrain(int observation, float reward) {
//            bool policyHasChanged = false;
//            if(lastState != -1) policyHasChanged = train(lastState, lastAction, reward, observation);
//            return {getAction(observation), policyHasChanged};
//        }

//        bool train(float reward, int observation) {
//            return train(lastState, lastAction, reward, observation);
//        }

    // Single training step for one action.
    // At equilibrium we have
    // Q(s0,a) = reward(s0,a,s1) + discount * max_a'(Q(s1,a'))
    // so we relax the table to equilibrium by setting
    // Q(s0,a) <- (1-l)Q(s0,a) + l*(reward(s0,a,s1) + discount * max_a'(Q(s1,a')))
    bool train(int startState, int action, float reward, int endState) {
//            std::cout << "training..." << std::endl;
        ++trainingStepsSinceLastPolicyChange;
        bool policyHasChanged = false;
        float forwardQ = reward + discount * Qtable[endState][bestAction[endState]];
        Qtable[startState][action] = (1.0 - learningRate) * Qtable[startState][action] + learningRate * forwardQ;
        for (int a = 0; a < NACTIONS; ++a) {
            if (Qtable[startState][a] > Qtable[startState][bestAction[startState]]) {
                policyHasChanged = true;
                trainingStepsSinceLastPolicyChange = 0;
//                    std::cout << "New policy " << policyID() << std::endl;
                bestAction[startState] = a;
            }
        }
//            if(policyHasChanged) policyChangeHook(*this);
        totalReward += reward;
        ++nTrainingSteps;
        return policyHasChanged;
    }

    void resetTrainingStats() {
        totalReward = 0.0;
        nTrainingSteps = 0;
    }

    double getMeanReward() const { return totalReward/nTrainingSteps; }

};


#endif //MULTIAGENTGOVERNMENT_QTABLEPOLICY_H
