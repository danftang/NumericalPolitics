//
// Created by daniel on 13/03/23.
//

#ifndef MULTIAGENTGOVERNMENT_QTABLEPOLICY_H
#define MULTIAGENTGOVERNMENT_QTABLEPOLICY_H

#include <array>
#include <cmath>
#include <random>
#include "../DeselbyStd/random.h"

namespace abm {

    template<int NSTATES, int NACTIONS>
    class QTablePolicy {
    public:
        static constexpr double DEFAULT_DISCOUNT = 0.99;
//        static constexpr double DEFAULT_EXPLORATION = 0.01;
        static constexpr double DEFAULT_INITIAL_EXPLORATION = 0.25;
        static constexpr double DEFAULT_EXPLORATION_DECAY = 0.9999999;
        static constexpr double DEFAULT_EXPLORATION_MINIMUM = 0.0025;
//        static constexpr double DEFAULT_INITIALQ = 1.0;
        static constexpr double DEFAULT_LEARNING_RATE = 0.001;
//        static constexpr long NPOLICIES = std::pow(NACTIONS, NSTATES);
//        static constexpr int ENDGAME_STATE = -1; // Inex of a state that has a fixed Q-value of 0 and no outgoing actions

        inline static bool verbose = false;

        std::array<std::array<double, NACTIONS>, NSTATES>   Qtable;
        std::array<std::array<uint, NACTIONS>, NSTATES>      nSamples;
        std::array<int, NSTATES> bestAction;
        double discount;     // exponential decay factor of future reward
        double learningRate;
        double pExplore; // probability of exploring
        double exploreDecay; // rate of exploration decay
        double exploreMin;
        int lastAction;
        int lastState = -1;
        uint trainingStepsSinceLastPolicyChange = 0;
        std::uniform_real_distribution<double> uniformReal;
        std::uniform_int_distribution<int> randomActionChooser;
//        std::function<void(const TabularQPolicy<INTERFACE> &)> policyChangeHook;
        double totalReward;
        uint nTrainingSteps;

        QTablePolicy(
                double discount = DEFAULT_DISCOUNT,
                double exploration = DEFAULT_INITIAL_EXPLORATION, // probability of not taking the best action
                double explorationDecay = DEFAULT_EXPLORATION_DECAY,
                double explorationMinimum = DEFAULT_EXPLORATION_MINIMUM,
  //              double initialQ = DEFAULT_INITIALQ,
                double learningRate = DEFAULT_LEARNING_RATE
        ) :
                discount(discount), learningRate(learningRate),
                pExplore(exploration), exploreDecay(explorationDecay), exploreMin(explorationMinimum),
                uniformReal(0.0, 1.0), randomActionChooser(0, NACTIONS - 1),
                totalReward(0.0f), nTrainingSteps(0) {
            std::uniform_int_distribution<int> actionDist(0, NACTIONS - 1);

            // initialise Q values and set random bestAction
            for (int state = 0; state < NSTATES; ++state) {
                for (int action = 0; action < NACTIONS; ++action) {
                    Qtable[state][action] = 0.0;
                    nSamples[state][action] = 0;
                }
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
        void setPolicy(long policyID, double qMin, double qMax) {
            for (int state = 0; state < NSTATES; ++state) {
                bestAction[state] = policyID % NACTIONS;
                for (int act = 0; act < NACTIONS; ++act) {
                    Qtable[state][act] = (act == bestAction[state]) ? qMax : qMin;
                }
                policyID /= NACTIONS;
            }
        }

        void setRandomPolicy(double qMin, double qMax) {
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


        void setExploration(double epsilon) {
            pExplore = epsilon;
        }

        // version that doesn't require interface
        int getActionAndTrain(int newState, double reward) {
            if (lastState != -1) train(lastState, lastAction, reward, newState, false);
            lastState = newState;
            lastAction = getAction(newState);
            return lastAction;
        }


        int getAction(int state) {
            if(verbose) {
                std::cout << state << ": ";
                for (int a = 0; a < NACTIONS; ++a) {
                    std::cout << Qtable[state][a] << ":" << nSamples[state][a];
                    if (a == bestAction[state]) std::cout << "* "; else std::cout << "  ";
                }
                std::cout << std::endl;
            }
            if (pExplore > exploreMin) pExplore *= exploreDecay;
            return (uniformReal(deselby::Random::gen) <= pExplore) ? randomActionChooser(deselby::Random::gen)
                                                                   : bestAction[state];
        }

//        std::pair<int,bool> getActionAndTrain(int observation, double reward) {
//            bool policyHasChanged = false;
//            if(stateBeforeLastAction != -1) policyHasChanged = train(stateBeforeLastAction, myLastAction, reward, observation);
//            return {getAction(observation), policyHasChanged};
//        }

//        bool train(double reward, int observation) {
//            return train(stateBeforeLastAction, myLastAction, reward, observation);
//        }

        // Single training step for one action.
        // At equilibrium we have
        // Q(s0,a) = reward(s0,a,s1) + discount * max_a'(Q(s1,a'))
        // so we relax the table to equilibrium by setting
        // Q(s0,a) <- (1-l)Q(s0,a) + l*(reward(s0,a,s1) + discount * max_a'(Q(s1,a')))
        //
        // If we weight the samples with the weights:
        // a_n, a_n r, a_n r^2, a_n r^3 ... a_n r^(n-1)
        //
        // The sum of the weights must be 1 so
        // S_n = a_n(1-r^n)/(1-r) = 1
        // so
        // a_n = (1-r)/(1-r^n)
        // so
        // a_{n+1} = a_n(1-r^n)/(1-r^{n+1})
        //
        // so we multiply the current Q value by (r-r^{n+1})/(1-r^{n+1}) and weight the current sample by (1-r)/(1-r^{n+1})
        //
        // But 1-((1-r)/(1-r^{n+1})) = (r-r^{n+1})/(1-r^{n+1})
        bool train(int startState, int action, double reward, int endState, bool isEndgame) {
//            std::cout << "training on " << startState << " " << action << " " << reward << " " << endState << std::endl;
            ++trainingStepsSinceLastPolicyChange;
            bool policyHasChanged = false;

            ++nSamples[startState][action];
            const double rrn = std::pow(1.0 - learningRate,nSamples[startState][action]);
            double sampleWeight = learningRate/(1.0-rrn);

            const double endStateQValue = (isEndgame ? 0.0 : Qtable[endState][bestAction[endState]]);
            const double forwardQ = reward + discount * endStateQValue;
            Qtable[startState][action] = (1.0-sampleWeight) * Qtable[startState][action] + sampleWeight * forwardQ;
            for (int a = 0; a < NACTIONS; ++a) { // TODO: optimise this
                if (Qtable[startState][a] > Qtable[startState][bestAction[startState]]) {
                    policyHasChanged = true;
                    trainingStepsSinceLastPolicyChange = 0;
                    bestAction[startState] = a;
                }
            }
            totalReward += reward;
            ++nTrainingSteps;
            return policyHasChanged;
        }

        void resetTrainingStats() {
            totalReward = 0.0;
            nTrainingSteps = 0;
        }

        double getMeanReward() const { return totalReward / nTrainingSteps; }

    };
}

#endif //MULTIAGENTGOVERNMENT_QTABLEPOLICY_H
