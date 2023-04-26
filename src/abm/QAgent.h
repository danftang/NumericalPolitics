//
// Created by daniel on 17/03/23.
//

#ifndef MULTIAGENTGOVERNMENT_QAGENT_H
#define MULTIAGENTGOVERNMENT_QAGENT_H

#include <concepts>
#include <algorithm>
#include "DiscreteEventSimulation.h"
#include "Schedule.h"

namespace abm {


    template<class INTERFACE>
    concept FiniteAgentInterface = requires(INTERFACE interface, INTERFACE::time_type time, int action) {
//        { VIEW::NSTATES } -> std::convertible_to<std::size_t>;
//        { VIEW::NACTIONS } -> std::convertible_to<std::size_t>;
        typename INTERFACE::time_type;
        { interface.observe(time) } -> std::convertible_to<int>;
        { interface.reward(time) } -> std::convertible_to<int>;
        { interface.act(time, action) };
    };

    template<class POLICY>
    concept AgentPolicy = requires(POLICY policy, int observation, float reward, int action) {
        { policy.getAction(observation) } -> std::convertible_to<int>;
        { policy.train(observation, action, reward, observation) };
    };

    // A Q
    template<AgentPolicy POLICY, FiniteAgentInterface INTERFACE, INTERFACE::time_type DELTAT=1>
    class QAgent { // : public std::enable_shared_from_this<QAgent<POLICY,INTERFACE>> {
    public:
        typedef POLICY policy_type;
        typedef INTERFACE interface_type;
        typedef INTERFACE::time_type time_type;

        POLICY policy;
        INTERFACE interface;
        int  currentState=-1;
        int  action;
        bool trainOnStep;
        bool policyChangedLastStep;
        time_type time;

        QAgent(POLICY policy = POLICY(), INTERFACE interface = INTERFACE()): policy(std::move(policy)), interface(std::move(interface)) {
            time = 0;
//            currentStep.task = [this]() { return step(); };
            trainOnStep = true;
            policyChangedLastStep = false;
        }

        Schedule<time_type> step() {
            int observation = interface.observe(time);
            if(trainOnStep) {
                policyChangedLastStep = false;
                if (currentState != -1) {
                    float reward = interface.reward(time);
                    policyChangedLastStep = policy.train(currentState, action, reward, observation);
                }
                currentState = observation;
            }
            action = policy.getAction(observation);
            interface.act(time, action);
            time += DELTAT;
            return Schedule<time_type>(*this, time);
        }

        // requires: interface.reward, interface.act, policy.getAction, policy.train
        // message handlers call this method on state transition
        Schedule<time_type> stateTransition(int newState, time_type time) {
            if(trainOnStep) {
                policyChangedLastStep = false;
                if (currentState != -1) {
                    float reward = interface.reward(time);
                    policyChangedLastStep = policy.train(currentState, action, reward, newState);
                }
                currentState = newState;
            }
            action = policy.getAction(newState);
            return interface.act(time, action);
        }

        // version that doesn't require interface
        int stateTransition(int newState, float reward) {
            if(trainOnStep) {
                policyChangedLastStep = false;
                if (currentState != -1) {
                    policyChangedLastStep = policy.train(currentState, action, reward, newState);
                }
                currentState = newState;
            }
            return policy.getAction(newState);
        }


    };
};

#endif //MULTIAGENTGOVERNMENT_QAGENT_H
