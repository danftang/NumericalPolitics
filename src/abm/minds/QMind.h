//
// Created by daniel on 24/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_QMIND_H
#define MULTIAGENTGOVERNMENT_QMIND_H

#include <bitset>
#include <optional>

namespace abm::minds {

    template<class T>
    concept HasPredict = requires(T qFunction, T::input_type input) {
        { qFunction.predict(input) };
    };

    template<class T>
    concept HasTimestepTraining = requires(T qFunction, T::input_type input, T::action_type act) {
        { qFunction.train(input, act, 1.0, input, true) };
    };


/**
     *
     * @tparam QFUNCTION    class that approximates a Q-function from body states to vectors of Q-values over acts
     *                      and implements a train function
     * @tparam POLICY       class that converts a vector of Q-values and a legal-act mask to an act to perform.
     */
    template<HasPredict QFUNCTION, class POLICY>
    class QMind: public QFUNCTION { // A QMind is a QFunction with an act member
    public:

        typedef QFUNCTION::input_type   observation_type;
        typedef POLICY::action_type  action_type;
        typedef double  reward_type;
        typedef std::bitset<action_type::size> action_mask;

        POLICY      policy;

    private:
        std::optional<observation_type> lastState;
        action_type lastAction;
    public:

        QMind(QFUNCTION qfunction, POLICY policy): QFUNCTION(std::move(qfunction)), policy(std::move(policy)) { }

        // --- Mind interface

        /**
         * @param body  current state of the body from which we base our decision to act
         * @param legalActs mask indicating which acts are physically possible from this state
         * @param reward reward since the last decision point
         * @return decision how to act in the current situation
         **/
        action_type act(observation_type observation, const action_mask &legalActs, reward_type rewardFromLastChoice) requires HasTimestepTraining<QFUNCTION> {
            if(lastState.has_value()) this->train(std::move(lastState.value()), lastAction, rewardFromLastChoice, observation, false);
            lastAction = policy.sample(this->predict(observation), legalActs);
            lastState = std::move(observation);
            return lastAction;
        }


        /** act with no training
         *
         * @param observation
         * @param legalActs
         * @return
         */
        action_type act(observation_type observation, const action_mask &legalActs) {
            lastAction = policy.sample(this->predict(observation), legalActs);
            lastState = std::move(observation);
            return lastAction;
        }


        template<Body BODY> requires std::convertible_to<BODY,observation_type>
        action_type operator()(BODY body) {
            return act(body, body.legalActs());
        }


        void endEpisode(double finalReward) requires HasTimestepTraining<QFUNCTION> {
            if(lastState.has_value()) this->train(lastState.value(), lastAction, finalReward, lastState.value(), true);
            lastState.reset();
            assert(!lastState.has_value());
        }

//        template<class STATE> requires(std::is_convertible_v<STATE,observation_type>)
//        void train(const STATE &startState, action_type action, const double &reward, const STATE &endState, bool isEnd) {
//            qFunction.train(startState, action, reward, endState, isEnd);
//        }

    };
}

#endif //MULTIAGENTGOVERNMENT_QMIND_H
