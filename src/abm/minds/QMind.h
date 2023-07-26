//
// Created by daniel on 24/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_QMIND_H
#define MULTIAGENTGOVERNMENT_QMIND_H

#include <bitset>

namespace abm::minds {

    template<class T>
    concept QFunction = requires(T qFunction, T::input_type input, T::action_type act) {
        { qFunction.predict(input) } -> std::same_as<typename T::output_type>;
        { qFunction.train(input, act, 1.0, input, true) };
    };

    /**
     *
     * @tparam QFUNCTION    class that approximates a Q-function from body states to vectors of Q-values over acts
     *                      and implements a train function
     * @tparam POLICY       class that converts a vector of Q-values and a legal-act mask to an act to perform.
     */
    template<QFunction QFUNCTION, class POLICY>
    class QMind {
    public:

        typedef QFUNCTION::input_type   observation_type;
        typedef QFUNCTION::action_type  action_type;

        QFUNCTION   qFunction;
        POLICY      policy;

        QMind(QFUNCTION qfunction, POLICY policy): qFunction(std::move(qfunction)), policy(std::move(policy)) { }

        // --- Mind interface

        /**
         * @param body  current state of the body from which we base our decision to act
         * @param legalActs mask indicating which acts are physically possible from this state
         * @param reward reward since the last decision point
         * @return decision how to act in the current situation
         **/
        action_type act(observation_type observation, const std::bitset<action_type::size> &legalActs) {
            return policy.sample(qFunction.predict(observation), legalActs);
        }

        template<class STATE> requires(std::is_convertible_v<STATE,observation_type>)
        void train(const STATE &startState, action_type action, const double &reward, const STATE &endState, bool isEnd) {
            qFunction.train(startState, action, reward, endState, isEnd);
        }

    };
}

#endif //MULTIAGENTGOVERNMENT_QMIND_H
