//
// Created by daniel on 24/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_QMIND_H
#define MULTIAGENTGOVERNMENT_QMIND_H

#include <bitset>

namespace abm {

    /**
     *
     * @tparam QFUNCTION    class that approximates a Q-function from body states to vectors of Q-values over acts
     *                      and implements a train function
     * @tparam POLICY       class that converts a vector of Q-values and a legal-act mask to an act to perform.
     */
    template<class QFUNCTION, class POLICY>
    class QMind {
    public:

        typedef int action_type;
        typedef QFUNCTION::input_type body_type;

        QFUNCTION qFunction;
        POLICY policy;

        QMind(QFUNCTION qfunction, POLICY policy): qFunction(std::move(qfunction)), policy(std::move(policy)) { }

        // --- Mind interface

        /**
         * @param body  current state of the body from which we base our decision to act
         * @param legalActs mask indicating which acts are physically possible from this state
         * @param reward reward since the last decision point
         * @return decision how to act in the current situation
         **/
        template<size_t action_size>
        action_type act(body_type body, const std::bitset<action_size> &legalActs) {
            return policy.sample(qFunction.predict(body), legalActs);
        }

        void train(body_type startState, action_type action, const double &reward, body_type endState, bool isEnd) {
            qFunction.train(startState, action, reward, endState, isEnd);
        }

    };
}

#endif //MULTIAGENTGOVERNMENT_QMIND_H
