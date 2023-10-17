//
// Created by daniel on 25/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_RANDOMQSTEPREPLAY_H
#define MULTIAGENTGOVERNMENT_RANDOMQSTEPREPLAY_H

#include "mlpack.hpp"

namespace abm {
    class RandomQStepReplay {
    public:
        typedef arma::mat StateType;
        typedef int ActionType;

        RandomQStepReplay() :
                batchSize(0),
                capacity(0),
                position(0),
                full(false)
//                nSteps(0)
                { /* Nothing to do here. */ }

        /**
         * Construct an instance of random experience replay class.
         *
         * @param batchSize Number of examples returned at each sample.
         * @param capacity Total memory size in terms of number of examples.
         * @param nSteps Number of steps to look in the future.
         * @param dimension The dimension of an encoded state.
         */
        RandomQStepReplay(const size_t batchSize,
                          const size_t capacity,
                          const size_t state_dimension
        ) :
                batchSize(batchSize),
                capacity(capacity),
                position(0),
                full(false),
                states(state_dimension, capacity),
                actions(capacity),
                rewards(capacity),
                nextStates(state_dimension, capacity),
                isTerminal(capacity) { /* Nothing to do here. */ }

        /**
         * Store the given experience.
         *
         * @param state Given state.
         * @param action Given action.
         * @param reward Given reward.
         * @param nextState Given next state.
         * @param isEnd Whether next state is terminal state.
         */
        void Store(const StateType &state,
                   ActionType action,
                   const double &reward,
                   const StateType &nextState,
                   bool isEnd) {
            states.col(position) = state;
            actions[position] = action;
            rewards(position) = reward;
            nextStates.col(position) = nextState;
            isTerminal(position) = isEnd;
            position++;
            if (position == capacity) {
                full = true;
                position = 0;
            }
        }


        /**
         * Sample some experiences.
         *
         * @param sampledStates Sampled encoded states.
         * @param sampledActions Sampled actions.
         * @param sampledRewards Sampled rewards.
         * @param sampledNextStates Sampled encoded next states.
         * @param isTerminal Indicate whether corresponding next state is terminal
         *        state.
         */
        void Sample(arma::mat &sampledStates,
                    std::vector<ActionType> &sampledActions,
                    arma::rowvec &sampledRewards,
                    arma::mat &sampledNextStates,
                    arma::irowvec &isTerminal) {
            size_t upperBound = full ? capacity : position;
            arma::uvec sampledIndices = arma::randi<arma::uvec>(
                    batchSize, arma::distr_param(0, upperBound - 1));

            sampledStates = states.cols(sampledIndices);
            for (size_t t = 0; t < sampledIndices.n_rows; t++)
                sampledActions.push_back(actions[sampledIndices[t]]);
            sampledRewards = rewards.elem(sampledIndices).t();
            sampledNextStates = nextStates.cols(sampledIndices);
            isTerminal = this->isTerminal.elem(sampledIndices).t();
        }

        /**
         * Get the number of transitions in the memory.
         *
         * @return Actual used memory size
         */
        const size_t &Size() {
            return full ? capacity : position;
        }

//        /**
//         * Update the priorities of transitions and Update the gradients.
//         *
//         * @param * (target) The learned value
//         * @param * (sampledActions) Agent's sampled action
//         * @param * (nextActionValues) Agent's next action
//         * @param * (gradients) The model's gradients
//         */
//        void Update(arma::mat /* target */,
//                    std::vector<ActionType> /* sampledActions */,
//                    arma::mat /* nextActionValues */,
//                    arma::mat & /* gradients */) {
//            /* Do nothing for random replay. */
//        }

//        //! Get the number of steps for n-step agent.
//        const size_t &NSteps() const { return nSteps; }

        //! Locally-stored number of examples of each sample.
        size_t batchSize;
    private:

        //! Locally-stored total memory limit.
        size_t capacity;

        //! Indicate the position to store new transition.
        size_t position;

        //! Locally-stored indicator that whether the memory is full or not
        bool full;


        //! Locally-stored number of steps to look into the future.
//        size_t nSteps;


        //! Locally-stored buffer containing n consecutive steps.
//    std::deque<Transition> nStepBuffer;

        //! Locally-stored encoded previous states.
        arma::mat states;

        //! Locally-stored previous actions.
        std::vector<ActionType> actions;

        //! Locally-stored previous rewards.
        arma::rowvec rewards;

        //! Locally-stored encoded previous next states.
        arma::mat nextStates;

        //! Locally-stored termination information of previous experience.
        arma::irowvec isTerminal;
    };
}

#endif //MULTIAGENTGOVERNMENT_RANDOMQSTEPREPLAY_H
