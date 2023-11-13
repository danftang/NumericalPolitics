//
// Created by daniel on 26/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_CARTPOLE_H
#define MULTIAGENTGOVERNMENT_CARTPOLE_H

#include <random>
#include <bitset>
#include "mlpack.hpp"

// TODO: make this just take a Mind and provides a step?
namespace tests {
    class CartPoleEnvironment {
    public:

        enum action_type {
            left,
            right,
            size
        };

        typedef std::bitset<action_type::size> action_mask;

//        enum message_type {
//            close,
//            step
//        };

        static constexpr int dimension = 4;

        /**
         * Construct a Cart Pole instance using the given constants.
         *
         * @param maxSteps The number of steps after which the episode
         *    terminates. If the value is 0, there is no limit.
         * @param gravity The gravity constant.
         * @param massCart The mass of the cart.
         * @param massPole The mass of the pole.
         * @param length The length of the pole.
         * @param forceMag The magnitude of the applied force.
         * @param tau The time interval.
         * @param thetaThresholdRadians The maximum angle.
         * @param xThreshold The maximum position.
         */
        CartPoleEnvironment(const size_t maxSteps = 200,
                            const double gravity = 9.8,
                            const double massCart = 1.0,
                            const double massPole = 0.1,
                            const double length = 0.5,
                            const double forceMag = 10.0,
                            const double tau = 0.02,
                            const double thetaThresholdRadians = 12 * 2 * 3.1416 / 360,
                            const double successReward = 1.0,
                            const double xThreshold = 2.4) :
                maxSteps(maxSteps),
                gravity(gravity),
                massCart(massCart),
                massPole(massPole),
                totalMass(massCart + massPole),
                length(length),
                poleMassLength(massPole * length),
                forceMag(forceMag),
                tau(tau),
                thetaThresholdRadians(thetaThresholdRadians),
                xThreshold(xThreshold),
                successReward(successReward),
                stepsPerformed(0) {
            reset();
        }

        /**
         * Sets the state to a random start state and sets stepsPerformed to 0
         */
        void reset() {
            std::uniform_real_distribution<double> rand(-0.05, 0.05);
            std::default_random_engine gen;
            position = rand(gen);
            velocity = rand(gen);
            angle = rand(gen);
            angularVelocity = rand(gen);
            stepsPerformed = 0;
        }

        /**
         * Dynamics of Cart Pole instance. Get reward and next state based on current
         * state and current action.
         *
         * @param action The current action.
         * @return reward, it's 1.0
         */
        double actToReward(int action) {
            // Update the number of steps performed.
            stepsPerformed++;

            // Calculate acceleration.
            double force = (action == right ? forceMag : -forceMag);
            double cosTheta = std::cos(angle);
            double sinTheta = std::sin(angle);
            double temp = (force + poleMassLength * angularVelocity *
                                   angularVelocity * sinTheta) / totalMass;
            double thetaAcc = (gravity * sinTheta - cosTheta * temp) /
                              (length * (4.0 / 3.0 - massPole * cosTheta * cosTheta / totalMass));
            double xAcc = temp - poleMassLength * thetaAcc * cosTheta / totalMass;

            // Update states.
            position += tau * velocity,
            velocity += tau * xAcc,
            angle += tau * angularVelocity,
            angularVelocity += tau * thetaAcc;
            bool isEndOfEpisode = isTerminal();
            if (isEndOfEpisode) reset();
            return isEndOfEpisode ? 0.0 : 1.0;
        }

        template<class MIND>
        int episode(MIND &mind) {
            int nSteps = 0;
            std::bitset<2> actionMask = 3;
            reset();
            double reward = 0.0;
            do {
                int nextAct = mind.act(*this, actionMask, 0);
                reward = actToReward(nextAct);
                ++nSteps;
            } while(reward == 1.0);
            mind.endEpisode(1.0);
            return nSteps;
        }


        operator arma::mat::fixed<4, 1>() const {
            return {position, velocity, angle, angularVelocity};
        }

        /**
         * This function checks if the cart has reached the terminal state.
         *
         * @return true if state is a terminal state, otherwise false.
         */
        bool isTerminal() const {
//            std::cout << position << " " << angle << std::endl;
            if (maxSteps != 0 && stepsPerformed >= maxSteps) return true;
            if (std::abs(position) > xThreshold ||
                std::abs(angle) > thetaThresholdRadians)
                return true;

            return false;
        }

        //! Get the number of steps performed.
        size_t StepsPerformed() const { return stepsPerformed; }

        //! Get the maximum number of steps allowed.
        size_t MaxSteps() const { return maxSteps; }

        //! Set the maximum number of steps allowed.
        size_t &MaxSteps() { return maxSteps; }

        friend std::ostream &operator<<(std::ostream &out, const CartPoleEnvironment &cartPole) {
            out << "{" << cartPole.position << ", " << cartPole.angle << ", " << cartPole.velocity << ", "
                << cartPole.angularVelocity << "}";
            return out;
        }

    private:
        double position;
        double velocity;
        double angle;
        double angularVelocity;


        //! Locally-stored maximum number of steps.
        size_t maxSteps;

        //! Locally-stored gravity.
        double gravity;

        //! Locally-stored mass of the cart.
        double massCart;

        //! Locally-stored mass of the pole.
        double massPole;

        //! Locally-stored total mass.
        double totalMass;

        //! Locally-stored length of the pole.
        double length;

        //! Locally-stored moment of pole.
        double poleMassLength;

        //! Locally-stored magnitude of the applied force.
        double forceMag;

        //! Locally-stored time interval.
        double tau;

        //! Locally-stored maximum angle.
        double thetaThresholdRadians;

        //! Locally-stored maximum position.
        double xThreshold;

        //! reward at end of episode for keeping the pole upright for maxSteps
        double successReward;

        //! Locally-stored number of steps performed.
        size_t stepsPerformed;
    };
}

#endif //MULTIAGENTGOVERNMENT_CARTPOLE_H
