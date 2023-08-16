//
// Created by daniel on 16/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_MEANREWARDMINDWRAPPER_H
#define MULTIAGENTGOVERNMENT_MEANREWARDMINDWRAPPER_H

#include <cmath>
#include "Mind.h"


namespace abm {

    /** Use this class to wrap a Mind in order to record the exponentially weighted mean
     * flux of reward, which we define as
     *
     * m(T) = int_{-inf}^{T} w(t-T)f(t) dt
     *
     * where
     *   w(t-T) = ke^{k(t-T)} is a weighting that sums to 1 between -inf and T and decays into the past
     * and
     *   f(t) is the historical flux of reward per unit time.
     *
     * Fluxes of reward are considered to be uniform between actions, and we take one unit of time to be
     * the time between actions so a reward of r_{T+1} at time T+1 is considered to be a flux
     * f(t) = r_{T+1} for T < t <= T+1.
     *
     * So,
     *
     * m(T+1)   = int_{-inf}^{T+1} ke^{k(t-(T+1))}f(t) dt
     *          = int_{-inf}^{T}  e^{-k} ke^{k(t-T)}f(t) dt + int_{T}^{T+1} ke^{k(t-(T+1))}f(t) dt
     *          = e^{-k}m(T) + r_{T+1} (1 - e^{-k})
     * Letting d = e^-k gives the recurrence relation
     * m(T+1)   = d.m(T) + (1-d)r_{T+1}
     *
     * At the end of an episode, if the last agent to receive a message is the first to move
     * in the next episode, then the next episode begins immediately after the end of the last.
     * However, if the last agent to send a message is the first to move in the next episode
     * then we assume there is a 1/2 unit delay between episodes. In this way, agents always
     * have one unit of time between actions, even between episodes.
     *
     */
    template<Mind MIND>
    class MeanRewardMindWrapper : public MIND {
    public:
        typedef MIND::observation_type observation_type;
        typedef MIND::action_mask action_mask;
        typedef MIND::reward_type reward_type;

        /** The exponentialy weighted mean reward */
        double  meanReward = 0.0;
    private:
        double  endOfEpisodeReward = 0.0;
        const double meanRewardDecay;
    public:

        /** Default constructed mind.
         * @param exponentialDecayRatePerTransaction this is the discount of rewards into the past applied when calculating the mean reward
         * (note that this is distinct from the discount used to calculate expected rewards into the future)
         */
        MeanRewardMindWrapper(double exponentialDecayRatePerTransaction) requires std::is_default_constructible_v<MIND> :
        meanRewardDecay(exponentialDecayRatePerTransaction) { }

        /** Use this to construct a Mind in place, send arguments for a constructor of MIND.
         * @param exponentialDecayRatePerTransaction this is the discount of rewards into the past applied when calculating the mean reward
         * (note that this is distinct from the discount used to calculate expected rewards into the future)
         */
        template<typename... MindConstructorArgs> requires std::is_constructible_v<MIND,MindConstructorArgs...>
        MeanRewardMindWrapper(double exponentialDecayRatePerTransaction, MindConstructorArgs &&... mindConstructorArgs):
                meanRewardDecay(exponentialDecayRatePerTransaction),
                MIND(std::forward<MindConstructorArgs>(mindConstructorArgs)...) { }

        /** Use this to save having to define the temlated type of Mind. Moves the supplied mind into this.
         * @param exponentialDecayRatePerTransaction this is the discount of rewards into the past applied when calculating the mean reward
         * (note that this is distinct from the discount used to calculate expected rewards into the future)
         * @param mind Mind to move into this.
         */
        MeanRewardMindWrapper(double exponentialDecayRatePerTransaction, MIND && mind):
                meanRewardDecay(exponentialDecayRatePerTransaction),
                MIND(std::move(mind)) { }

        /** Use this to save having to define the temlated type of Mind. Copies the supplies mind to this.
         * @param exponentialDecayRatePerTransaction this is the discount of rewards into the past applied when calculating the mean reward
         * (note that this is distinct from the discount used to calculate expected rewards into the future)
         * @param mind Mind to copy into this.
         */
        MeanRewardMindWrapper(double exponentialDecayRatePerTransaction, const MIND & mind):
                meanRewardDecay(exponentialDecayRatePerTransaction),
                MIND(mind) { }


        void endEpisode(double residualReward) {
            endOfEpisodeReward = residualReward;
            MIND::endEpisode(residualReward);
        }


        auto act(observation_type observation, action_mask actMask, reward_type reward) {
            meanReward = meanReward*meanRewardDecay + (1.0-meanRewardDecay)*(reward + endOfEpisodeReward);
            endOfEpisodeReward = 0.0;
            return MIND::act(observation, actMask, reward);
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_MEANREWARDMINDWRAPPER_H
