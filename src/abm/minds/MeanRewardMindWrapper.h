//
// Created by daniel on 16/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_MEANREWARDMINDWRAPPER_H
#define MULTIAGENTGOVERNMENT_MEANREWARDMINDWRAPPER_H

#include <cmath>
#include "../Mind.h"
#include "../Agent.h"


namespace abm::minds {

    /** Use this class to wrap a Mind in order to record the exponentially weighted mean
     *  reward, which we define as
     *
     * E_n[R] = a_n.R_n + a_n.r.R_{n-1} + a_n.r^2.R_{n-2} + ... + a_n.r^{n-1}.R_1
     *
     * where
     * R_t is the reward at time t,
     * r is a supplied constant decay rate
     * a_n = (1-r)/(1-r^n)
     * but we have the recurrence relation
     * E_n[R] = a_n.R_n + (a_n.r/a_{n-1}).E_{n-1}[R]
     * and, by expansion
     * a_n.r/a_{n-1} + a_n = (r-r^n)/(1-r^n) + (1-r)/(1-r^n) = 1
     * so
     * a_n.r/a_{n-1} = 1-a_n
     * and
     * E_n[R] = a_n.R_n + (1-a_n).E_{n-1}[R]
     *
     * At the end of an episode, if the last agent to receive a message is the first to move
     * in the next episode, then the next episode begins immediately after the end of the last.
     * However, if the last agent to send a message is the first to move in the next episode
     * then we assume there is a 1/2 unit delay between episodes. In this way, agents always
     * have one unit of time between actions, even between episodes.
     *
     */
     //// USE abm::callbacks::Meaneward
//    template<class MIND>
//    class MeanRewardMindWrapper : public MIND {
//    public:
//        typedef MIND::observation_type observation_type;
//        typedef MIND::action_mask action_mask;
//        typedef MIND::reward_type reward_type;
//
//        /** The exponentialy weighted mean reward */
//        double  meanReward = 0.0;
//        int     nSamples = 0;
//    private:
//        double  endOfEpisodeReward = 0.0;
//        const double meanRewardDecay;
//    public:
//
//        /** Default constructed mind.
//         * @param exponentialDecayRatePerTransaction this is the discount of rewards into the past applied when calculating the mean reward
//         * (note that this is distinct from the discount used to calculate expected rewards into the future)
//         */
//        MeanRewardMindWrapper(double exponentialDecayRatePerTransaction) requires std::is_default_constructible_v<MIND> :
//        meanRewardDecay(exponentialDecayRatePerTransaction) { }
//
//        /** Use this to construct a Mind in place, send arguments for a constructor of MIND.
//         * @param exponentialDecayRatePerTransaction this is the discount of rewards into the past applied when calculating the mean reward
//         * (note that this is distinct from the discount used to calculate expected rewards into the future)
//         */
//        template<typename... MindConstructorArgs> requires std::is_constructible_v<MIND,MindConstructorArgs...>
//        MeanRewardMindWrapper(double exponentialDecayRatePerTransaction, MindConstructorArgs &&... mindConstructorArgs):
//                meanRewardDecay(exponentialDecayRatePerTransaction),
//                MIND(std::forward<MindConstructorArgs>(mindConstructorArgs)...) { }
//
//        /** Use this to save having to define the temlated type of Mind. Moves the supplied mind into this.
//         * @param exponentialDecayRatePerTransaction this is the discount of rewards into the past applied when calculating the mean reward
//         * (note that this is distinct from the discount used to calculate expected rewards into the future)
//         * @param mind Mind to move into this.
//         */
//        MeanRewardMindWrapper(double exponentialDecayRatePerTransaction, MIND && mind):
//                meanRewardDecay(exponentialDecayRatePerTransaction),
//                MIND(std::move(mind)) { }
//
//        /** Use this to save having to define the temlated type of Mind. Copies the supplies mind to this.
//         * @param exponentialDecayRatePerTransaction this is the discount of rewards into the past applied when calculating the mean reward
//         * (note that this is distinct from the discount used to calculate expected rewards into the future)
//         * @param mind Mind to copy into this.
//         */
//        MeanRewardMindWrapper(double exponentialDecayRatePerTransaction, const MIND & mind):
//                meanRewardDecay(exponentialDecayRatePerTransaction),
//                MIND(mind) { }
//
//
//        void endEpisode(double residualReward) {
//            endOfEpisodeReward = residualReward;
//            MIND::endEpisode(residualReward);
//        }
//
//
//        auto act(observation_type observation, action_mask actMask, reward_type reward) {
//            double a_n = (1.0-meanRewardDecay)/(1.0-std::pow(meanRewardDecay, ++nSamples));
//            meanReward *= 1.0-a_n;
//            meanReward += a_n*(reward + endOfEpisodeReward);
//            endOfEpisodeReward = 0.0;
//            return MIND::act(observation, actMask, reward);
//        }
//    };


}

#endif //MULTIAGENTGOVERNMENT_MEANREWARDMINDWRAPPER_H
