// More efficient way of dealing with many timestepping agents, this
// agent keeps a vector of timestep functions and schedules itself every timestep to
// execute all of the timestepping functions.
//
// Created by daniel on 04/06/23.
//

#ifndef MULTIAGENTGOVERNMENT_TIMESTEPMANAGER_H
#define MULTIAGENTGOVERNMENT_TIMESTEPMANAGER_H


#include <vector>
#include <functional>
#include "../Schedule.h"

namespace abm {
    namespace agents {

        template<class EXECUTIONPOLICY, class TIMETYPE = uint>
        class TimestepManager: public std::vector<std::function<void()>> {
        public:
            const EXECUTIONPOLICY &execPolicy;
            TIMETYPE stepDuration;

            TimestepManager(const EXECUTIONPOLICY &execPolicy = std::execution::par_unseq, TIMETYPE stepDuration = 1):
            execPolicy(execPolicy),
            stepDuration(stepDuration) {}

            Schedule<TIMETYPE> start(TIMETYPE time = 0) {
                return Schedule<TIMETYPE>([this, time]() { this->handleStep(time); }, time);
            }

            Schedule<TIMETYPE> handleStep(TIMETYPE time) {
                std::for_each(
                        execPolicy,
                        this->begin(),
                        this->end(),
                        [](std::function<void()> &timestep) { return timestep(); });
                TIMETYPE nextStepTime = time + stepDuration;
                return Schedule<TIMETYPE>([this, nextStepTime]() { this->handleStep(nextStepTime); }, nextStepTime);
            }
        };

    }
}


#endif //MULTIAGENTGOVERNMENT_TIMESTEPMANAGER_H
