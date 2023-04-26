// An agent based simulation where agents schedule events at given times
//
// Created by daniel on 21/03/23.
//

#ifndef MULTIAGENTGOVERNMENT_DISCRETEEVENTSIMULATION_H
#define MULTIAGENTGOVERNMENT_DISCRETEEVENTSIMULATION_H

#include <queue>
#include <functional>

class DiscreteEventSimulation {
public:
    class Event {
    public:
        float time=-1.0;
        std::function<bool(DiscreteEventSimulation &)>   runnable;

//        Event(float time, const std::function<bool(DiscreteEventSimulation &)> &runnable): time(time), runnable(runnable) { }
        Event(float time, std::function<bool(DiscreteEventSimulation &)> &&runnable): time(time), runnable(std::move(runnable)) { }

        bool operator <(const Event &other) const { return time < other.time; }
        bool run(DiscreteEventSimulation &sim) const { return runnable(sim); }
    };

    std::priority_queue<Event>  _schedule;
    float end_time = std::numeric_limits<float>::infinity();

    void run() {
        while(!_schedule.empty()) {
            _schedule.top().run(*this);
            _schedule.pop();
        }
    }

    inline bool schedule(float scheduledTime, std::function<bool(DiscreteEventSimulation &)> &&runnable) {
        if(scheduledTime >= time() && scheduledTime < end_time) {
            _schedule.emplace(scheduledTime, std::move(runnable));
            return true;
        }
        return false;
    }

    float time() { return _schedule.empty()?0.0:_schedule.top().time; }

};


#endif //MULTIAGENTGOVERNMENT_DISCRETEEVENTSIMULATION_H
