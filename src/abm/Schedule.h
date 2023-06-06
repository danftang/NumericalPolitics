// A schedule of tasks that are executed in order.
// If two or more tasks are scheduled at the same time, there
// is no guarantee what order they are executed in, and may be
// executed on different threads, so for reproducible results
// tasks scheduled at the same time should be commutative and
// thread-safe.
//
// Created by daniel on 06/04/23.
//

#ifndef MULTIAGENTGOVERNMENT_SCHEDULE_H
#define MULTIAGENTGOVERNMENT_SCHEDULE_H

#include <functional>
#include <forward_list>
#include <map>
#include <numeric>
#include <ostream>
#include <execution>

namespace abm {

    template<class TIME>
    class Schedule;

//    template<class AGENT, class TIME>
//    concept HasStepFunction = requires(AGENT agent) {
//        { agent.step() } -> std::convertible_to<Schedule<TIME>>;
//    };

//    template<class T, class TIME>
//    concept IterableOfSteppables = requires(T container) {
//        { *container.begin() } -> HasStepFunction<TIME>;
//        { *container.end() } -> HasStepFunction<TIME>;
//    };

//    template<class T, class TIME>
//    concept Task = requires(T callable) {
//        { callable() } -> std::convertible_to<Schedule<TIME>>;
//    };


    template<class TIME>
    class Schedule {
    public:
        typedef TIME time_type;
        typedef std::function<Schedule<TIME>()> task_type;
        typedef std::forward_list<task_type> tasklist_type;

        std::map<TIME, tasklist_type> tasks; // TODO: better to be a sorted vector of <time,list> pairs?

        Schedule() {}

        inline Schedule(task_type task, TIME time = 0) {
            insert(std::move(task), time);
        }

        Schedule(std::forward_list<task_type> &&taskList, TIME time = 0) {
            tasks[time] = std::move(taskList);
        }

//        Schedule(std::initializer_list<Schedule<TIME> &&> schedulesToMerge) {
//            auto pSchedule = schedulesToMerge.begin();
//            if(schedulesToMerge.size() > 0) (*this) = std::move(*pSchedule);
//            ++pSchedule;
//            while(pSchedule != schedulesToMerge.end()) {
//                merge(std::move(*pSchedule));
//                ++pSchedule;
//            }
//        }

//        template<HasStepFunction<TIME> AGENT>
//        Schedule(AGENT &agent, TIME time = 0) {
//            insert(agent, time);
//        }
//
//        template<IterableOfSteppables<TIME> AGENTCONTAINER>
//        Schedule(AGENTCONTAINER &agents, TIME time = 0) {
//            insert(agents, time);
//        }

//        template<Task<TIME> TASK>
        inline void insert(task_type task, TIME time) {
            tasks[time].push_front(std::move(task));
        }

        inline void insert(std::forward_list<task_type> &&taskList, TIME time) {
            std::forward_list<task_type> &currentTaskList = tasks[time];
            if (currentTaskList.empty()) {
                currentTaskList = std::move(taskList);
            } else {
                currentTaskList.splice_after(currentTaskList.begin(), std::move(taskList));
            }
        }

//        template<HasStepFunction<TIME> AGENT>
//        inline void insert(AGENT &agent, TIME time) {
//            insert([&agent]() { return agent.step(); }, time);
//        }
//
//        template<IterableOfSteppables<TIME> AGENTCONTAINER>
//        inline void insert(AGENTCONTAINER &agents, TIME time) {
//            tasklist_type &taskList = tasks[time];
//            for (auto &agent: agents) taskList.push_front([&agent]() { return agent.step(); });
//        }

        Schedule<TIME> operator+(Schedule<TIME> &&other) &&{
            return std::move(merge(std::move(other)));
        }

        Schedule<TIME> &operator+=(Schedule<TIME> &&other) {
            merge(std::move(other));
            return *this;
        }


        Schedule<TIME> &merge(Schedule<TIME> &&other) {
            for (auto &element: other.tasks) {
                insert(std::move(element.second), element.first);
            }
            return *this;
        }

        static Schedule<TIME> mergeAndDestroy(Schedule<TIME> &a, Schedule<TIME> &&b) {
            return std::move(a.merge(std::move(b)));
        }

        template<class ExecutionPolicy>
        void execFirstEntry(ExecutionPolicy &&executionPolicy) {
            std::forward_list<task_type> &taskList = tasks.begin()->second;
//            time_type time = tasks.begin()->first;
            Schedule<TIME> newTasks = std::transform_reduce(
//                std::forward<ExecutionPolicy>(executionPolicy),
                    std::execution::par,
                    taskList.begin(),
                    taskList.end(),
                    Schedule<TIME>(),
                    &mergeAndDestroy,
                    [](task_type &task) { return task(); });
            tasks.erase(tasks.begin());
            merge(std::move(newTasks));
        }

        // Execute until no more tasks
        template<class ExecutionPolicy = const std::execution::parallel_policy &>
        void exec(ExecutionPolicy &&executionPolicy = std::execution::par) {
            while (!tasks.empty()) execFirstEntry(std::forward<ExecutionPolicy>(executionPolicy));
        }

        template<class ExecutionPolicy = const std::execution::parallel_policy &>
        void execUntil(std::function<bool()> hasEnded, ExecutionPolicy &&executionPolicy = std::execution::par) {
            if (!tasks.empty()) {
                do {
                    execFirstEntry(std::forward<ExecutionPolicy>(executionPolicy));
                } while (!tasks.empty() && !hasEnded());
            }
        }

        TIME time() {
            return tasks.empty() ? 0 : tasks.begin()->first;
        }

        friend std::ostream &operator<<(std::ostream &out, const Schedule<TIME> &schedule) {
            std::forward_list<int> f;
            for (auto &entry: schedule.tasks) {
                out << entry.first << " -> " << std::distance(entry.second.begin(), entry.second.end()) << " tasks"
                    << std::endl;
            }
            return out;
        }
    };
}
#endif //MULTIAGENTGOVERNMENT_SCHEDULE_H
