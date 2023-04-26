// Represents a one-way communication channel between two agents
//
// Under this view, a channel is a history of values, so can be
// queried at a given time.
//
// An alternative view is that a channel is a receiver of objects.
// And so is just a callable handler of incoming objects.
// In this case, a channel can be just a std::function<Shedule(const OBJ &)>
// and an event is an OBJ, std::function<Schedule(const OBJ &)> pair which
// defines a task
//
// Alternatively, if an agent has a limited number of actions, each action
// can be just a task std::function<Schedule()>
//
// Created by daniel on 26/02/23.
//

#ifndef MULTIAGENTGOVERNMENT_CHANNEL_H
#define MULTIAGENTGOVERNMENT_CHANNEL_H

#include <queue>
#include <mutex>
#include <functional>
#include <future>

template<class MESSAGE>
class Channel {
public:
    std::deque<MESSAGE> messages;
    std::mutex          mutex;
    std::function<void(Channel<MESSAGE> &)>       readerCallback;

    void write(const MESSAGE &message) {
        std::unique_lock<std::mutex> lock(mutex);
        messages.push_back(message);
        lock.unlock();
        std::async(std::launch::async , readerCallback, *this);
    }

    void write(MESSAGE &&message) {
        std::unique_lock<std::mutex> lock(mutex);
        messages.push_back(message);
        lock.unlock();
    }

    MESSAGE read() {
        assert(!messages.empty()); // agent won't get scheduled unless there is a message to read
        MESSAGE message(std::move(messages.front()));
        std::unique_lock<std::mutex> lock(mutex);
        messages.pop_front();
        lock.unlock();
        return message;
    }

};


#endif //MULTIAGENTGOVERNMENT_CHANNEL_H
