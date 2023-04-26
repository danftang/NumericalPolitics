// Represents a communication channel between agents that uses a fixed sized array to store historical values
// For a buffer of size SIZE, the value of the variable at time t is stored at location t modulo SIZE.
// If t >= SIZE then the value at time t-SIZE is overwritten,
//
// It is assumed that
//   * only one thread can write to the buffer at any time.
//   * A read on an overwritten time is guaranteed never to occur.
// Together these imply thread safety since there is only one writer and there will never be a
// data race between the writer and a reader since this implies the possibility of
// a read on an overwritten time.
//
// Created by daniel on 27/02/23.
//

#ifndef MULTIAGENTGOVERNMENT_ARRAYCHANNEL_H
#define MULTIAGENTGOVERNMENT_ARRAYCHANNEL_H

#include <array>

namespace abm {
    template<class T, int SIZE>
    class ArrayChannel {
    public:
        std::array<T, SIZE> buffer;

        T &operator[](size_t time) { return buffer[time % SIZE]; }

        const T &operator[](size_t time) const { return buffer[time % SIZE]; }
    };
};

#endif //MULTIAGENTGOVERNMENT_ARRAYCHANNEL_H
