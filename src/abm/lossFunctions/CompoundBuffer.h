//
// Created by daniel on 17/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_COMPOUNDBUFFER_H
#define MULTIAGENTGOVERNMENT_COMPOUNDBUFFER_H

#include <cstddef>
#include <tuple>

template<class...BUFFERS>
class CompoundBuffer {
public:



    template<class INPUTS>
    static auto getLoss(INPUTS &inputs, BUFFERS &&...buffers) {
        size_t col = 0;
        return std::tuple(buffers.getLoss(inputs.cols(col, col += buffers.nPoints()))...); // buffer has a percentage set so knows the size of its own loss (so also is a loss)
    }

};


#endif //MULTIAGENTGOVERNMENT_COMPOUNDBUFFER_H
