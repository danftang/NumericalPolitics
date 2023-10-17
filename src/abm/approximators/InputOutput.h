//
// Created by daniel on 09/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_INPUTOUTPUT_H
#define MULTIAGENTGOVERNMENT_INPUTOUTPUT_H

namespace approximators::events {
    /** Represents an observation of a (possibly batched) input/output pair of a function.
     * @tparam INPUT
     * @tparam OUTPUT
     */
    template<class INPUT, class OUTPUT>
    struct InputOutput {
        INPUT input;
        OUTPUT output;
    };
    template<class IN, class OUT>
    InputOutput(IN &&in, OUT &&out) -> InputOutput<IN,OUT>;
}

#endif //MULTIAGENTGOVERNMENT_INPUTOUTPUT_H
