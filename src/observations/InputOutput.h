//
// Created by daniel on 06/09/23.
//

#ifndef MULTIAGENTGOVERNMENT_INPUTOUTPUT_H
#define MULTIAGENTGOVERNMENT_INPUTOUTPUT_H

namespace observations {
/** Represents an observation of a (possibly batched) input/output pair of a function.
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
    template<class INPUT, class OUTPUT>
    class InputOutput {
    public:
        INPUT input;
        OUTPUT output;


    };
};


#endif //MULTIAGENTGOVERNMENT_INPUTOUTPUT_H
