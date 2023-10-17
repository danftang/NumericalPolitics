//
// Created by daniel on 09/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_OBSERVATION_H
#define MULTIAGENTGOVERNMENT_OBSERVATION_H


namespace approximators {
    /** An Obeservation(loss) provides some information about an approximating function F
     * at a given input point X. It provides a loss, given Y=F(X), and gradient dLoss/dY
     * both as a funcion of the output, Y.
     *
     * When training, we'll have a number of observations possibly of different types and we'll
     * want to sample a batch from the observations and find the gradient wrt these observations.
     * This involves forming a vector of inputs and a function from vector of outputs to gradient
     * of objective w.r.t. output for each observation.
     *
     * So, what we end up with is a matrix transform where each row(column) is calculated separately
     * by a different object type. So, we can have a collection of observations, when we add an observation
     * we add an input vector and a gradient function. We can then choose a batch and train.
     *
     * @tparam INPUT
     * @tparam OUTPUT
     * @tparam MATTYPE
     */
    template<class INPUT, class OUTPUT, class GRADTYPE>
    class Observation {
    public:
        INPUT &     X() = 0;// function input to which this observation is concerned
        GRADTYPE    gradient(OUTPUT &Y) = 0; // Gradient of objective with respect to Y at a given Y=F(X).
        // double      loss(OUTPUT &Y) = 0;         // value of the objective at a given Y=F(X)
    };
}

#endif //MULTIAGENTGOVERNMENT_OBSERVATION_H
