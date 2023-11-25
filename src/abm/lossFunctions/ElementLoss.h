//
// Created by daniel on 15/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_ELEMENTLOSS_H
#define MULTIAGENTGOVERNMENT_ELEMENTLOSS_H

#include <vector>
#include <armadillo>

/** Represents a set of (intput,output-element) pairs for a function whose output is a vector, for use when we
 * wish to constrain only one element of the vector output
 * TODO: this should be an output layer (pure loss function)
 * */
template<class INPUTS, class MatType>
class ElementLoss {
public:
    arma::sp_mat elements;      // the elements we have observations of, corresponding to inputs

    ElementLoss() { }

    template<class OUT, class RESULTS>
    void gradientByOutputs(const OUT &outputs, RESULTS &&results) {
        results.zeros();
        auto elemIndices = arma::find(elements);
        results.elem(elemIndices) = outputs.elem(elemIndices) - arma::nonzeros(elements);

//        results = (arma::abs(arma::sign(elements)) % outputs) - elements;
    }

    template<class OUT, class ELEMENTS, class RESULTS>
    static void gradient(const OUT &outputs, ELEMENTS elements, RESULTS &&results) {
        results.zeros();
        auto elemIndices = arma::find(elements);
        results.elem(elemIndices) = outputs.elem(elemIndices) - arma::nonzeros(elements);
    }

};


#endif //MULTIAGENTGOVERNMENT_ELEMENTLOSS_H
