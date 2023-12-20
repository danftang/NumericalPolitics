//
// Created by daniel on 20/12/23.
//

#ifndef MULTIAGENTGOVERNMENT_UPDATESTEP_H
#define MULTIAGENTGOVERNMENT_UPDATESTEP_H

#include <ensmallen_bits/adam/adam.hpp>
#include <armadillo>

namespace abm::approximators {
    template<class ENSMALLENUPDATE, class MatType = arma::mat, class GradType = arma::mat>
    class UpdateStep {
    public:
        size_t nRows;
        size_t nCols;
        ENSMALLENUPDATE updateParams;
        typename ENSMALLENUPDATE::template Policy<MatType,GradType> policy;

        UpdateStep(ENSMALLENUPDATE &&ensmallenupdate, const size_t gradRows, const size_t gradCols) :
            nRows(gradRows),
            nCols(gradCols),
            updateParams(std::move(ensmallenupdate)),
            policy(this->updateParams, gradRows, gradCols) {}

        UpdateStep(const UpdateStep<ENSMALLENUPDATE> &other) :
                nRows(other.nRows),
                nCols(other.nCols),
                updateParams(other.updateParams),
                policy(this->updateParams, other.nRows, other.nCols) {}

        UpdateStep(UpdateStep<ENSMALLENUPDATE> &&other) :
                nRows(other.nRows),
                nCols(other.nCols),
                updateParams(std::move(other.updateParams)),
                policy(this->updateParams, other.nRows, other.nCols) {
        }

        void Update(MatType& iterate, const double stepSize, const GradType& gradient) {
            policy.Update(iterate, stepSize, gradient);
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_UPDATESTEP_H
