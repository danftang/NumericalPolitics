//
// Created by daniel on 13/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_ADAPTIVEFUNCTION_H
#define MULTIAGENTGOVERNMENT_ADAPTIVEFUNCTION_H

#include "../../DeselbyStd/typeutils.h"
#include "Concepts.h"
#include "../CallbackUtils.h"

namespace abm::events {
    template<class PARAM>
    struct ParameterUpdate {
        const PARAM &parameters;

        ParameterUpdate(const PARAM &params): parameters(params) {}
    };
}

namespace abm::approximators {

    /** A function that intercepts events and adapts to those events.
     * An instance of this class is the joining together of three objects:
     *   - A parameterised object that can update its own parameters given a LossFunction
     *   - A LossFunction describing a loss in the space of all functions
     *   - A training policy that defines when parameter updates should occur.
     */
    template<LossFunction LOSSFUNCTION, ParameterisedFunction APPROXIMATOR, class TRAININGPOLICY>
    class AdaptiveFunction : public APPROXIMATOR {
    public:
        LOSSFUNCTION    lossFunction;
        TRAININGPOLICY  trainingPolicy;


        AdaptiveFunction(APPROXIMATOR approximator, TRAININGPOLICY trainingPolicy, LOSSFUNCTION lossFunction):
                APPROXIMATOR(std::move(approximator)),
                lossFunction(std::move(lossFunction)),
                trainingPolicy(std::move(trainingPolicy))
        { }


        template<class EVENT>
        void on(const EVENT &event) {
            callback(event, lossFunction);
            bool didTrain = trainingPolicy.train(event, *this, lossFunction);
            if(didTrain) callback(events::ParameterUpdate(this->parameters()), lossFunction);
        }
    };


    /** A training policy decides exactly how and when to update the parameters of a parameterised
     * function given a loss function.
     * This class takes an ensmallen update step and a schedule, which should be a function object
     * which takes events and returns a bool if an update step should be made in response to the
     * event.
     */
    template<class UPDATESTEP, class SCHEDULE, class MatType = arma::mat>
    class DifferentialTrainingPolicy {
    public:
        UPDATESTEP  update;
        typename UPDATESTEP::template Policy<MatType,MatType> updatePolicy; // requires gradient to be same type as param type
        double      stepSize;
        SCHEDULE    schedule;


        DifferentialTrainingPolicy(UPDATESTEP update, double stepSize, size_t parameterRows, size_t parameterCols, SCHEDULE schedule) :
        update(std::move(update)),
        updatePolicy(this->update, parameterRows, parameterCols),
        stepSize(stepSize),
        schedule(std::move(schedule)) {
        }


        template<class EVENT, LossFunction LOSSFUNCTION, DifferentiableParameterisedFunction<LOSSFUNCTION> APPROXIMATOR>
        inline bool train(const EVENT &event, APPROXIMATOR &&approximator, LOSSFUNCTION &&lossFunction) {
            bool doTrain = deselby::invoke_or(schedule, false, event);
            if (doTrain) updatePolicy.Update(approximator.parameters(), stepSize, approximator.gradientByParams(lossFunction));
            return  doTrain;
        }
    };

    /** A parameterised function and an update step, giving rise to an OptimisableFunction that can update
     * its parameters given a LossFunction */
//    template<class UPDATESTEP, class APPROXIMATOR>
//    class DifferentiableOptimisableFunction: public APPROXIMATOR {
//    public:
//        typedef std::remove_cvref_t<decltype(std::declval<APPROXIMATOR>().parameters())> MatType;
//
//        UPDATESTEP                  updateStep;
//        typename UPDATESTEP::template Policy<MatType,MatType> updatePolicy; // requires gradient to be same type as param type
//        double                      stepSize;
//
//        DifferentiableOptimisableFunction(APPROXIMATOR approximator, UPDATESTEP updatestep, double stepSize):
//        APPROXIMATOR(std::move(approximator)),
//        updateStep(std::move(updatestep)),
//        updatePolicy(updateStep, this->parameters().n_rows, this->parameters().n_cols),
//        stepSize(stepSize) {
//        }
//
//        template<LossFunction LOSS> requires DifferentiableParameterisedFunction<APPROXIMATOR,LOSS>
//        void updateParameters(LOSS &loss) {
//            updatePolicy.Update(this->parameters(), stepSize, this->gradientByParams(loss));
//        }
//    };

}

#endif //MULTIAGENTGOVERNMENT_ADAPTIVEFUNCTION_H
