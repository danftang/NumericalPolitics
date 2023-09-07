//
// Created by daniel on 06/09/23.
//

#ifndef MULTIAGENTGOVERNMENT_CONCEPTS_H
#define MULTIAGENTGOVERNMENT_CONCEPTS_H

#include "mlpack.hpp"

/** An approximator is any computable function that is used as an approximation of some
 * intractable or partially-known target function.
 *
 * A approximator class will often represent a family of functions.
 * An instance of a parameterised class contains a set of parameters, identified by a parameters()
 * method, that identifies a single function from the family.
 * Given some information about the target function, we can "train" the parameters to minimise
 * (or at least improve) some (perhaps stochastic) objective function. So, a "training strategy"
 * involves the collection of evidence on the one hand, and the update of the parameters of an
 * approximator on the other, all with respect to some objective function. This splits up further
 * into objective function (from parameters to real and gradient, with respect to the evidence collected so far)
 * and optimiser (from current parameters and objective function to new parameters).
 *
 * In full generality, evidence can take any form but should be supplied to the training strategy as
 * callbacks as the evidence becomes available. These ultimately will be called from the callbacks
 * of the Mind that the Agent calls. Standardised overloads of the train(...) method can be used to
 * distinguish between different types of evidence.
 *
 * So, we have a heirarchy of class concepts
 *   1) a class implementing Predict(), possibly batched
 *   2) a parameterised family of approximator functions, implementing Parameters()
 *   3) a parameterised approximator with an (possibly separable, possibly differentiable) objective loss function
 *   4) a parameterised approsimator with an objective loss function and an observe(evidence) function for some
 *       class of evidence. This should update the objective function to account for the new evidence, but does
 *       not change the parameters.
 *   5) a parameterised approximator with a train(evidence) function for some class of evidence. This should
 *       update the parameters in response to the new evidence.
 *
 *  To go from 2 to 3, we supply an objective function to a parameterised approximator
 *  To go from 3 to 4 we supply an observation function. If the objective is expressed as a probability distribution over
 *  target functions then the observation function is is a likelihood function of evidence given a target function
 *  and possibly a prior over evidence.
 *  To go from 4 to 5 we require a training strategy which requires an optimisation algorithm given the type of objective
 *  (separable, differentiable etc) and a procedure for execution of the optimisation algorithm (i.e. when to execute,
 *  hyper-parameters).
 *
 *  What we're trying to do is find the parameters, p, that minimise the expected error in an approximation given a set
 *  of evidence, e:
 *
 *  min_p(E_{P(x),P(T|e)}(|f_p(x) - T(x)|)) = min_p \int \int P(x)P(T|e) |f_p(x)-T(x)| dx dT
 *
 * where |.| is some metric on the range of f and P(x) is a prior distribution over the domain.
 *
 * Often, this optimisation will be done by taking batches of samples from <x,T>-space and using stochastic optimisation
 * so the observation function consists of generating a batch of samples <x,y> with probability
 * P(x,y)dx dy = \int_x^x+dx \int _y^y+dy \int delta(y'=T(x'))P(T|e) dT dx' dy'
 *
 * The evidence will also often appear incrementally, which can be processed with the train(evidence) method
 * which should update the parameters in response to the evidence.
 *
 *
 */
namespace approximators {
    // ================  OBJECTIVE CONCEPTS  =====================
    // =================================================
    template<class T>
    concept EvaluableWithGradient = requires(T f, const arma::mat &in, arma::mat &gradient) {
        { f.EvaluateWithGradient(in, gradient) } -> std::convertible_to<double>;
    };

    template<class T>
    concept Evaluable = EvaluableWithGradient<T> || requires(T f, const arma::mat &in) {
        { f.Evaluate(in) } -> std::convertible_to<double>;
    };

    template<class T>
    concept Differentiable = EvaluableWithGradient<T> || requires(T f, const arma::mat &in, arma::mat &gradient) {
        { f.Gradient(in, gradient) } -> std::same_as<void>;
    };

    template<class T>
    concept Separable = requires(T f, const arma::mat &in, const size_t index, const size_t batchSize) {
        { f.Shuffle() } -> std::same_as<void>;
        { f.NumFunctions()} -> std::convertible_to<size_t>;
    };

    template<class T>
    concept SeparableEvaluableWithGradient  =
    requires(T f, const arma::mat &in, arma::mat &gradient, const size_t index, const size_t batchSize) {
        { f.EvaluateWithGradient(in, gradient, index, batchSize) } -> std::convertible_to<double>;
    };

    template<class T>
    concept SeparableEvaluable =
            SeparableEvaluableWithGradient<T> ||
            requires(T f, const arma::mat &in, const size_t index, const size_t batchSize) {
                { f.Evaluate(in, index, batchSize) } -> std::convertible_to<double>;
            };

    template<class T>
    concept SeparableDifferentiable =
            SeparableEvaluableWithGradient<T> ||
            requires(T f, const arma::mat &in, arma::mat &gradient, const size_t index, const size_t batchSize) {
                { f.Gradient(in, gradient, index, batchSize) } -> std::same_as<void>;
            };

    template<class T, bool MUSTBEDIFFERENTIABLE>
    concept EnsmallenUnSeperableObjective =
        Evaluable<T> &&
        (!MUSTBEDIFFERENTIABLE || Differentiable<T>);

    template<class T, bool MUSTBEDIFFERENTIABLE>
    concept EnsmallenSeperableObjective =
        Separable<T> &&
        SeparableEvaluable<T> &&
        (!MUSTBEDIFFERENTIABLE || SeparableDifferentiable<T>);

    template<class T, bool ISSEPARABLE, bool MUSTBEDIFFERENTIABLE>
    concept EnsmallenObjective =
        (ISSEPARABLE && EnsmallenSeperableObjective<T,MUSTBEDIFFERENTIABLE>) ||
        (!ISSEPARABLE && EnsmallenUnSeperableObjective<T,MUSTBEDIFFERENTIABLE>);


    // ============ TRAINABLE FUNCTION CONCEPTS ==================


    template<class T, class IOSIGNATURE>
    concept Approximator = std::convertible_to<T,std::function<IOSIGNATURE>>;

    template<class T, class IOSIGNATURE>
    concept ParameterisedApproximator = Approximator<T,IOSIGNATURE> && requires(T f) {
        { f.Parameters() };
    };

    template<class T, bool ISSEPERABLE, bool MUSTBEDIFFERENTIABLE>
    concept ObjectiveApproximator =
        ParameterisedApproximator<T,arma::mat> &&
        EnsmallenObjective<T,ISSEPERABLE,MUSTBEDIFFERENTIABLE>;

//    template<class T, class EVIDENCE>
//    concept CanObserve = requires(T f, EVIDENCE evidence) {
//        { f.observe(evidence) };
//    };

    template<class T, class OBSERVATION>
    concept Trainable1 = requires(T f, OBSERVATION observation) {
        { f.train(observation) };
    };

    template<class T, class... OBSERVATION>
    concept Trainable = (Trainable1<T,OBSERVATION> && ...);

    template<class T, class IOSIGNATURE, class... OBSERVATIONTYPES>
    concept TrainableApproximator = Approximator<T,IOSIGNATURE> && Trainable<T,OBSERVATIONTYPES...>;

}

#endif //MULTIAGENTGOVERNMENT_CONCEPTS_H
