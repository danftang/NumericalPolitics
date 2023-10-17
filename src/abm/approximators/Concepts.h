//
// Created by daniel on 06/09/23.
//

#ifndef MULTIAGENTGOVERNMENT_CONCEPTS_H
#define MULTIAGENTGOVERNMENT_CONCEPTS_H

#include "mlpack.hpp"

/** An approximator is any computable function that is used as an approximation of some
 * intractable or partially-known target function.
 *
 * A parameterised approximator is a family of functions, each function identified by a point
 * in the parameter space. A parameterised approximator implements a parameters() method, that identifies the
 * current set of parameters.
 *
 * Given some information about the target function, we can "train" the parameters of a parameterised approximator
 * to minimise (or at least improve) some (perhaps stochastic) objective function. A "training strategy"
 * defines a set of types of evidence that can be used to update the approximators parameters, along with the
 * code to actually do the parameter update. Eveidence is presented to an object by calling the train(.) mothod
 * with an object that represents some class of evidence.
 *
 * A training strategy can often be split into:
 *   - An (maybe differentiable) objective function from parameter space to the set of reals, given the approximator.
 *   - An optimisation algorithm/step that updates the parameters to improve the objective
 *   - A strategy on when to run the optimisation steps
 * The objective function can itself often be split into
 *   - a function from evidence to a (samplable) vector of input/output pairs
 *   - a (maybe differentiable) function from (actual output, required output) to real (loss)
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
namespace abm::approximators {
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

//    template<class T, class OBSERVATION>
//    concept Trainable1 = requires(T f, OBSERVATION observation) {
//        { f.train(observation) };
//    };
//
//    template<class T, class... OBSERVATION>
//    concept Trainable = (Trainable1<T,OBSERVATION> && ...);
//
//    template<class T, class IOSIGNATURE, class... OBSERVATIONTYPES>
//    concept TrainableApproximator = Approximator<T,IOSIGNATURE> && Trainable<T,OBSERVATIONTYPES...>;

    /** With respect to a function Y=F(X) and a set of input points (X_1...X_n)
     * a LossFunction represents a function from outputs (Y_1...Y_n) where Y_i = F(X_i)
     * to a real. In particular, the gradient of this funciton.
     *
     * The loss function may represent a sum of log-probs.
     *
     */
    template<class T> // ensmallen objective?
    concept LossFunction = requires(T obj, arma::mat &result, arma::mat predictions) {
        obj.nPoints();  // number of points in the training set
        obj.trainingSet(result); // put (X_1...X_n) in vec [can C++ optimise the moving of an Array to a memory address?]
        obj.gradientByPrediction(predictions, result); // ...and input identifiers?
//        obj.loss(functionOutputs, result); //
    };

//    template<class T> // ensmallen objective?
//    concept ViewLossFunction = requires(T obj, arma::mat &result, arma::mat functionOutputs) {
//        obj.nPoints();  // number of intout/output points
//        { obj.inputs() } -> std::ranges::view; // returns a lazy evaluated range of input vectors
//        {obj.gradientByOutputs(functionOutputs) } -> std::ranges::view; // returns a lazy evaluated range of gradients given the function outputs
////        obj.loss(functionOutputs, result); //
//    };


//    template<class T> // ensmallen objective?
//    concept LossGradient = requires(T obj, arma::mat &result, arma::mat functionOutputs) {
//        { obj.inputs(result) } -> std::same_as<void>; // put (X_1...X_n) in vec [can C++ optimise the moving of an Array to a memory address?]
//        { obj.operator()(functionOutputs, result) } -> std::same_as<void>;
//    };
//
//    template<class T, class RESULTTYPE> // ensmallen objective?
//    concept ComposableLossGradient = LossGradient<T> && requires(T obj, arma::mat &result, arma::mat functionOutputs) {
//        obj.nPoints();  // number of intout/output points
//        obj.inputs(result); // put (X_1...X_n) in vec [can C++ optimise the moving of an Array to a memory address?]
//        obj.exportGradient(functionOutputs, result); // ...and input identifiers?
//    };


    /** */
//    template<class T>
//    concept SumOfLossFunctions = requires(T obj, size_t n) {
//        obj.size();   // number of terms in the sum that this function represents
//        obj.batchLoss(n); // generate a loss function consisting of the sum of n terms
////        obj.on(event); // insert information into the buffer
//    };

    ////////  A loss function is a set of inputs and an Output layer,
    /// a stochastic loss function is a generator of loss functions (or a loss function with a sample() member)

    template<class T>
    concept StochasticLossFunction = requires(T obj) {
        { obj.getNextLossFunction() } -> LossFunction;
        obj.setSampleRatio();   // sets the size of a batch as a fraction of the buffer size. (or set batch size?)
    };

    /** */
    template<class T, class LOSSFUNCTION>
    concept ParameterisedFunction = requires(T obj, LOSSFUNCTION loss) {
        { obj.parameters() } -> std::same_as<arma::mat &>;
        { obj.gradientByParams(loss) } -> std::same_as<arma::mat>;
    };

    template<class T, class LOSSFUNCTION>
    concept OptimisableFunction = requires(T obj, LOSSFUNCTION loss) {
        obj.updateParameters(loss);
    };


//    /** */
//    template<class T, class PARAMETERISEDFUNC>
//    concept TrainingStrategy = requires(T obj, PARAMETERISEDFUNC parameterisedFunction) {
//        obj.doTrainingStep(parameterisedFunction);   // number of terms in the sum that this function represents
////        obj.on(event); // insert information into the buffer
//    };

//    /** For use with a trainable function */
    template<class T, class EVENT>
    concept TrainingPolicy = requires(T obj, const EVENT &event) {
        { obj(event) } -> std::same_as<bool>; // true if we should execute a training step
    };
//
//    template<class T>
//    concept OptimisationStep = requires(T obj) {
//        obj.update(Parameters, parameterGradient); // true if we should execute a training step
//    };
//



}

#endif //MULTIAGENTGOVERNMENT_CONCEPTS_H
