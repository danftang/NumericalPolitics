//
// Created by daniel on 25/11/23.
//

#ifndef MULTIAGENTGOVERNMENT_IIMCTSLOSSES_H
#define MULTIAGENTGOVERNMENT_IIMCTSLOSSES_H

#include <map>
#include <cstdlib>
#include <armadillo>
#include "../minds/qLearning/SoftMaxPolicy.h"
#include "../minds/qLearning/QVector.h"
#include "WeightedLoss.h"
#include "SumOfLosses.h"

namespace abm::minds::IIMCTS {
    template<size_t Qsize> class QEntry;
};

namespace abm::events {
    template<class BODY>
    struct IncomingMessageObservation {
        const std::map<BODY,uint> &bodySamples;
        BODY::message_type message;
    };

    template<class BODY>
    struct QVectorObservation {
        const BODY &body;
        const minds::QVector<BODY::action_type::size> &qVector;
    };
}

namespace abm::lossFunctions {
    /** Loss of a Q-function given an observed incoming message, a PMF over body states and a policy.
     * This allows the agent to model/copy other.
     *
     * Given a Q-policy and a (S,w,m) triplet where S is a set of body-states, w(b) is a weight (unnormalised prob)
     * associated with body state b, and m is an incoming message,
     * we define the loss of a Q-function Q(b) as minus the log-posterior probability of making the observation, where:
     *
     * P(message | S, w, Policy) = A\sum_{b \in S} w(b) \sum_a P(Policy(Q(b)) == a)P(b.handleAct(a) == message)
     * where
     * A = 1/\sum_{b' \in S}w(b')
     * we assume that P(b.handleAct(a) == message) is non-zero for only one action a, i.e. given a message, we can
     * identify a unique action that caused that message a = b.handleAct^{-1}(message). If this isn't the case, then
     * we define b.handleAct^{-1}(message) to be a stochastic function from which we draw an action. So, we can consider
     * the probability
     *
     * P(a | S, w, Policy) = A\sum_{b \in S} w(b) P(Policy(Q(b)) == a)
     * so
     *
     * d(-log(P(a | S,w,Policy)))/dQ_a(b) =  -1/(P(a | S,w,Policy)) dP(a | S,w,Policy)/dQ_a(b)
     * where Q_a(b) is the a'th element of Q(b) and
     *
     * dP(a | S,w,Policy)/dQ_a(b) = A.w(b) dP(Policy(Q(b))==a)/dQ_a(b)
     * So, substituting and cancelling the A's:
     *
     * d(-log(P(a | S,w,Policy)))/dQ_a(b) =  -w(b) dP(Policy(Q(b))==a)/dQ_a(b)
     *                                      /\sum_{b' \in S} w(b') P(Policy(Q(b')) == a)
     *
     * For a batch of messages, the total rate of change is the sum of the rates of the individual messages. We can
     * treat each set S as separate.
     *
     * In the implementation we have a batch of PMFs (S_i,w_i), from which we form a matrix of training points by
     * concatenating the members of each S_i. The start of each S_i is signalled by a non-zero in isPMFStart.
     *
     *
     */
    template<class BODY, class POLICY = minds::SoftMaxPolicy>
    class MessageLoss {
    public:
        typedef decltype(std::declval<BODY>().legalActs()) mask_type;

        struct Observation {
            std::vector<std::pair<BODY,uint>> pmf;
            BODY::message_type message;

            Observation(const events::IncomingMessageObservation<BODY> &event) : message(event.message) {
                pmf.reserve(event.bodySamples.size());
                for(const auto &pmfEntry : event.bodySamples) pmf.push_back(pmfEntry);
            }
        };

        std::vector<Observation>  observations;
        std::vector<uint>   batchIndices;
        POLICY policy;              // differentiable policy from which to get gradient of dP(Q)/dQ

        uint insertCol = 0;
        uint bufferCapacity;
        uint nTrainingPoints = 0;

        /** N.B. bufferCapacity is measured in number of training points, not in number of incoming messages.
         * Each incoming message adds a set of training points, one for each possible state of other.*/
        MessageLoss(size_t bufferCapacity, size_t batchObservations, POLICY policy = minds::SoftMaxPolicy()) :
                bufferCapacity(bufferCapacity),
                batchIndices(batchObservations,0),
                policy(std::move(policy)) {
            observations.reserve(bufferCapacity);
        }


        void on(const events::IncomingMessageObservation<BODY> &observation) {
            std::cout << "Intercepting IncomingMessageObservation" << std::endl;
            if(insertCol < observations.size()) {
                observations[insertCol] = observation;
            } else {
                observations.emplace_back(observation);
                // maintain batchIndices as random sample so we can calculate batchSize at any time
                if(observations.size() == 1) {
                    nTrainingPoints = observation.bodySamples.size() * batchIndices.size();
                } else {
                    uint i = deselby::random::geometric(1.0 / observations.size());;
                    while (i < batchIndices.size()) {
                        nTrainingPoints += observation.bodySamples.size() - observations[batchIndices[i]].pmf.size();
                        batchIndices[i] = observations.size() - 1;
                        i += 1 + deselby::random::geometric(1.0 / observations.size());
                    }
                }
            }
            insertCol = (insertCol + 1)%capacity();
        }

        size_t capacity() const { return bufferCapacity; } // number of observations
        size_t bufferSize() const { return observations.size(); } // number of observations
        size_t batchObservations() const { return batchIndices.size(); } // number of observations
        size_t batchSize() const { return nTrainingPoints; }

        /** TODO: Sort this out: need to return a batch of sets rather than individual states.
         *
         * @tparam INPUTS
         * @param trainingMat
         */
        template<class INPUTS>
        void trainingSet(INPUTS &trainingMat) {
            assert(bufferSize() > 0);
            int col = 0;
            for(const uint &i : batchIndices) {
                for(const auto &[body, weight] : observations[i].pmf) {
                    trainingMat.col(col++) = static_cast<const arma::mat &>(body);
                }
            };
        }

        /** If we assume a given action a...
         * d(-log(P(a | S,w,Policy)))/dQ_i(b_j) =  -1/(P(a | S,w,Policy)) dP(a | S,w,Policy)/dQ_i(b_j)
         * where Q_i(b_j) is the i'th element of Q(b_j) and b_j is the j'th body state
         *
         * dP(a | S,w,Policy)/dQ_i(b_j) = A.w(b_j) dP(Policy(Q(b_j))==a)/dQ_i(b_j)
         * and
         *
         * P(a | S, w, Policy) = A\sum_{b' \in S} w(b') P(Policy(Q(b')) == a)
         *
         * so, for each column b_j, for which we now have a Q(b_j)
         * d(-log(P(a | S,w,Policy)))/dQ(b_j) = (w(b_j) / \sum_{b' \in S} w(b') P(Policy(Q(b')) == a)) dP(Policy(Q(b_j))==a)/dQ(b_j)
         *      = (1/gamma(S)) w(b_j) dP(Policy(Q(b_j))==a)/dQ(b_j)
         * where
         * gamma(S) =  \sum_{b' \in S} w(b') P(Policy(Q(b')) == a)
         *
         * d(-log(P(message | S,w,Policy)))/dQ(b) =  -w(b) dP(Policy(Q(b)) == b.messageToAct(message))/dQ(b)
         *                                      /\sum_{b' \in S} w(b') P(Policy(Q(b')) == b.messageToAct(message))
         *
         * @tparam OUTPUTS
         * @tparam RESULT
         * @param qVectors
         * @param result
         */
        template<class OUTPUTS, class RESULT>
        void gradientByPrediction(const OUTPUTS &qVectors, RESULT &result) {
            result.zeros();

            uint col = 0;
            for(const uint &i : batchIndices) {
                auto &message = observations[i].message;
                double gamma = 0.0;
                uint startCol = col;
                for(const auto &[body, weight] : observations[i].pmf) {
                    auto act = body.messageToAct(message);
                    result.col(col) = -weight * policy.gradient(qVectors.col(col), body.legalActs(), act);
                    gamma += weight * policy.probability(qVectors.col(col), body.legalActs(), act);
                    ++col;
                }
                while(startCol != col) {
                    result.col(startCol) /= gamma;
                    ++startCol;
                }
            }
            // resample batchIndices for next time
            for(uint &i : batchIndices) {
                nTrainingPoints -= observations[i].pmf.size();
                i = deselby::random::uniform(observations.size()); // choose new batch for next time
                nTrainingPoints += observations[i].pmf.size();
            }
        }

    };


    /** loss between a real vector of doubles and a QVector of QValues (i.e. (sample sum, sample count) pairs)
     * For eqch (q,Q)-value pair, we define as the loss as minus the log-prob that the sum of samples in Q came from a
     * Gaussian distribution with mean q
     * So,
     * P(q-\bar{Q}) = (sqrt(n/2pi.v)) e^{-(n/2v)(q-\bar{Q})^2}
     * where v is the variance of the samples, so
     * logP(q-\bar{Q}) = 0.5ln(n/v) - (n/2v)(q-\bar{Q})^2  - 0.5ln(2pi)
     * so
     * d(-logP)/dq = (n/v)(q-\bar{Q})
     *
     * since we have no information about the actual variance of the samples (we only know the sample count and
     * sample sum), we assume that the samples in Q always have the same variance, and choose a variance to
     * make the gradient around unity.
     *
     */
    template<class BODY>
    class QEntryLoss {
    public:
        static constexpr double sampleVariance = 100.0;

        arma::mat   trainingPoints;  // by-column list of training points
        std::vector<const minds::QVector<BODY::action_type::size> *> qVectorPtrs; // TODO: this may outlive the treeNode!
        size_t      insertCol = 0;

        QEntryLoss(size_t bufferSize) : trainingPoints(BODY::dimension, bufferSize) {
            qVectorPtrs.reserve(bufferSize);
        }

        void on(const events::QVectorObservation<BODY> &observation) {
//            std::cout << "Intercepting QEntryObservation" << std::endl;
            trainingPoints.col(insertCol) = static_cast<const arma::mat &>(observation.body);
            const minds::QVector<BODY::action_type::size> *qVecPtr = &(observation.qVector);
            if(insertCol == qVectorPtrs.size()) {
                qVectorPtrs.push_back(qVecPtr);
            } else {
                qVectorPtrs[insertCol] = qVecPtr;
            }
            insertCol = (insertCol + 1)%trainingPoints.n_cols;
        }

        template<class INPUTS>
        void trainingSet(INPUTS &trainingMat) {
            if(bufferIsFull()) {
                trainingMat = trainingPoints;
            } else {
                trainingMat = trainingPoints.cols(0, insertCol-1);
            }
        }

        bool   bufferIsFull() const { return qVectorPtrs.size() == trainingPoints.n_cols; }
        size_t capacity() const { return trainingPoints.n_cols; }
        size_t bufferSize() const { return bufferIsFull()?capacity():insertCol; }
        size_t batchSize() const { return bufferSize(); }

        template<class OUTPUTS, class RESULT>
        void gradientByPrediction(const OUTPUTS &qMeans, RESULT &gradient) {
            size_t s = batchSize();
            gradient.zeros();
            for(size_t col = 0; col < s; ++col) {
                const minds::QVector<BODY::action_type::size> &qVec = *qVectorPtrs[col];
                for(size_t action = 0; action < qMeans.n_rows; ++action) {
                    gradient(action,col) =
                            qVec[action].sampleCount * (qMeans(action,col) - qVec[action].mean())
                            / sampleVariance;
                }
            }
        }
    };

    /** Loss function for training the off-tree qFunction. This is a weighted sum of the MessageLoss
     * from experience of other and the IOLoss compared to the Tree predictions.
     * Intercepts (BODY,Q-vector) pairs and (BODY state PMF, message) pairs to make a loss function
     * for a parameterised approximator.
     *
     * The loss is given by:
     *
     * L = IOLoss + w*MessageLoss
     * where w is a weight
     *
     * TODO: We could view the tree itself as a buffer of (Body,QVector) pairs, with perhaps sample count
     *      as importance, from which we can sample a batch on which to train. The trajectory of self-training
     *      minds provides a sample mechanism for free.
     */
    template<class BODY, class QPOLICY = minds::SoftMaxPolicy>
    class OffTreeLoss : public SumOfLosses<QEntryLoss<BODY>, WeightedLoss<MessageLoss<BODY,QPOLICY>>> {
    public:
        typedef SumOfLosses<QEntryLoss<BODY>, WeightedLoss<MessageLoss<BODY,QPOLICY>>> base_type;

        static constexpr size_t defaultQEntryBufferSize = 16;
        static constexpr size_t defaultMessageBatchSize = 16;
        static constexpr size_t defaultMessageBufferSize = 512;
        static constexpr double selfOtherLearningRatio = 2.0;

        OffTreeLoss(size_t qEntryBufferSize = defaultQEntryBufferSize,
                    size_t messageBufferSize = defaultMessageBufferSize,
                    size_t messageBatchSize = defaultMessageBatchSize,
                    QPOLICY qPolicy = minds::SoftMaxPolicy()):
                base_type(
                        QEntryLoss<BODY>(qEntryBufferSize),
                        WeightedLoss(
                                selfOtherLearningRatio,
                                MessageLoss<BODY,QPOLICY>(messageBufferSize, messageBatchSize, qPolicy))) {
        }

    };


}


#endif //MULTIAGENTGOVERNMENT_IIMCTSLOSSES_H
