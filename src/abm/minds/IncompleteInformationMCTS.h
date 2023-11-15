// Implementation of Monte Carlo Tree Search for two agents that have
// incomplete information about the other's internal state, but
// each agent knows the reward function of both agents and the
// state transitions for
//
// At any point we only care about other's state to the extent that it
// affects other's reward structure...so state is an input to reward.
// Reward of any agent is a function of
// (state-before-last-decision, channel/message, state-before-next-decision)
//
// We assume that both agents share a prior distribution over their own and other's state
// and that an agent's belief about the other's state is simply the conditional distribution
// given the acts observed so far [esp, this doesn't depend on the believer's state].
//
// BODY must provide:
//  - state transition function: newState(oldState, outMessage, inMessage)
//  - reward function: reward(oldState, outMessage, newState)
// AGENTSTATEDISTRIBUTION must provide:
//  - bayesianUpdate(likelihoodFunction)
//  - map(stateTransitionFunction) which maps the distribution to a different domain
//
// NB: If an agent's policy is completely random. it tells us nothing about its state
// but as the Q values change, so too do the state distributions and so the weight of the
// samples. However, MCTS takes the current Q values for the next sample, glossing over this
// change (should we exponentially weight down?).
//
// A change in policy at the ancestors of a tree node affects the posterior there through
// changes in the likelihood function, so we need to, at least periodically, update the
// posteriors [this is bad because a change in policy at the root node changes all likelihoods
// ].
//
// Perhaps agents should fixate on the opponent hidden states that maximise probability times reward
// for each action, as these are the greatest contributors to expected reward...
//
// Or we choose the playout that will give us the most information about the best next action given
// the current state of the tree. We can calculate this by considering the playout as drawn from the
// distribution of a fictitious pair of agents, and then use importance sampling for all states on the
// same path of public information.
//
// Or play out on categories of states, grouped by posterior probability.
//
// If we could sample the trajectories with a known probability (approximately) proportional to
// the cumulative reward at the root times the current probability of the trajectory, then
// we could approximate Q values at the root very efficiently.
//
// TODO: dealing with multiple equilibria (language learning). If we use a Q-approximator for
//  off-tree decisions (and perhaps bias new tree nodes towards the Q-approximation?) then the Q-approximator
//  defines which equilibrium we end up in (as well as potentially making the algorithm more efficient).
//  If the Q-approximator is from body state to Q-vector, then there is flexibility on how much of the
//  episode history is included in the body state.
//  The Q-approximation is not changed during self-play. When I make a real move, my Q-approximation
//  learns from the Q-values of the root of the tree on my current body state [also the other body states in the root-node give
//  us information, should we use these?]. On your move, I observe your actual move but don't know your body-state.
//  However, I have samples from your body distribution given past moves in the episode.
//  If there are one or more body states on the tree-root that would have chosen the observed move, choose one
//  at random and learn from its Q-values. If no body states would choose the observed move, then choose the
//  one that requires the minimal zero-sum perturbation to its Q-values to make the observed action, and learn from
//  the minimally perturbed Q-values [or choose weighted by the size of the perturbation? or just don't learn?].
//  [N.B. we should properly use the hindcast body-states at the end of the episode]
//  No batching? (Adam update)
//  N.B. The test of the whole algorithm is whether we end up with a society that ends up converging to social norms.
//  (e.g. can it converge to language use by convention?)
//
// Created by daniel on 09/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_INCOMPLETEINFORMATIONMCTS_H
#define MULTIAGENTGOVERNMENT_INCOMPLETEINFORMATIONMCTS_H

#include <vector>
#include <map>
#include <functional>
#include <cassert>
#include <algorithm>
#include <random>
#include <bitset>
#include <ranges>
#include <boost/circular_buffer.hpp>
#include <armadillo>

#include "../Agent.h"
#include "../societies/RandomEncounterSociety.h"
#include "../minds/qLearning/QVector.h"
#include "../minds/qLearning/GreedyPolicy.h"
#include "../minds/qLearning/UpperConfidencePolicy.h"
#include "ZeroIntelligence.h"
#include "../../DeselbyStd/stlstream.h"
#include "../episodes/SimpleEpisode.h"
#include "../lossFunctions/SumOfLosses.h"
#include "../lossFunctions/WeightedLoss.h"
#include "QMind.h"
#include "../lossFunctions/IOLoss.h"
#include "../minds/qLearning/SoftMaxPolicy.h"
#include "../../DeselbyStd/DiscreteObjectDistribution.h"

namespace abm::events {
    template<class BODY>
    struct IncomingMessageObservation {
        std::map<BODY,uint> &bodySamples;
        BODY::message_type message;
    };
}

namespace abm::minds {

    namespace IIMCTS {

        /** Loss function for incoming messages given a Q-function, a policy, a PMF over body states and an observed
         * message.
         * Given a (BODY state PMF, message) pair and a Q-policy, we define the loss of a Q-function Q(b) as the
         * posterior probability of making the observation:
         * P(message | P(b), Policy(Q(.))) = \sum_b P(b) P(b.handleAct(Policy(Q(b))) == message)
         * If P(b) is expressed as a set of weighted body-state samples, b \in S, with weights w(b) then
         * P(message | S, Policy(Q(.))) \approx 1/\sum_{b' \in S}w(b)  \sum_{b \in S} w(b)\sum_a P(b.handleAct(a) == message)P(Policy(Q(b)) == a)
         * so
         *
         * dP(message | S, Policy(Q(.)))/dQ_a(b) \approx w(b)/\sum_{b' \in S}w(b) \sum_a' P(b.handleAct(a') == message)dP(Policy(Q(b))==a')/dQ_a(b)
         *
         * So, we need to use a differentiable policy. Also, we don't have the probability of getting a message given an
         * act, but we have a sampler of messages given an act, so for each 'a' we sample the message and add the dP/dF
         * term if sample == message. So for each b \in S
         *
         * dP(message | S, Policy(Q(.)))/dQ_a(b) \approx w(b)/\sum_{b' \in S}w(b) \sum_a' [b.handleAct(a') == message]dP(Policy(Q(b))==a')/dQ_a(b)
         *
         * so, we can calculate a loss at a single training point, b, given Q(b), w(b), the set of actions, A, which
         * gave rise to the observed message on sampling and the differential policy,
         *
         * N.B. if we assume a deterministic act->message map, then for a given observed message, the set of acts that give
         * rise to that message is fixed and the associated probabiities are 1, so the sampling over a is exact.
         */
        template<class BODY, class POLICY = minds::SoftMaxPolicy>
        class MessageLoss {
        public:

            arma::mat  trainingPoints;  // by-column list of training points
            arma::vec  weights;         // normalised weight of the training point
            arma::umat actions;         // actions over which to sum for each training point
            POLICY policy;              // policy from which to get gradient of P(act) w.r.t. qVector

            size_t insertCol = 0;
            bool bufferIsFull = false;
            arma::uvec batchCols;

            MessageLoss(size_t bufferSize, size_t batchSize, POLICY policy = SoftMaxPolicy()) :
            trainingPoints(BODY::dimensions, bufferSize),
            weights(bufferSize),
            actions(BODY::action_type::size, bufferSize),
            policy(std::move(policy)),
            batchCols(batchSize) {
            }


            void on(const events::IncomingMessageObservation<BODY> &observation) {
                double sumOfWeights = std::views::keys(observation.bodySamples).sum();
                for(const auto &[body, uweight] : observation.bodySamples) {
                    trainingPoints.col(insertCol) = static_cast<const arma::mat &>(observation.body);
                    weights(insertCol) = uweight/sumOfWeights;
                    actions.col(insertCol) = body.messageToAct(observation.message);
                }
                insertCol = (insertCol + 1)%trainingPoints.n_cols;
                if(insertCol == 0) bufferIsFull = true;
            }

            size_t capacity() const { return trainingPoints.n_cols; }
            size_t bufferSize() const { return bufferIsFull?capacity():insertCol; }
            size_t batchSize() const { return batchCols.n_rows; }

            template<class INPUTS>
            void trainingSet(INPUTS &trainingMat) {
                assert(bufferSize() > 0);
                batchCols = arma::randi<arma::uvec>(batchCols.n_rows, arma::distr_param(1, bufferSize() - 1));
                trainingMat = trainingPoints.cols(batchCols);
            }

            template<class OUTPUTS, class RESULT>
            void gradientByPrediction(const OUTPUTS &qVectors, RESULT &gradient) {
                size_t s = batchSize();
                gradient.zeroes();
                for(size_t col = 0; col < s; ++col) {
                    for(uint action : actions.col(col)) {
                        gradient.col(col) += policy.gradient(qVectors.col(col)).col(action);
                    }
                    gradient.col(col) *= weights(col);
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
         */
        using lossFunctions::SumOfLosses;
        using lossFunctions::IOLoss;
        using lossFunctions::WeightedLoss;
        template<class BODY, class QPOLICY = SoftMaxPolicy>
        class OffTreeLoss : public SumOfLosses<IOLoss,WeightedLoss<MessageLoss<BODY,QPOLICY>>> {
        public:
            typedef SumOfLosses<IOLoss,WeightedLoss<MessageLoss<BODY,QPOLICY>>> base_type;


            static constexpr double selfOtherLearningRatio = 2.0;

            OffTreeLoss(size_t qVectorBufferSize, size_t messageBufferSize, size_t messageBatchSize, QPOLICY qPolicy = SoftMaxPolicy()):
            base_type(
                    IOLoss(qVectorBufferSize, BODY::dimensions, BODY::action_type::size),
                    WeightedLoss(
                            selfOtherLearningRatio,
                            MessageLoss<BODY,QPOLICY>(messageBufferSize, messageBatchSize, qPolicy))) {
            }

        };


        /** A single node in the tree. Represents Q-values of all hidden states with a given observable history.
         *  For each hidden state, there is an associated "QEntry" (a map<BODY,QVector>::iterator) which
         *  identifies the body state and the Q-values for all acts from this state.
         **/
        template<class BODY>
        class TreeNode {
        public:
            typedef BODY::message_type message_type;
            typedef BODY body_type;
            typedef BODY::action_type action_type;

            struct QEntry {
                uint traceCount;
                QVector<action_type::size> qVector;

                QEntry() : traceCount(0) {}
            };

            std::map<BODY, QEntry> qEntries; // qVectors for current player.
            std::map<BODY, uint> otherPlayerDistribution; // sample counts of other player body states during self play
        private:
            std::map<message_type, TreeNode *> children;
//            std::array<TreeNode *, static_cast<size_t>(message_type::size)> children;   // ...indexed by actId. nullptr if child not present.
        public:

            ~TreeNode() { for (auto &child: children) delete (child.second); }

            /** Qvector for current body state, given complete episode history */
            QVector<action_type::size> &operator()(const BODY &body) {
                auto qEntryIt = qEntries.find(body);
                assert(qEntryIt != qEntries.end());
                return qEntryIt->second.qVector;
            }

//            TreeNode *getChildOrCreate(message_type message);
//            TreeNode *getChildOrNull(message_type message);
            TreeNode *getChild(message_type message, bool createNodeIfAbsent);
            TreeNode *unlinkChild(message_type message);
            void leavePassiveTrace(const BODY &body);
            template<bool LEAVETRACE> std::pair<QVector<action_type::size> *,bool> getQVecPtr(const BODY &body, bool canAddEntry);

            auto activePlayerBodySampler() {
                assert(!qEntries.empty());
                return DiscreteObjectDistribution<std::reference_wrapper<const BODY>>(
                        qEntries | std::views::keys,
                        qEntries | std::views::values | std::views::transform([](auto &item){ return item.traceCount; }));
            }

            auto passivePlayerBodySampler() {
                assert(!qEntries.empty());
                return DiscreteObjectDistribution<std::reference_wrapper<const BODY>>(
                        otherPlayerDistribution | std::views::keys,
                        otherPlayerDistribution | std::views::values);
            }

            size_t nActivePlayerSamples() {
                size_t count = 0;
                for(auto &item : qEntries) count += item.second.traceCount;
                return count;
            }

            const BODY *sampleActorBodyGivenMessage(message_type actorMessage);
        };


        // ===========================================================================
        // ======================= TREE NODE IMPLEMENTATION ==========================
        // ===========================================================================

        /** Gets a pointer to the qVector for the given body state.
         * If canAddEntry is true and there is no entry for body, then a new entry is added
         * if LEAVETRACE is true, and an entry exists, then the trace counter for body is incremented.
         * if no entry exists or can be added, then nullptr is returned.
         * The return value is a pair, the second entry of which, if true, indicates that a new entry was added */
        template<class BODY>
        template<bool LEAVETRACE>
        std::pair<QVector<BODY::action_type::size> *,bool> TreeNode<BODY>::getQVecPtr(const BODY &body, bool canAddEntry) {
            std::pair<QVector<action_type::size> *,bool> result(nullptr,false);
            decltype(qEntries.begin()) it;
            if(canAddEntry) {
                std::tie(it, result.second) = qEntries.try_emplace(body);
            } else {
                it = qEntries.find(body);
                if(it == qEntries.end()) return result;
            }
            result.first = &it->second.qVector;
            if constexpr(LEAVETRACE) ++(it->second.traceCount);
            return result;
        }


        /** */
        template<class BODY>
        void TreeNode<BODY>::leavePassiveTrace(const BODY &body) {
            auto [it, addedNewEntry] = otherPlayerDistribution.try_emplace(body, 0);
            ++(it->second);
        }


        /** Given that the current actor in this node produced a given message, sample from the posterior
         * This is needed when training the off tree mind on actual observed opponent behaviour.
         *  TODO: this should use the prior distribution given the move history too!
         * @tparam BODY
         * @tparam OffTreeQFunction
         * @tparam SelfPlayPolicy
         * @param actorMessage
         * @return
         */
        template<class BODY>
        const BODY *TreeNode<BODY>::sampleActorBodyGivenMessage(message_type actorMessage) {
            double totalWeight = 0.0;
            std::vector<double>         cumulativeWeights;
            std::vector<const BODY *>   states;
            for(const auto &[body, qVec]: qEntries) {
                auto maxQAct = sampleMaxQ(qEntries, body.legalActs());
                double w = body.actToMessageProb(maxQAct, actorMessage);
                totalWeight += w;
                cumulativeWeights.push_back(totalWeight);
                states.push_back(&body);
            }
            if(totalWeight == 0.0) { /* no state in this node could have produced this message */
                return nullptr;
            }
            // sample an element from cumulativeWeights with weighted prob
            double rand = deselby::random::uniform(0.0, totalWeight);
            auto chosenIt = std::ranges::upper_bound(cumulativeWeights, rand);
            assert(chosenIt != cumulativeWeights.end());
            return states[chosenIt - cumulativeWeights.begin()];
        }



//    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
//    void IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::SelfPlayMind::halfStepObservationHook(const observation_type &body) {
//        if(treeNode != nullptr && !hasAddedQEntry) treeNode->otherPlayerDistribution[body]++;
//    }

//        template<class BODY>
//        TreeNode<BODY> *TreeNode<BODY>::getChildOrNull(message_type message) {
//            auto childIt = children.find(message);
//            return childIt==children.end()?nullptr:childIt->second;
//        }
//
//        /** Returns the child of this node corresponding to the given act,
//        * creating a new one if necessary
//        *
//        * @param act the act that identifies the child
//        * @return a pointer to the child
//        */
//        template<class BODY>
//        TreeNode<BODY> *TreeNode<BODY>::getChildOrCreate(message_type message) {
//            auto [it, didInsert] = children.try_emplace(message, nullptr);
//            if (didInsert) it->second = new TreeNode();
//            return it->second;
//        }



        /**
         *
         * @param message identifies the child to unlink
         * @return the unlinked child
         */
        template<class BODY>
        TreeNode<BODY> *TreeNode<BODY>::unlinkChild(message_type message) {
            auto childIt = children.find(message);
            if(childIt != children.end()) {
                TreeNode *child = childIt->second;
                children.erase(childIt);
                return child;
            }
            return nullptr;
        }

        template<class BODY>
        TreeNode<BODY> *TreeNode<BODY>::getChild(message_type message, bool createNodeIfAbsent) {
            if(createNodeIfAbsent) {
                auto [it, didInsert] = children.try_emplace(message, nullptr);
                if (didInsert) it->second = new TreeNode();
                return it->second;
            }
            auto childIt = children.find(message);
            return childIt==children.end()?nullptr:childIt->second;
        }


        /** Make an entry for a new body state.
        *
        * @param agent
        * @return
        */
//        template<class BODY>
//        auto TreeNode<BODY>::addQEntry(const BODY &agent) {
//            auto [newEntry, wasInserted] = qEntries.try_emplace(
//                    agent); // insert uniform quality with zero sample count
//            assert(wasInserted);
//            return newEntry;
//        }




        /**
         * @return the total number of samples that have passed this tree node.
         */
//    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
//    size_t  IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::TreeNode::
//    nCurrentPlayerSamples() {
//        auto sampleCounts = std::ranges::views::values(currentPlayerDistribution);
//        return sampleCounts.sum(); // TODO: where's sum?
//    }


//    /**
//     * @return A function that returns a sample from the body states of other with a probability
//     * proportional to the total number of samples in each qEntry of this node.
//     */
//    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
//    std::function<BODY()>   IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::TreeNode::
//    createNextMoverSampler() {
//        std::vector<BODY>   states;
//        std::vector<double> weights;
//        assert(qEntries.size() > 0);
//        states.reserve(currentPlayerDistribution.size());
//        weights.reserve(currentPlayerInfo.size());
//        for(auto &entry: qEntries) {
//            states.emplace_back(entry.first);
//            weights.emplace_back(entry.second.totalSamples());
//        }
//        auto distribution = std::discrete_distribution(weights.begin(), weights.end());
//
//        return [distribution, states, &gen = randomGenerator]() mutable {
//            return states[distribution(gen)];
//        };
//    }
//
//    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
//    std::function<BODY()> IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::TreeNode::
//    createOtherPlayerSampler() {
//        std::vector<BODY>   states;
//        std::vector<double> weights;
//        if(otherPlayerDistribution.size() == 0) return {}; // undefined if no data
////        assert(otherPlayerDistribution.size() > 0);
//        states.reserve(otherPlayerDistribution.size());
//        weights.reserve(otherPlayerDistribution.size());
//        for(auto &entry: otherPlayerDistribution) {
//            states.emplace_back(entry.first);
//            weights.emplace_back(entry.second);
//        }
//        auto distribution = std::discrete_distribution(weights.begin(), weights.end());
//
//        return [distribution, states, &gen = randomGenerator]() mutable {
//            return states[distribution(gen)];
//        };
//    }



//    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy> template<bool LEAVETRACE, bool DOBACKPROP>
//    auto /* horrible typename otherwise */
//    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::createAgentSampler(BODY body) {
//        return [agent = Agent(body,SelfPlayMind<LEAVETRACE,DOBACKPROP>(*this))]() {
//            return agent;
//        };
//    }



        /** A SelfPlayMind is a Mind used to build the tree using self-play.
         * It can be thought of as a smart pointer into the tree that moves from the root
         * to a leaf then back-propagates to update Q-values.
         * During self-play, two SelfPlayMinds simultaneously navigate the same tree.
         * Each mind can add at most a single qEntry on their turn to act, creating new
         * TreeNodes in which to put the new qEntry if necessary.
         * [So, there are three states, onTree/notAddedQEntry, onTree/AddedQentry. offTree]
         *
         * The parameters LEAVETRACE and DOBACKPROP allow normal leraning (<true.true> plays against <true,true>)
         * or targeted improvement of Q-values without invalidating the posterior distributions or Q-values via
         * leaking information about the state of other player (<false,true> plays against <true,false>)
         *
         * @tparam LEAVETRACE   If true, the mind will increment counts in otherPlayerDistribution
         *                      as it passes through TreeNodes as the other player.
         * @tparam DOBACKPROP   If true, after the end of an episode, the mind will backpropogate and
         *                      update the Q-values of the TreeNodes it passed through as currentPlayer.
         */
        template<class BODY, class OFFTREEQFUNC, bool LEAVETRACE, bool DOBACKPROP>
        class SelfPlayQFunction {
        public:
            typedef BODY body_type;
            typedef body_type::message_type message_type;
            typedef body_type::action_type action_type;
            typedef OFFTREEQFUNC offtreeqfunc_type;

            TreeNode<BODY> *treeNode;// current treeNodes for player's experience, null if off the tree
            std::vector<QValue *> qValues; // Q values at choice points of the player
            std::vector<double> rewards; // reward between choice points of the player
            bool canAddToTree; // have we added a QEntry to the tree yet?
            offtreeqfunc_type &offTreeQFunction;// current treeNodes for player's experience, null if off the tree
            QVector<body_type::action_type::size> *lastQVector = nullptr;
            const double discount;

            SelfPlayQFunction(TreeNode<BODY> &treeNode, offtreeqfunc_type &offtreeqfunction, const double &discount) :
                    treeNode(treeNode), canAddToTree(DOBACKPROP), offTreeQFunction(offtreeqfunction), discount(discount) {}

            template<class TREE>
            SelfPlayQFunction(TREE &tree, deselby::ConstExpr<LEAVETRACE> /* LeaveTrace */, deselby::ConstExpr<DOBACKPROP> /* DoBackprop */) :
                    treeNode(tree.rootNode), canAddToTree(DOBACKPROP), offTreeQFunction(tree.offTreeQFunc), discount(tree.discount) {}


            void init(TreeNode<BODY> *rootNode) {
                treeNode = rootNode;
                qValues.clear();
                rewards.clear();
                canAddToTree = DOBACKPROP;
            }

            // ==== Mind interface

            auto operator()(const body_type &);

            void on(const events::AgentStep<action_type, message_type> &);
            void on(const events::PostActBodyState<body_type> &);
            void on(const events::IncomingMessage<message_type> &);
            void on(const events::AgentEndEpisode<body_type> &);

            // =====
        protected:
            bool isOnTree() const { return treeNode != nullptr; }
            QVector<action_type::size> offTreeQVector(const BODY &body) {
                QVector<action_type::size> offTreeQVec;
                arma::mat offTreeQMat = offTreeQFunction(body);
                for(int i=0; i < action_type::size; ++i) {
                    offTreeQVec[i].addSample(offTreeQMat[i]);
                }
                return offTreeQVec;
            }
        };

        template<class TREE, bool LEAVETRACE, bool DOBACKPROP>
        SelfPlayQFunction(TREE &tree, deselby::ConstExpr<LEAVETRACE> /* LeaveTrace */, deselby::ConstExpr<DOBACKPROP> /* DoBackprop */) ->
        SelfPlayQFunction<typename TREE::body_type, typename TREE::offtree_type, LEAVETRACE, DOBACKPROP>;


        /** On Incoming message:
         *  - update the treeNode
         *  - increment reward
         *  - leave trace if necessary
         * */
        template<class TREENODE, class OFFTREEQFUNC, bool LEAVETRACE, bool DOBACKPROP>
        void SelfPlayQFunction<TREENODE, OFFTREEQFUNC, LEAVETRACE, DOBACKPROP>::
        on(const events::IncomingMessage<message_type> &event) {
            if(isOnTree()) {
                treeNode = treeNode->getChild(event.message, canAddToTree);
            }
            if(!rewards.empty()) rewards.back() += event.reward;
        }



        template<class TREENODE, class OFFTREEQFUNC, bool LEAVETRACE, bool DOBACKPROP>
        void SelfPlayQFunction<TREENODE, OFFTREEQFUNC, LEAVETRACE, DOBACKPROP>::
        on(const events::AgentStep<action_type,message_type> &event) {
            if(isOnTree()) {
                assert(lastQVector != nullptr);
                if constexpr (DOBACKPROP) qValues.push_back(&((*lastQVector)[event.act]));
                treeNode = treeNode->getChild(event.message, canAddToTree);
            }
            rewards.push_back(event.reward); // new reward entry
        }


        template<class TREENODE, class OFFTREEQFUNC, bool LEAVETRACE, bool DOBACKPROP>
        void SelfPlayQFunction<TREENODE, OFFTREEQFUNC, LEAVETRACE, DOBACKPROP>::
        on(const events::PostActBodyState<body_type> &event) {
            if constexpr (LEAVETRACE) {
                if(isOnTree()) treeNode->leavePassiveTrace(event.body);
            }
        }


        template<class TREENODE, class OFFTREEQFUNC, bool LEAVETRACE, bool DOBACKPROP>
        void SelfPlayQFunction<TREENODE, OFFTREEQFUNC, LEAVETRACE, DOBACKPROP>::
        on(const events::AgentEndEpisode<body_type> &event) { // back propagate rewards
            if constexpr (DOBACKPROP) {
//                std::cout << "Starting backprop..." << std::endl;
                double cumulativeReward = 0.0;
                while (rewards.size() > qValues.size()) {
                    cumulativeReward = cumulativeReward * discount + rewards.back();
//                    std::cout << "Got offtree reawrd " << rewards.back() << std::endl;
                    rewards.pop_back();
                }
                while (!rewards.empty()) {
                    cumulativeReward = cumulativeReward * discount + rewards.back();
                    qValues.back()->addSample(cumulativeReward);
//                    std::cout << "Got ontree reawrd " << rewards.back() << " Cumulative reward = " << cumulativeReward << " qValue = " << *qValues.back() << std::endl;
                    rewards.pop_back();
                    qValues.pop_back();
                }
//                std::cout << "Ending backprop..." << std::endl;
            }
            // reset for next episode
//            treeNode = tree.rootNode;
//            qValues.clear();
//            rewards.clear();
//            hasAddedQEntry = false;
        }


// TODO: intercept AgentStep
//        template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy> template<bool LEAVETRACE,bool DOBACKPROP>
//        IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::action_type
//        IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::SelfPlayMind<LEAVETRACE,DOBACKPROP>::
//        act(const observation_type &body, action_mask legalActs, IncompleteInformationMCTS::reward_type rewardFromLastAct) {
//            QVector<action_type::size> *qEntry = nullptr; // if null, choose at random
//            if(isOnTree()) {
//                if (hasAddedQEntry) {
//                    // onTree but can't add new qEntry, so look for existing entry or null
//                    auto qIt = treeNode->qEntries.find(body);
//                    if(qIt != treeNode->qEntries.end()) qEntry = &(qIt->second.qVector);
//                } else {
//                    auto [qIt, insertedNewEntry] = treeNode->qEntries.try_emplace(body);
//                    if (insertedNewEntry) {
//                        hasAddedQEntry = true;
//                    }
//                    qEntry = &(qIt->second);
//                }
//            }
//            action_type action;
//            if(qEntry != nullptr) { // still on tree
//                action = tree.selfPlayPolicy.sample(*qEntry, legalActs);
//                if constexpr (DOBACKPROP) {
//                    rewards.push_back(rewardFromLastAct);
//                    qValues.push_back(&(*qEntry)[static_cast<size_t>(action)]);
//                }
//            } else { // off-tree
//                action = tree.offTreeQMind(body);
//                // static_cast<action_type>(sampleUniformly(legalActs));
//                if constexpr (DOBACKPROP) rewards.push_back(rewardFromLastAct);
//            }
//            return action;
//        }

        template<class TREENODE, class OFFTREEQFUNC, bool LEAVETRACE, bool DOBACKPROP>
        auto SelfPlayQFunction<TREENODE, OFFTREEQFUNC, LEAVETRACE, DOBACKPROP>::
        operator ()(const body_type &body) {
            if(isOnTree()) {
                bool addedNewEntry;
                std::tie(lastQVector, addedNewEntry) = treeNode->template getQVecPtr<LEAVETRACE>(body, canAddToTree);
//                if(addedNewEntry) { // set initial value to offTree value
//                    auto offTreeQVector = offTreeQFunction(body);
//                    for(int i=0; i<lastQVector->size(); ++i) (*lastQVector)[i].addSample(offTreeQVector[i]);
//                }
                canAddToTree = !addedNewEntry;
                if(lastQVector == nullptr) { // no qVector and can't add to tree
                    treeNode = nullptr;
                    return offTreeQVector(body);
                }
            } else {
                lastQVector == nullptr;
                return offTreeQVector(body);
            }
//            std::cout << "Ontree QVec = " << *lastQVector << std::endl;
            return *lastQVector;
//            QVector<action_type::size> *qEntry = nullptr;
//            if(isOnTree()) {
//                if (hasAddedQEntry) {
//                    // onTree but can't add new qEntry, so look for existing entry or null
//                    auto qIt = treeNode->qEntries.find(body);
//                    if(qIt != treeNode->qEntries.end()) qEntry = &(qIt->second.qVector);
//                } else {
//                    auto [qIt, insertedNewEntry] = treeNode->qEntries.try_emplace(body);
//                    if (insertedNewEntry) {
//                        hasAddedQEntry = true;
//                    }
//                    qEntry = &(qIt->second);
//                }
//            }
//            if(qEntry != nullptr) { // still on tree
//                qVal = *qEntry;
//                if constexpr (DOBACKPROP) {
//                    qValues.push_back(&(*qEntry)[static_cast<size_t>(action)]); //
//                }
//            } else { // off-tree
//                qVal = offTreeQFunction(body);
//            }
//            return action;
        }

    }


    /** An IncompleteInformationMCTS is a Q-function from current
     * body state to Q-vector.
     * The function learns from the following events:
     *  - AgentStartEpisode,
     *  - Act,
     *  - IncomingMessage,
     *  - BodyStateDrawnFrom   - Signifies that a given agent's body was publicly drawn from a given distribution
     *                           (but the results of the draw are private).
     *
     * At the start of an episode, it is assumed that the start states of the agents are drawn from a distribution
     * known to both agents.
     *
     * On each function evaluation, a playout tree is constructed by Monte-Carlo sampling (making use of any
     * valid tree that may already be known) using a pair of SelfPlayMinds and the assumed-to-be-known bodies.
     * In each state, my belief about the other agent's state and other agent's belief about my state are explicitly
     * represented as samples. Higher order beliefs aren't necessary as the start state distribution is known and
     * the bodies are assumed to be known.
     *
     * The SelfPlayMinds navigate a single tree, which they modify by adding at most one Q-entry per episode.
     * During back-propogation, they also modify all Q-values they have passed through. SelfPlayMinds intercept
     *  - IncomingMessage
     *  - Act
     *
     * If the SelfPlayMind goes off off-tree, then a single, shared OFFTREEMIND is used to generate actions from
     * body states. The OFFTREEMIND can learn from:
     *  - The Q-vectors generated by the tree.
     *  - The observed messages from the opponent, given our belief about his state.
     * These events are generated by the tree, on evaluation and on receipt of incoming messages.
     *
     *
     * @tparam BODY The body with which we should play out in order to build this tree
     */
    template<
            class OffTreeApproximator,
            class BODY,
            class SelfPlayPolicy = UpperConfidencePolicy<typename BODY::action_type>>
    class IncompleteInformationMCTS {
    public:

//        typedef BODY observation_type;
//        typedef BODY::action_mask action_mask;
        typedef BODY body_type;
        typedef OffTreeApproximator offtree_type;

//        typedef double reward_type;
        typedef BODY::action_type action_type;
        typedef BODY::message_type message_type; // in and out messages must be the same for self-play to be possible
//        typedef const episodes::SimpleEpisode<BODY,BODY> & init_type;

//        inline static std::default_random_engine randomGenerator = std::default_random_engine();

        /** A single node in the tree. Represents Q-values of all hidden states with a given observable history.
         *  For each hidden state, there is an associated "QEntry" (a map<BODY,QVector>::iterator) which
         *  identifies the body state and the Q-values for all acts from this state.
         **/
//        class TreeNode {
//        public:
//            std::map<BODY,QVector<action_type::size>>   qEntries; // qVectors for current player.
//            std::map<BODY,uint>     otherPlayerDistribution; // posterior of other player body states given move history
//        private:
//            std::map<message_type, TreeNode *> children;
////            std::array<TreeNode *, static_cast<size_t>(message_type::size)> children;   // ...indexed by actId. nullptr if child not present.
//        public:
//
//            TreeNode() { children.fill(nullptr); }
//            ~TreeNode() { for (TreeNode *child: children) delete (child); }
//
//            TreeNode *getChildOrCreate(message_type message);
//            TreeNode *getChildOrNull(message_type message);
//            TreeNode *unlinkChild(message_type message);
//            auto addQEntry(const BODY &agent);
//            const BODY *sampleActorBodyGivenMessage(message_type actorMessage);
//        };

        /** A SelfPlayMind is a Mind used to build the tree using self-play.
         * It can be thought of as a smart pointer into the tree that moves from the root
         * to a leaf then back-propagates to update Q-values.
         * During self-play, two SelfPlayMinds simultaneously navigate the same tree.
         * Each mind can add at most a single qEntry on their turn to act, creating new
         * TreeNodes in which to put the new qEntry if necessary.
         * [So, there are three states, onTree/notAddedQEntry, onTree/AddedQentry. offTree]
         *
         * The parameters LEAVETRACE and DOBACKPROP allow normal leraning (<true.true> plays against <true,true>)
         * or targeted improvement of Q-values without invalidating the posterior distributions or Q-values via
         * leaking information about the state of other player (<false,true> plays against <true,false>)
         *
         * @tparam LEAVETRACE   If true, the mind will increment counts in otherPlayerDistribution
         *                      as it passes through TreeNodes as the other player.
         * @tparam DOBACKPROP   If true, after the end of an episode, the mind will backpropogate and
         *                      update the Q-values of the TreeNodes it passed through as currentPlayer.
         */
//        template<bool LEAVETRACE, bool DOBACKPROP>
//        class SelfPlayMind {
//        public:
//            typedef BODY observation_type;
//            typedef BODY::action_mask action_mask;
//            typedef double reward_type;
//
//            IncompleteInformationMCTS<BODY,OffTreeMind, SelfPlayPolicy> &           tree;
//            TreeNode *                                  treeNode;// current treeNodes for player's experience, null if off the tree
//            std::vector<QValue *>                       qValues; // Q values at choice points of the player
//            std::vector<double>                         rewards; // reward between choice points of the player
//            bool                                        hasAddedQEntry;
//
//            SelfPlayMind(IncompleteInformationMCTS<BODY,OffTreeMind, SelfPlayPolicy> &tree):
//                    tree(tree), treeNode(tree.rootNode), hasAddedQEntry(!DOBACKPROP) {}
//
//            action_type act(const observation_type &body, action_mask legalActs, reward_type rewardFromLastAct);
//
//            // ==== Mind interface
//
//            action_type operator()(const BODY &);
//
//            void on(const events::Act<BODY,action_type, message_type> &outgoingMessage);
//            void on(const events::IncomingMessage<message_type,BODY> &incomingMessage);
//            template<class AGENT1, class AGENT2>
//            void on(const events::EndEpisode<AGENT1,AGENT2> &startEpisode);
//            void on(const events::Reward &);
//
//            // =====
//
//            bool isOnTree() { return treeNode != nullptr; }
//        };

        IIMCTS::TreeNode<BODY> *rootNode;                 // points to the rootNode. nullptr signifies no acts this episode yet.
        double discount;                    // discount of rewards into the future
        SelfPlayPolicy  selfPlayPolicy;     // policy used when building tree
        OffTreeApproximator     offTreeQFunc;       // mind to decide acts during self-play when off the tree.
//        std::map<BODY,uint> currentPlayerDistribution;  // posterior of rootNode current player state, given move history
                                            // N.B. entries may differ from info in qEntries due to sample boosting
                                            // but should be equal to other player distribution of parent.
        std::function<BODY(const BODY &)> selfStatePriorSampler; // other's belief about my state given his body state
        std::function<BODY(const BODY &)> otherStatePriorSampler;   // my belief about other's state given my body state
                                                                    // By assumption, other's belief about my state is
                                                                    // the same for all states in the support of the PMF
                                                                    // given my body state. Also by assumption we have the
        const uint minSelfPlaySamples;             // number of samples taken to build the tree before a decision is made
        const uint minQVecSamples ;

        static constexpr uint SelfPlayQVecSampleRatio = 10;
//        inline static const auto defaultOffTreeMind =
//                GreedyPolicy(explorationStrategies::NoExploration(),
//                                        approximators::FeedForwardNeuralNet(
//                                                {new mlpack::Linear(100),
//                                                 new mlpack::ReLU(),
//                                                 new mlpack::Linear(50),
//                                                 new mlpack::ReLU(),
//                                                 new mlpack::Linear(BODY::action_type::size)
//                                                }));

//        IncompleteInformationMCTS(size_t nSamplesPerTree, double discount, OffTreeMind offTreeMind = defaultOffTreeMind);

        IncompleteInformationMCTS(
                OffTreeApproximator offTreeApproximator,
                std::function<BODY(const BODY &)> selfStatePriorSampler,
                std::function<BODY(const BODY &)> otherStatePriorSampler,
                double discount,
                size_t nSamplesInATree,
                SelfPlayPolicy selfplaypolicy = UpperConfidencePolicy<typename BODY::action_type>()
        );

        IncompleteInformationMCTS(const IncompleteInformationMCTS<OffTreeApproximator, BODY, SelfPlayPolicy> &other) :
        rootNode(nullptr),
        discount(other.discount),
        selfPlayPolicy(other.selfPlayPolicy),
        offTreeQFunc(other.offTreeQFunc),
        selfStatePriorSampler(other.selfStatePriorSampler),
        otherStatePriorSampler(other.otherStatePriorSampler),
        minSelfPlaySamples(other.minSelfPlaySamples),
        minQVecSamples(other.minQVecSamples)
        {
            assert(other.rootNode == nullptr);
        }

        ~IncompleteInformationMCTS() { delete (rootNode); }

        // ----- Q-value function interface -----

//        action_type act(const observation_type &body, action_mask legalActs, [[maybe_unused]] reward_type rewardFromLastAct);

        /** rebuilds the tree using a new draw of distributions of player states.
         * Note that we assume that other's belief about our body state is independent of
         * the draw we take from otherSampler(), i.e. is the same for all states in the
         * support of the distribution. */
        void on(const events::AgentStartEpisode<BODY> & event) {
            delete(rootNode);
            rootNode = new IIMCTS::TreeNode<BODY>();
            auto otherSampler = [&selfBody = event.body, &sampler = otherStatePriorSampler]() {
                return sampler(selfBody);
            };
            auto selfSampler = [otherBody = otherSampler(), &sampler = selfStatePriorSampler]() {
                return sampler(otherBody);
            };
            if(event.isFirstMover) {
                doSelfPlay<true>(selfSampler, otherSampler, minSelfPlaySamples);
            } else {
                doSelfPlay<true>(otherSampler, selfSampler, minSelfPlaySamples);
            }
        }

        /** Train off-tree QFunction on message and shift the root */
        void on(const events::IncomingMessage<message_type> &incomingMessage) {
            assert(rootNode != nullptr);
            // learn from opponent move
//            const BODY *sampledOpponentBodyPtr = rootNode->sampleActorBodyGivenMessage(incomingMessage);
//            if(sampledOpponentBodyPtr != nullptr) {
//                QVector<action_type::size> qVec = rootNode->qEntries[*sampledOpponentBodyPtr];
//                callback(events::InputOutput((const BODY &)*sampledOpponentBodyPtr, qVec.toVector()), offTreeQFunc);
//            }
            callback(events::IncomingMessageObservation(rootNode->otherPlayerDistribution, incomingMessage.message), offTreeQFunc);
            shiftRoot(incomingMessage.message);
        }

        void on(const events::OutgoingMessage<message_type> &outgoingMessage) {
            assert(rootNode != nullptr);
            shiftRoot(outgoingMessage.message);
        }

//
//        void endEpisode([[maybe_unused]] double rewardFromFinalAct) {
//            delete(rootNode);
//            rootNode = nullptr;
//        }
//
//        /** rebuilds the tree using a new set of posterior distributions of player states.
//         * Should be called at the start of an episode with priors */
//        template<class CURRENTPLAYERSAMPLER, class OTHERPLAYERSAMPLER>
//        void startEpisode(CURRENTPLAYERSAMPLER currentPlayerBodySampler, OTHERPLAYERSAMPLER otherPlayerBodySampler) {
//            delete(rootNode);
//            rootNode = new IIMCTS::TreeNode<body_type>();
//            for (int nSamples = 0; nSamples < nSamplesPerTree; ++nSamples) {
//                Agent player1(currentPlayerBodySampler(), SelfPlayMind<true,true>(*this));
//                Agent player2(otherPlayerBodySampler(), SelfPlayMind<true,true>(*this));
//                currentPlayerDistribution[player1.body]++;
//                episodes::runAsync(player1,player2);
//            }
//        }

        // -------- Q-function interface --------
        // This whole tree is a QFunction, not a mind

        /** Ensuere correct number of samples for root node and body entry,
         * train offTreeQFunction on retreived Q-vector and return qvector */
        const QVector<action_type::size> &operator()(const body_type &body) {
            assert(rootNode != nullptr);
            const QVector<action_type::size> &qVec = (*rootNode)(body);

            auto rootNodeSamples = rootNode->nActivePlayerSamples();
            if(rootNodeSamples < minSelfPlaySamples) selfPlay(minSelfPlaySamples - rootNodeSamples);

            uint qVecSamples = qVec.totalSamples();
            if(qVecSamples < minQVecSamples) augmentSamples(body, minQVecSamples - qVecSamples);

            return qVec;
        }

    protected:
//        template<bool LEAVETRACE, bool DOBACKPROP> auto createAgentSampler(const std::map<BODY,uint> &);
//        // template<bool LEAVETRACE, bool DOBACKPROP> auto createAgentSampler(BODY);

        void shiftRoot(message_type message) {
            IIMCTS::TreeNode<BODY> *newRoot = rootNode->unlinkChild(message);
            assert(newRoot != nullptr);
            if(newRoot == nullptr) {
//                std::cerr << "Warning: Reality has gone off=tree (probably a sign of not enough samples in the tree)." << std::endl;
//                newRoot = new IIMCTS::TreeNode<BODY>();
                // TODO: Deal with particle depletion. Probably with MCMC over trajectories since the start
                //  of the episode, given the observations. This can be done without modelling the other
                //  agent, as the observations make the internal states of each agent independent.
                //
            }
            delete(rootNode);
            rootNode = newRoot;
        }

        void selfPlay(uint nEpisodes) {
            doSelfPlay<true>(rootNode->activePlayerBodySampler(), rootNode->passivePlayerBodySampler(), nEpisodes);
        }

        void augmentSamples(const BODY &body, uint nEpisodes) {
            doSelfPlay<false>([&body]() { return body; }, rootNode->passivePlayerBodySampler(), nEpisodes);
        }

        template<bool TRACECURRENTPLAYER, class SAMPLER1, class SAMPLER2>
        void doSelfPlay(
                SAMPLER1 &&player1BodySampler,
                SAMPLER2 &&player2BodySampler,
                uint nEpisodes) {
            constexpr bool BACKPROPOTHERPLAYER = TRACECURRENTPLAYER; // just to be explicit
            Agent player1(
                    player1BodySampler(),
                    QMind(
                            IIMCTS::SelfPlayQFunction(*this, deselby::ConstExpr<TRACECURRENTPLAYER>(), deselby::ConstExpr<true>()),
                            selfPlayPolicy)
                    );
            Agent player2(
                    player2BodySampler(),
                    QMind(
                            IIMCTS::SelfPlayQFunction(*this, deselby::ConstExpr<true>(), deselby::ConstExpr<BACKPROPOTHERPLAYER>()),
                            selfPlayPolicy)
                    );
            for (int nSamples = 0; nSamples < nEpisodes; ++nSamples) {
//                episodes::runAsync(player1, player2, callbacks::Verbose());
                episodes::runAsync(player1, player2);
                player1.body = player1BodySampler();
                player2.body = player2BodySampler();
                player1.mind.init(rootNode);
                player2.mind.init(rootNode);
            }
        }
    };



    template<class OffTreeApproximator, class BODY, class SelfPlayPolicy>
    IncompleteInformationMCTS<OffTreeApproximator, BODY, SelfPlayPolicy>::IncompleteInformationMCTS(
            OffTreeApproximator offTreeQFunction,
            std::function<BODY(const BODY &)> selfStatePriorSampler,
            std::function<BODY(const BODY &)> otherStatePriorSampler,
            double discount,
            size_t minSelfPlaySamples,
            SelfPlayPolicy selfplaypolicy)
            :
            offTreeQFunc(offTreeQFunction),
            selfStatePriorSampler(selfStatePriorSampler),
            otherStatePriorSampler(otherStatePriorSampler),
            rootNode(nullptr),
            minSelfPlaySamples(minSelfPlaySamples),
            minQVecSamples(minSelfPlaySamples / SelfPlayQVecSampleRatio),
            discount(discount),
            selfPlayPolicy(selfplaypolicy) {
    }


//    template<class OffTreeQFunction, class BODY, class SelfPlayPolicy>
//            template<bool LEAVETRACE, bool DOBACKPROP>
//    auto /* horrible typename otherwise */
//    IncompleteInformationMCTS<OffTreeQFunction, BODY, SelfPlayPolicy>::
//    createAgentSampler(const std::map<BODY,uint> &visitCounts) {
//////        std::vector<double> weights;
////        if(otherPlayerDistribution.size() == 0) return {}; // undefined if no data
//////        assert(otherPlayerDistribution.size() > 0);
////        states.reserve(visitCounts.size());
////        weights.reserve(visitCounts.size());
////        for(auto &entry: visitCounts) {
////            states.emplace_back(entry.first);
////            weights.emplace_back(entry.second);
////        }
//        assert(!visitCounts.empty());
//        std::vector<BODY>   states(std::ranges::views::keys(visitCounts));
//        auto weights = std::ranges::views::values(visitCounts);
//        auto distribution = std::discrete_distribution(weights.begin(), weights.end());
//
//        IIMCTS::SelfPlayQFunction<BODY, OffTreeQFunction, LEAVETRACE,DOBACKPROP> qFunction(*this);
//        return [distribution, states, mind, &gen = randomGenerator]() {
//            return Agent(states[distribution(gen)],mind);
//        };
//    }


//    template<class BODY, class OffTreeQFunction, class SelfPlayPolicy>
//    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::
//    IncompleteInformationMCTS(size_t nSamplesPerTree, double discount, OffTreeQFunction offTreeMind) :
//            rootNode(nullptr),
//            nSamplesPerTree(nSamplesPerTree),
//            discount(discount),
//            offTreeQMind(std::move(offTreeMind)) { }



//    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
//    IncompleteInformationMCTS<BODY,OffTreeQFunction,SelfPlayPolicy>::action_type
//    IncompleteInformationMCTS<BODY,OffTreeQFunction,SelfPlayPolicy>::act(const observation_type &body, action_mask legalActs,
//                                                                         IncompleteInformationMCTS::reward_type rewardFromLastAct) {
//        // onTree and can add entry so try emplace
////            std::cout << "Available bodies \n" << (rootNode->qEntries | std::views::keys) << std::endl;
////            std::cout << "Search body \n" << body << std::endl;
////            std::cout << "Equality vector\n" << (rootNode->qEntries | std::views::keys | std::views::transform([&body](auto &bod) { return bod == body; })) << std::endl;
////            std::cout << "Size of qEntry map " << rootNode->qEntries.size() << std::endl;
////            std::cout << "Size of other  map " << rootNode->otherPlayerDistribution.size() << std::endl;
////            std::cout << "Nsamples " << rootNode->nSamples() << std::endl;
//
//        assert(rootNode != nullptr);
//        buildTree(episodes::SimpleEpisode(rootNode->createNextMoverSampler(), rootNode->createOtherPlayerSampler()));
//
//        QVector<action_type::size> qVec = rootNode->qEntries.at(body); // TODO: what if body isn't present?
////            std::cout << "QVector = " << qVec << std::endl;
//        // Train off-tree function on tree Q-values
//        offTreeQMind.train(observations::InputOutput(body, qVec.toVector()));
//
//        action_type action = finalDecisionPolicy.sample(qVec, legalActs);
//        return action;
//    }

//    /** Build a tree from the
//     *
//     * @param nSamples
//     * @param discount
//     */
//    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
//    void    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::
//    buildTree(const episodes::SimpleEpisode<BODY> &episode, int nSamples) {
//        for (int nSamples = rootNode->nSamples(); nSamples < nSamples; ++nSamples) {
//            Agent<BODY,SelfPlayMind> player1(episode.sampleFirstMoverBody(), SelfPlayMind(*this));
//            Agent<BODY,SelfPlayMind> player2(episode.sampleSecondMoverBody(), SelfPlayMind(*this));
//            episodes::episode(player1,player2);
//        }
//    }
}
#endif //MULTIAGENTGOVERNMENT_INCOMPLETEINFORMATIONMCTS_H
