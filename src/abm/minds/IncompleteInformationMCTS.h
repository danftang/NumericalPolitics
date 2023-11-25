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
//#include <boost/circular_buffer.hpp>
//#include <armadillo>

#include "../Agent.h"
//#include "../societies/RandomEncounterSociety.h"
#include "../minds/qLearning/QVector.h"
#include "../minds/qLearning/GreedyPolicy.h"
#include "../minds/qLearning/UpperConfidencePolicy.h"
#include "ZeroIntelligence.h"
#include "../../DeselbyStd/stlstream.h"
#include "../episodes/SimpleEpisode.h"
#include "../lossFunctions/SumOfLosses.h"
#include "../lossFunctions/WeightedLoss.h"
#include "QMind.h"
//#include "../lossFunctions/IOLoss.h"
#include "../minds/qLearning/SoftMaxPolicy.h"
//#include "../../DeselbyStd/DiscreteObjectDistribution.h"
#include "../approximators/AdaptiveFunction.h"
#include "../lossFunctions/IIMCTSLosses.h"


namespace abm::minds {
    namespace IIMCTS {

        /** If qFunc is a DifferentiableParameterisedFunction and doesn't intercept IncomingMessageObservation events
         * then assume it's a raw approximator and wrap it in a DifferentiableAdaptiveFunction with OffTreeLoss
         * as a loss function.
         * Uses default optimizer and buffer sizes.  */
        template<class BODY, DifferentiableParameterisedFunction<lossFunctions::OffTreeLoss<BODY>> QFUNC> requires (!HasCallback<QFUNC, events::QEntryObservation<BODY>>)
        static auto convertToOffTreeQFunc(QFUNC &&qFunc) {
            return approximators::DifferentiableAdaptiveFunction(
                    std::forward<QFUNC>(qFunc),
                    lossFunctions::OffTreeLoss<BODY>());
        }

        template<class BODY, class QFUNC> requires HasCallback<QFUNC, events::QEntryObservation<BODY>>
        static std::remove_reference_t<QFUNC> convertToOffTreeQFunc(QFUNC &&qFunc) {
            return std::forward<QFUNC>(qFunc);
        }

        template<size_t Qsize>
        struct QEntry {
            uint traceCount = 0;
            QVector<Qsize> qVector;
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
            typedef QVector<BODY::action_type::size> qvector_type;
            typedef std::map<BODY, QEntry<BODY::action_type::size>> qentries_type;
            typedef qentries_type::iterator q_iterator_type;

            qentries_type qEntries; // qVectors for current player.
            std::map<BODY, uint> otherPlayerDistribution; // sample counts of other player body states during self play
        private:
            std::map<message_type, TreeNode *> children;
//            std::array<TreeNode *, static_cast<size_t>(message_type::size)> children;   // ...indexed by actId. nullptr if child not present.
        public:

            ~TreeNode() { for (auto &child: children) delete (child.second); }

            /** Qvector for current body state, given complete episode history */
            qvector_type &operator()(const BODY &body) {
                auto [qEntryIt, addedEntry ] = findQEntry<false>(body, true);
                return qEntryIt->second.qVector;
            }

            TreeNode *getChild(message_type message, bool createNodeIfAbsent);
            TreeNode *unlinkChild(message_type message);
            void leavePassiveTrace(const BODY &body);

            template<bool LEAVETRACE>
            std::pair<q_iterator_type, bool>
                    findQEntry(const BODY &body, bool canAddEntry);

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

//            const BODY *sampleActorBodyGivenMessage(message_type actorMessage);
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
        std::pair<typename TreeNode<BODY>::q_iterator_type, bool> TreeNode<BODY>::findQEntry(const BODY &body, bool canAddEntry) {
            std::pair<q_iterator_type ,bool> result =
                    canAddEntry ? qEntries.try_emplace(body) : std::pair(qEntries.find(body), false);
            if constexpr(LEAVETRACE) if(result.first != qEntries.end()) ++(result.first->second.traceCount);
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
//        template<class BODY>
//        const BODY *TreeNode<BODY>::sampleActorBodyGivenMessage(message_type actorMessage) {
//            double totalWeight = 0.0;
//            std::vector<double>         cumulativeWeights;
//            std::vector<const BODY *>   states;
//            for(const auto &[body, qVec]: qEntries) {
//                auto maxQAct = sampleMaxQ(qEntries, body.legalActs());
//                double w = body.actToMessageProb(maxQAct, actorMessage);
//                totalWeight += w;
//                cumulativeWeights.push_back(totalWeight);
//                states.push_back(&body);
//            }
//            if(totalWeight == 0.0) { /* no state in this node could have produced this message */
//                return nullptr;
//            }
//            // sample an element from cumulativeWeights with weighted prob
//            double rand = deselby::random::uniform(0.0, totalWeight);
//            auto chosenIt = std::ranges::upper_bound(cumulativeWeights, rand);
//            assert(chosenIt != cumulativeWeights.end());
//            return states[chosenIt - cumulativeWeights.begin()];
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




        /** A SelfPlayQFunction is a Q-function used to build the tree using self-play.
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
            TreeNode<BODY>::q_iterator_type lastQEntry;
            const double discount;

            static constexpr bool INITTREEFROMOFFTREE = true; // Initialise new Q-vectors from the offtree Q-function?

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

            TreeNode<BODY>::qvector_type operator()(const body_type &);

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
         *  - increment reward for this step (if not start of episode and we're second mover)
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
                if constexpr (DOBACKPROP) qValues.push_back(&((lastQEntry->second.qVector)[event.act]));
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



        template<class BODY, class OFFTREEQFUNC, bool LEAVETRACE, bool DOBACKPROP>
        TreeNode<BODY>::qvector_type SelfPlayQFunction<BODY, OFFTREEQFUNC, LEAVETRACE, DOBACKPROP>::
        operator ()(const BODY &body) {
            if(isOnTree()) {
                bool addedNewEntry;
                std::tie(lastQEntry, addedNewEntry) = treeNode->template findQEntry<LEAVETRACE>(body, canAddToTree);
                if constexpr(INITTREEFROMOFFTREE) if(addedNewEntry) { // set initial value to offTree value
                    lastQEntry->second.qVector = offTreeQFunction(body);
                }
                canAddToTree = canAddToTree && !addedNewEntry;
                if(lastQEntry == treeNode->qEntries.end()) { // no qVector and can't add to tree
                    treeNode = nullptr;
                    return offTreeQFunction(body);
                } else {
                    callback(events::QEntryObservation<BODY>{lastQEntry}, offTreeQFunction); // teach offTreeQFunc on qEntry
                }
            } else {
                lastQEntry == treeNode->qEntries.end();
                return offTreeQFunction(body);
            }
//            std::cout << "Ontree QVec = " << *lastQVector << std::endl;
            return lastQEntry->second.qVector;
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
            class SelfPlayPolicy /*= UpperConfidencePolicy<typename BODY::action_type>*/>
    class IncompleteInformationMCTS {
    public:
        typedef BODY body_type;
        typedef OffTreeApproximator offtree_type;

        typedef BODY::action_type action_type;
        typedef BODY::message_type message_type; // in and out messages must be the same for self-play to be possible

        IIMCTS::TreeNode<BODY> *rootNode;                 // points to the rootNode. nullptr signifies no acts this episode yet.
        double                  discount;                    // discount of rewards into the future
        SelfPlayPolicy          selfPlayPolicy;     // policy used when building tree
        OffTreeApproximator     offTreeQFunc;       // mind to decide acts during self-play when off the tree.
        std::function<BODY(const BODY &)> selfStatePriorSampler; // other's belief about my state given his body state
        std::function<BODY(const BODY &)> otherStatePriorSampler;   // my belief about other's state given my body state
                                                                    // By assumption, other's belief about my state is
                                                                    // the same for all states in the support of the PMF
                                                                    // given my body state. Also by assumption we have the
        const uint minSelfPlaySamples;              // minimum no of samples in a tree before a Q-vector is returned
        const uint minQVecSamples ;                 // minimum number of samples in a returned Q-vector.

        static constexpr uint SelfPlayQVecSampleRatio = 10;

        /** 
         * @param offTreeApproximator If this is a DifferentiableParameterisedFunction. If it doesn't intercept
         *                              IncomingMessageObservation events, it will be paired with OffTreeLoss
         *                              to make an adaptible function. Otherwise will be used as-is.
         * @param selfStatePriorSampler Sampler for other's belief in my body state, given other's body state
         * @param otherStatePriorSampler Sampler for my belief in other's body state, given my body state.
         * @param discount              future discount when calculating rewards
         * @param minSelfPlaySamples    minimum number of samples a tree should have on evaluation
         * @param selfplaypolicy        q-policy to use when doing self-play.
         */
        template<class OFFTREEAPPROX>
        IncompleteInformationMCTS(
                OFFTREEAPPROX offTreeApproximator,
                std::function<BODY(const BODY &)> selfStatePriorSampler,
                std::function<BODY(const BODY &)> otherStatePriorSampler,
                double discount,
                size_t minSelfPlaySamples,
                SelfPlayPolicy selfplaypolicy = UpperConfidencePolicy<typename BODY::action_type>()
        ):
                offTreeQFunc(IIMCTS::convertToOffTreeQFunc<BODY>(offTreeApproximator)),
                selfStatePriorSampler(selfStatePriorSampler),
                otherStatePriorSampler(otherStatePriorSampler),
                rootNode(nullptr),
                minSelfPlaySamples(minSelfPlaySamples),
                minQVecSamples(minSelfPlaySamples / SelfPlayQVecSampleRatio),
                discount(discount),
                selfPlayPolicy(selfplaypolicy) {
        }

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
            // train off-tree QFunction on other's observed move
            callback(events::IncomingMessageObservation(rootNode->otherPlayerDistribution, incomingMessage.message), offTreeQFunc);
            shiftRoot(incomingMessage.message);
        }

        void on(const events::OutgoingMessage<message_type> &outgoingMessage) {
            assert(rootNode != nullptr);
            shiftRoot(outgoingMessage.message);
        }


        // -------- Q-function interface --------

        /** Ensuere correct number of samples for root node and body entry,
         * train offTreeQFunction on retreived Q-vector and return qvector */
        const QVector<action_type::size> &operator()(const body_type &body) {
            assert(rootNode != nullptr);

            auto rootNodeSamples = rootNode->nActivePlayerSamples();
            if(rootNodeSamples < minSelfPlaySamples) selfPlay(minSelfPlaySamples - rootNodeSamples);

            const QVector<action_type::size> &qVec = (*rootNode)(body);

            uint qVecSamples = qVec.totalSamples();
            if(qVecSamples < minQVecSamples) augmentSamples(body, minQVecSamples - qVecSamples);

            return qVec;
        }

    protected:

        void shiftRoot(message_type message) {
            IIMCTS::TreeNode<BODY> *newRoot = rootNode->unlinkChild(message);
            if(newRoot == nullptr) {
                throw(std::logic_error("Reached a tree-node with no samples. Resampling not implemented yet. Try increasing the number of samples in the tree."));
//                std::cerr << "Warning: Reality has gone off=tree (probably a sign of not enough samples in the tree)." << std::endl;
//                newRoot = new IIMCTS::TreeNode<BODY>();
                // TODO: Deal with particle depletion. Probably with MCMC over trajectories since the start
                //  of the episode, given the observations. This can be done without modelling the other
                //  agent, as the observations make the internal states of each agent independent.
                //  The Q-function itself defines a likkelihood function, so can be used to approximate the
		        //  posterior!!
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

    template<class BODY, class SelfPlayPolicy = UpperConfidencePolicy<typename BODY::action_type>, class OFFTREEQFUNC>
    IncompleteInformationMCTS(
            OFFTREEQFUNC offtreeqfunc,
            std::function<BODY(const BODY &)>,
            std::function<BODY(const BODY &)>,
            double,
            size_t,
            SelfPlayPolicy = SelfPlayPolicy())
            ->
            IncompleteInformationMCTS<
                decltype(IIMCTS::convertToOffTreeQFunc<BODY>(offtreeqfunc)),
                BODY,
                SelfPlayPolicy>;
}
#endif //MULTIAGENTGOVERNMENT_INCOMPLETEINFORMATIONMCTS_H
