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

#include "../Agent.h"
#include "../societies/RandomEncounterSociety.h"
#include "../minds/qLearning/QVector.h"
#include "../minds/qLearning/GreedyPolicy.h"
#include "../UpperConfidencePolicy.h"
#include "ZeroIntelligence.h"
#include "../../DeselbyStd/stlstream.h"
#include "../episodes/SimpleEpisode.h"
#include "QMind.h"

namespace abm::events {
    template<class BODY>
    struct IncomingMessageObservation {
        std::vector<BODY> bodySamples;
        BODY::message_type message;
    };

    template<class BODY>
    struct TreeQVectorObservation {
        BODY body;
        std::vector<double> qVector;
    };
}

namespace abm::minds {

    namespace IIMCTS {

        /** Intercepts (BODY,Q-vector) pairs and (BODY state PMF, message) pairs to make a loss function
         * for a parameterised approximator.
         *
         * Given a (BODY state PMF, message) pair and a Q-policy, we define the loss of a function F(x) as the
         * posterior probability of making the observation:
         * P(message | P(B), Policy(F)) = \sum_B P(B) P(B.handleAct(Policy(F(B))) == message)
         * If P(B) is expressed as a set of body-state samples, S, then
         * P(message | S, Policy(F)) \approx 1/|S| \sum_{B \in S} \sum_a P(B.handleAct(a) == message)P(Policy(F(B)) == a)
         * so
         * dP(message | S, Policy(F))/dF(B) \approx 1/|S| \sum_a P(B.handleAct(a) == message)dP(Policy(F(B))==a)/dF(B)
         * So, we need to use a differentiable policy. Also, we don't have the probability of getting a message given an
         * act, but we have a sampler of messages given an act, so for each a we sample the message and add the dP/dF
         * term if sample == message.
         */
        template<class BODY, class QPOLICY>
        class OffTreeLossFunction {
        public:
            typedef size_t action_type;
            QPOLICY qPolicy;
            std::vector<events::TreeQVectorObservation<BODY>>     qVectorObservationBuffer;
            std::vector<events::IncomingMessageObservation<BODY>> messageObservationBuffer;


        };

        /** A single node in the tree. Represents Q-values of all hidden states with a given observable history.
         *  For each hidden state, there is an associated "QEntry" (a map<BODY,QVector>::iterator) which
         *  identifies the body state and the Q-values for all acts from this state.
         **/
        template<class BODY, class MESSAGE>
        class TreeNode {
        public:
            typedef MESSAGE message_type;
            typedef BODY body_type;

            std::map<BODY, QVector<body_type::action_type::size>> qEntries; // qVectors for current player.
            std::map<BODY, uint> otherPlayerDistribution; // posterior of other player body states given move history
        private:
            std::map<message_type, TreeNode *> children;
//            std::array<TreeNode *, static_cast<size_t>(message_type::size)> children;   // ...indexed by actId. nullptr if child not present.
        public:

            TreeNode() { children.fill(nullptr); }

            ~TreeNode() { for (TreeNode *child: children) delete (child); }

            TreeNode *getChildOrCreate(message_type message);

            TreeNode *getChildOrNull(message_type message);

            TreeNode *unlinkChild(message_type message);

            auto addQEntry(const BODY &agent);

            const BODY *sampleActorBodyGivenMessage(message_type actorMessage);
        };


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
        template<class TREENODE, class OFFTREEQFUNC, bool LEAVETRACE, bool DOBACKPROP>
        class SelfPlayQFunction {
        public:
            typedef TREENODE::body_type body_type;
            typedef TREENODE::message_type message_type;
            typedef body_type::action_type action_type;
            typedef OFFTREEQFUNC offtreeqfunc_type;

            TREENODE *treeNode;// current treeNodes for player's experience, null if off the tree
            std::vector<QValue *> QValues; // Q values at choice points of the player
            std::vector<double> rewards; // reward between choice points of the player
            bool hasAddedQEntry; // have we added a QEntry to the tree yet?
            offtreeqfunc_type &offTreeQFunction;// current treeNodes for player's experience, null if off the tree

            explicit SelfPlayQFunction(TREENODE &treeNode, offtreeqfunc_type &offtreeqfunction) :
                    treeNode(treeNode), hasAddedQEntry(!DOBACKPROP), offTreeQFunction(offtreeqfunction) {}

            // ==== Mind interface

            action_type operator()(const body_type &);

            void on(const events::OutgoingMessage<action_type, message_type> &outgoingMessage);

            void on(const events::IncomingMessage<message_type> &incomingMessage);
//            template<class AGENT1, class AGENT2>
//            void on(const events::EndEpisode<AGENT1,AGENT2> &startEpisode);
//            void on(const events::Reward &);

            // =====

            bool isOnTree() const { return treeNode != nullptr; }
        };
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
            class OffTreeApproximator, // Parameterised approximator from Body to Q-vector
            class BODY,
            class OPPONENTBODY = BODY,
            class SelfPlayPolicy = UpperConfidencePolicy<typename BODY::action_type>>
    class IncompleteInformationMCTS {
    public:

//        typedef BODY observation_type;
//        typedef BODY::action_mask action_mask;
//        typedef BODY body_type;

        typedef double reward_type;
        typedef BODY::action_type action_type;
        typedef BODY::message_type message_type; // in and out messages must be the same for self-play to be possible
//        typedef const episodes::SimpleEpisode<BODY,BODY> & init_type;

        inline static std::default_random_engine randomGenerator = std::default_random_engine();

        /** A single node in the tree. Represents Q-values of all hidden states with a given observable history.
         *  For each hidden state, there is an associated "QEntry" (a map<BODY,QVector>::iterator) which
         *  identifies the body state and the Q-values for all acts from this state.
         **/
        class TreeNode {
        public:
            std::map<BODY,QVector<action_type::size>>   qEntries; // qVectors for current player.
            std::map<BODY,uint>     otherPlayerDistribution; // posterior of other player body states given move history
        private:
            std::map<message_type, TreeNode *> children;
//            std::array<TreeNode *, static_cast<size_t>(message_type::size)> children;   // ...indexed by actId. nullptr if child not present.
        public:

            TreeNode() { children.fill(nullptr); }
            ~TreeNode() { for (TreeNode *child: children) delete (child); }

            TreeNode *getChildOrCreate(message_type message);
            TreeNode *getChildOrNull(message_type message);
            TreeNode *unlinkChild(message_type message);
            auto addQEntry(const BODY &agent);
            const BODY *sampleActorBodyGivenMessage(message_type actorMessage);
        };

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
        template<bool LEAVETRACE, bool DOBACKPROP>
        class SelfPlayMind {
        public:
            typedef BODY observation_type;
            typedef BODY::action_mask action_mask;
            typedef double reward_type;

            IncompleteInformationMCTS<BODY,OffTreeMind, SelfPlayPolicy> &           tree;
            TreeNode *                                  treeNode;// current treeNodes for player's experience, null if off the tree
            std::vector<QValue *>                       qValues; // Q values at choice points of the player
            std::vector<double>                         rewards; // reward between choice points of the player
            bool                                        hasAddedQEntry;

            SelfPlayMind(IncompleteInformationMCTS<BODY,OffTreeMind, SelfPlayPolicy> &tree):
                    tree(tree), treeNode(tree.rootNode), hasAddedQEntry(!DOBACKPROP) {}

            action_type act(const observation_type &body, action_mask legalActs, reward_type rewardFromLastAct);

            // ==== Mind interface

            action_type operator()(const BODY &);

            void on(const events::Act<BODY,action_type, message_type> &outgoingMessage);
            void on(const events::IncomingMessage<message_type,BODY> &incomingMessage);
            template<class AGENT1, class AGENT2>
            void on(const events::EndEpisode<AGENT1,AGENT2> &startEpisode);
            void on(const events::Reward &);

            // =====

            bool isOnTree() { return treeNode != nullptr; }
        };


        TreeNode *rootNode;                 // points to the rootNode. nullptr signifies no acts this episode yet.
        size_t nSamplesPerTree;             // number of samples taken to build the tree before a decision is made
        double discount;                    // discount of rewards into the future
        SelfPlayPolicy  selfPlayPolicy;     // policy used when building tree
        OffTreeMind     offTreeQMind;       // mind to decide acts during self-play when off the tree.
        std::map<BODY,uint> currentPlayerDistribution;  // posterior of rootNode current player state, given move history
                                            // N.B. entries may differ from info in qEntries due to sample boosting
                                            // but should be equal to other player distribution of parent.

//        inline static const auto defaultOffTreeMind =
//                GreedyPolicy(explorationStrategies::NoExploration(),
//                                        approximators::FeedForwardNeuralNet(
//                                                {new mlpack::Linear(100),
//                                                 new mlpack::ReLU(),
//                                                 new mlpack::Linear(50),
//                                                 new mlpack::ReLU(),
//                                                 new mlpack::Linear(BODY::action_type::size)
//                                                }));

        IncompleteInformationMCTS(size_t nSamplesPerTree, double discount, OffTreeMind offTreeMind = defaultOffTreeMind);

        IncompleteInformationMCTS(
                OffTreeApproximator offTreeApproximator,
                std::function<BODY(const BODY &)> selfStateSampler,
                std::function<OPPONENTBODY(const BODY &)> opponentStateSampler,
                double discount,
                size_t nSamplesInATree,
                SelfPlayPolicy selfplaypolicy = UpperConfidencePolicy<typename BODY::action_type>()
        );

        ~IncompleteInformationMCTS() { delete (rootNode); }

        // ----- Q-value function interface -----

//        action_type act(const observation_type &body, action_mask legalActs, [[maybe_unused]] reward_type rewardFromLastAct);


        void on(const events::IncomingMessage<message_type,BODY> &incomingMessage) {
            assert(rootNode != nullptr);
            // learn from opponent move
            const BODY *sampledOpponentBodyPtr = rootNode->sampleActorBodyGivenMessage(incomingMessage);
            if(sampledOpponentBodyPtr != nullptr) {
                QVector<action_type::size> qVec = rootNode->qEntries[*sampledOpponentBodyPtr];
                callback(approximators::events::InputOutput((const BODY &)*sampledOpponentBodyPtr, qVec.toVector()), offTreeQMind);
            }

            shiftRoot(incomingMessage);
        }


        void on(const events::Act<BODY,action_type, message_type> &outgoingMessage) {
            assert(rootNode != nullptr);
            shiftRoot(outgoingMessage);
        }

        void on(events::Reward reward) {
            // TODO: handle reward.
        }
//
//        void endEpisode([[maybe_unused]] double rewardFromFinalAct) {
//            delete(rootNode);
//            rootNode = nullptr;
//        }
//
        /** rebuilds the tree using a new set of posterior distributions of player states.
         * Should be called at the start of an episode with priors */
        template<class CURRENTPLAYERSAMPLER, class OTHERPLAYERSAMPLER>
        void startEpisode(CURRENTPLAYERSAMPLER currentPlayerBodySampler, OTHERPLAYERSAMPLER otherPlayerBodySampler) {
            delete(rootNode);
            rootNode = new TreeNode();
            for (int nSamples = 0; nSamples < nSamplesPerTree; ++nSamples) {
                Agent player1(currentPlayerBodySampler(), SelfPlayMind<true,true>(*this));
                Agent player2(otherPlayerBodySampler(), SelfPlayMind<true,true>(*this));
                currentPlayerDistribution[player1.body]++;
                episodes::runAsync(player1,player2);
            }
        }

        // -------- Q-function interface --------
        // This whole tree is a QFunction, not a mind

        const QVector<action_type::size> &operator()(const BODY &body) {
            assert(rootNode != nullptr);
            const size_t minQVecSamples = nSamplesPerTree/10;
            const QVector<action_type::size> &qVec = rootNode->qEntries[body];
            size_t treeSampleDeficit = nSamplesPerTree - rootNode->nCurrentPlayerSamples();
            auto currentPlayerSampler = createAgentSampler<true,true>(rootNode->currentPlayerDistribution);
            auto otherPlayerSampler = createAgentSampler<true,true>(rootNode->otherPlayerDistribution);
            while(treeSampleDeficit > 0) {
                --treeSampleDeficit;
                episodes::runAsync(currentPlayerSampler(), otherPlayerSampler());
            }
            size_t qVecSampleDeficit = minQVecSamples - qVec.totalSamples();
            if(qVecSampleDeficit > 0) { // need to boost number of samples in our qVec
                auto noTraceOtherPlayerSampler = createAgentSampler<true,false>(rootNode->otherPlayerDistribution);
                while(qVecSampleDeficit > 0) {
                    --qVecSampleDeficit;
                    episodes::runAsync(Agent(body,SelfPlayMind<false,true>(*this)), noTraceOtherPlayerSampler());
                }
            }
            return qVec;
        }

    protected:
        template<bool LEAVETRACE, bool DOBACKPROP> auto createAgentSampler(const std::map<BODY,uint> &);
        template<bool LEAVETRACE, bool DOBACKPROP> auto createAgentSampler(BODY);
        void shiftRoot(message_type message);
    };

    template<class OffTreeApproximator, class BODY, class OPPONENTBODY, class SelfPlayPolicy>
    IncompleteInformationMCTS<OffTreeApproximator, BODY, OPPONENTBODY, SelfPlayPolicy>::IncompleteInformationMCTS(
            OffTreeApproximator offTreeApproximator,
            std::function<BODY(const BODY &)> selfStateSampler,
            std::function<OPPONENTBODY(const BODY &)> opponentStateSampler,
            double discount,
            size_t nSamplesInATree,
            SelfPlayPolicy selfplaypolicy) :
    rootNode(nullptr),
    nSamplesPerTree(nSamplesPerTree),
    discount(discount),
    selfPlayPolicy(selfplaypolicy) {
    }


    template<class BODY, class OffTreeQFunction, class SelfPlayPolicy>
    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::
    IncompleteInformationMCTS(size_t nSamplesPerTree, double discount, OffTreeQFunction offTreeMind) :
            rootNode(nullptr),
            nSamplesPerTree(nSamplesPerTree),
            discount(discount),
            offTreeQMind(std::move(offTreeMind)) { }



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

    /** When either agent actually moves, we need to move the root node down the tree
     * and delete tree nodes that are above the new root.
     *
     * @param message identifies the child of the rootNode that will become the new root
     */
    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    void    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::
    shiftRoot(message_type message) {
        TreeNode *newRoot = rootNode->unlinkChild(message);
        assert(newRoot != nullptr);
        currentPlayerDistribution = std::move(rootNode->otherPlayerDistribution);
        delete(rootNode);
        rootNode = newRoot;
    }


    // ===========================================================================
    // ======================= TREE NODE IMPLEMENTATION ==========================
    // ===========================================================================

    /** Given that the current actor in this node produced a given message, sample from the posterior
     * This is needed when training the off tree mind on actual observed opponent behaviour.
     *  TODO: this should use the prior distribution given the move history too!
     * @tparam BODY
     * @tparam OffTreeQFunction
     * @tparam SelfPlayPolicy
     * @param actorMessage
     * @return
     */
    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    const BODY *
    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::TreeNode::
    sampleActorBodyGivenMessage(message_type actorMessage) {
        double totalWeight = 0.0;
        std::vector<double>         cumulativeWeights;
        std::vector<const BODY *>   states;
        for(const auto &[body, qVec]: qEntries) {
            action_type maxQAct = sampleMaxQ(qEntries, body.legalActs());
            double w = body.actToMessageProb(maxQAct, actorMessage);
            totalWeight += w;
            cumulativeWeights.push_back(totalWeight);
            states.push_back(&body);
        }
        if(totalWeight == 0.0) { /* no state in this node could have produced this message */
            return nullptr;
        }
        // sample an element from cumulativeWeights with weighted prob
        double rand = deselby::Random::nextDouble(0.0, totalWeight);
        auto chosenIt = std::ranges::upper_bound(cumulativeWeights, rand);
        assert(chosenIt != cumulativeWeights.end());
        return states[chosenIt - cumulativeWeights.begin()];
    }



//    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
//    void IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::SelfPlayMind::halfStepObservationHook(const observation_type &body) {
//        if(treeNode != nullptr && !hasAddedQEntry) treeNode->otherPlayerDistribution[body]++;
//    }

    template<Body BODY, class OffTreeMind, class SelfPlayPolicy>
    IncompleteInformationMCTS<BODY, OffTreeMind, SelfPlayPolicy>::TreeNode *
    IncompleteInformationMCTS<BODY, OffTreeMind, SelfPlayPolicy>::TreeNode::getChildOrNull(message_type message) {
        auto childIt = children.find(message);
        return childIt==children.end()?nullptr:childIt->second;
    }


    /**
     *
     * @param message identifies the child to unlink
     * @return the unlinked child
     */
    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::TreeNode *
    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::TreeNode::unlinkChild(message_type message) {
        auto childIt = children.find(message);
        if(childIt != children.end()) {
            TreeNode *child = childIt->second;
            children.erase(childIt);
            return child;
        }
        return nullptr;
    }

    /** Make an entry for a new body state.
    *
    * @param agent
    * @return
    */
    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    auto
    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::TreeNode::addQEntry(const BODY &agent) {
        auto [newEntry, wasInserted] = qEntries.try_emplace(
                agent); // insert uniform quality with zero sample count
        assert(wasInserted);
        return newEntry;
    }


    /** Returns the child of this node corresponding to the given act,
    * creating a new one if necessary
    *
    * @param act the act that identifies the child
    * @return a pointer to the child
    */
    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::TreeNode *
    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::TreeNode::
    getChildOrCreate(message_type message) {
        auto [it, didInsert] = children.try_emplace(message, nullptr);
        if (didInsert) it->second = new TreeNode();
        return it->second;
    }



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

    /**
     *
     * @param visitCounts
     * @return
     */
    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy> template<bool LEAVETRACE, bool DOBACKPROP>
    auto /* horrible typename otherwise */
    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::
    createAgentSampler(const std::map<BODY,uint> &visitCounts) {
////        std::vector<double> weights;
//        if(otherPlayerDistribution.size() == 0) return {}; // undefined if no data
////        assert(otherPlayerDistribution.size() > 0);
//        states.reserve(visitCounts.size());
//        weights.reserve(visitCounts.size());
//        for(auto &entry: visitCounts) {
//            states.emplace_back(entry.first);
//            weights.emplace_back(entry.second);
//        }
        assert(!visitCounts.empty());
        std::vector<BODY>   states(std::ranges::views::keys(visitCounts));
        auto weights = std::ranges::views::values(visitCounts);
        auto distribution = std::discrete_distribution(weights.begin(), weights.end());

        SelfPlayMind<LEAVETRACE,DOBACKPROP> mind = SelfPlayMind<LEAVETRACE,DOBACKPROP>(*this);
        return [distribution, states, mind, &gen = randomGenerator]() {
            return Agent(states[distribution(gen)],mind);
        };
    }


//    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy> template<bool LEAVETRACE, bool DOBACKPROP>
//    auto /* horrible typename otherwise */
//    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::createAgentSampler(BODY body) {
//        return [agent = Agent(body,SelfPlayMind<LEAVETRACE,DOBACKPROP>(*this))]() {
//            return agent;
//        };
//    }


    // ===========================================================================
    // =========================== SELF PLAY MIND ================================
    // ===========================================================================

    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy> template<bool LEAVETRACE, bool DOBACKPROP>
    void    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::SelfPlayMind<LEAVETRACE,DOBACKPROP>::
    on(const events::IncomingMessage<message_type,BODY> &event) {
        if(isOnTree()) {
            if constexpr (DOBACKPROP) {
                if(hasAddedQEntry)
                    treeNode = treeNode->getChildOrNull(event.message);
                else
                    treeNode = treeNode->getChildOrCreate(event.message);
            } else {
                treeNode = treeNode->getChildOrNull(event.message);
            }
        }
    }


    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy> template<bool LEAVETRACE, bool DOBACKPROP>
    void    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::SelfPlayMind<LEAVETRACE,DOBACKPROP>::
    on(const events::Act<message_type,BODY> &event) {
        if(isOnTree()) {
            if constexpr (DOBACKPROP) {
                if(hasAddedQEntry)
                    treeNode = treeNode->getChildOrNull(event.message);
                else
                    treeNode = treeNode->getChildOrCreate(event.message);
            } else {
                treeNode = treeNode->getChildOrNull(event.message);
            }
            if constexpr (LEAVETRACE) treeNode->otherPlayerDistribution[event.body]++;
        }
        rewards.push_back(0.0); // new reward entry
    }


    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    template<bool LEAVETRACE, bool DOBACKPROP>
    template<class AGENT1, class AGENT2>
    void    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::SelfPlayMind<LEAVETRACE,DOBACKPROP>::
    on(const events::EndEpisode<AGENT1,AGENT2> &event) { // back propagate rewards
        if constexpr (DOBACKPROP) {
            double cumulativeReward = 0.0;
            while (rewards.size() > qValues.size()) {
                cumulativeReward = cumulativeReward * tree.discount + rewards.back();
                rewards.pop_back();
            }
            while (!rewards.empty()) {
                qValues.back()->addSample(cumulativeReward);
                cumulativeReward = cumulativeReward * tree.discount + rewards.back();
                rewards.pop_back();
                qValues.pop_back();
            }
        }
        // reset for next episode
        treeNode = tree.rootNode;
        qValues.clear();
        rewards.clear();
        hasAddedQEntry = false;
    }

    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    template<bool LEAVETRACE, bool DOBACKPROP>
    void    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::SelfPlayMind<LEAVETRACE,DOBACKPROP>::
    on(const events::Reward &event) {
        rewards.back() += event.value;
    }

    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy> template<bool LEAVETRACE,bool DOBACKPROP>
    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::action_type
    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::SelfPlayMind<LEAVETRACE,DOBACKPROP>::
    act(const observation_type &body, action_mask legalActs, IncompleteInformationMCTS::reward_type rewardFromLastAct) {
        QVector<action_type::size> *qEntry = nullptr; // if null, choose at random
        if(isOnTree()) {
            if (hasAddedQEntry) {
                // onTree but can't add new qEntry, so look for existing entry or null
                auto qIt = treeNode->qEntries.find(body);
                if(qIt != treeNode->qEntries.end()) qEntry = &(qIt->second.qVector);
            } else {
                auto [qIt, insertedNewEntry] = treeNode->qEntries.try_emplace(body);
                if (insertedNewEntry) {
                    hasAddedQEntry = true;
                }
                qEntry = &(qIt->second);
            }
        }
        action_type action;
        if(qEntry != nullptr) { // still on tree
            action = tree.selfPlayPolicy.sample(*qEntry, legalActs);
            if constexpr (DOBACKPROP) {
                rewards.push_back(rewardFromLastAct);
                qValues.push_back(&(*qEntry)[static_cast<size_t>(action)]);
            }
        } else { // off-tree
            action = tree.offTreeQMind(body);
            // static_cast<action_type>(sampleUniformly(legalActs));
            if constexpr (DOBACKPROP) rewards.push_back(rewardFromLastAct);
        }
        return action;
    }

    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy> template<bool LEAVETRACE,bool DOBACKPROP>
    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::action_type
    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::SelfPlayMind<LEAVETRACE,DOBACKPROP>::
    operator ()(const BODY &body) {
        QVector<action_type::size> *qEntry = nullptr;
        if(isOnTree()) {
            if (hasAddedQEntry) {
                // onTree but can't add new qEntry, so look for existing entry or null
                auto qIt = treeNode->qEntries.find(body);
                if(qIt != treeNode->qEntries.end()) qEntry = &(qIt->second.qVector);
            } else {
                auto [qIt, insertedNewEntry] = treeNode->qEntries.try_emplace(body);
                if (insertedNewEntry) {
                    hasAddedQEntry = true;
                }
                qEntry = &(qIt->second);
            }
        }
        action_type action;
        if(qEntry != nullptr) { // still on tree
            action = tree.selfPlayPolicy.sample(*qEntry, body.legalActs);
            if constexpr (DOBACKPROP) {
                qValues.push_back(&(*qEntry)[static_cast<size_t>(action)]);
            }
        } else { // off-tree
            action = tree.offTreeQMind(body);
        }
        return action;
    }

}
#endif //MULTIAGENTGOVERNMENT_INCOMPLETEINFORMATIONMCTS_H
