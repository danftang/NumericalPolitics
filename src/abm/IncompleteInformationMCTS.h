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
// AGENT must provide:
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

template<class AGENT>
class IncompleteInformationMCTS {
public:

    inline static std::default_random_engine randomGenerator = std::default_random_engine();

    class TreeNode {
    public:
        class QVector {
        public:
            std::vector<double> sumOfQ;
            std::vector<double> sumOfQSq;
            std::vector<int>    sampleCounts;

            QVector() :
            sumOfQ(AGENT::Action::size, 0.0),
            sumOfQSq(AGENT::Action::size, 0.0),
            sampleCounts(AGENT::Action::size, 0) { }


            void addSample(int actId, double cumulativeReward) {
                sumOfQ[actId] += cumulativeReward;
                sumOfQSq[actId] += cumulativeReward*cumulativeReward;
                ++sampleCounts[actId];
            }


            // calculates the best action according to UCT. i.e. the action with maximum
            // mean value + sqrt(ln(N)) standard errors in the mean, where N is the total
            // number of samples
            int getUCTAction() {
                int totalSamples = std::reduce(sampleCounts.begin(), sampleCounts.end());
                double nStandardErrors = sqrt(log(totalSamples));
                int bestActId = -1;
                double bestQ = -std::numeric_limits<double>::infinity();
                for(int actId = 0; actId < size(); ++actId) {
                    if(sampleCounts[actId] == 0) {
                        bestActId = actId;
                        break; // if there is an unexplored option, always explore.
                    }
                    double mean = sumOfQ[actId] / sampleCounts[actId];
                    double standardVarianceInMean = (sumOfQSq[actId] - pow(sumOfQ[actId], 2))
                                                        / pow(sampleCounts[actId], 3);
                    double Q = mean + nStandardErrors * sqrt(standardVarianceInMean);
                    if (Q >= bestQ) {
                        bestQ = Q;
                        bestActId = actId;
                    }
                }
                return bestActId;
            }

            size_t size() const { return sumOfQ.size(); }
        };


        ~TreeNode() {
            for(TreeNode *child : children) delete(child);
        }


        typedef std::map<AGENT, QVector>::iterator iterator;

        TreeNode *parent;   // if parent is nullptr, then we're at root
        int actId; // act associated with parent edge.
        std::map<AGENT, QVector> qEntries;   // (sample count, Q-valueSum) indexed by actId as a function of agent state
        std::vector<TreeNode *> children; // ...indexed by actId. if empty, then this is a leaf node.

        TreeNode(): TreeNode(nullptr,-1) {} // make root node

        TreeNode(TreeNode *parent, int actId):
        parent(parent),
        actId(actId) { }

        // Returns the childActId'th child, creating a new one if necessary
        TreeNode *getChildOrCreate(int childActId) {
            if(children.empty()) children.resize(AGENT::Action::size, nullptr);
            TreeNode *childNode = children[childActId];
            if(childNode == nullptr) {
                childNode = new TreeNode(this);
                children[childActId] = childNode;
            }
            return childNode;
        }


//        // add a uniform qEntry in the given child, adding a new TreeNode if necessary
//        void addChildQEntry(int treeNodeActId, AGENT qEntryState) {
//            if(children.empty()) children.resize(AGENT::Action::size, nullptr);
//            TreeNode *childNode = children[treeNodeActId];
//            if(childNode == nullptr) {
//                childNode = new TreeNode(this);
//                children[treeNodeActId] = childNode;
//            }
//            childNode->addQEntry(qEntryState);
//        }

        // cuts the tree between this node and the childActId'th child to create a new tree
        // returns the root of the new tree (i.e. the childActId'th child)
        TreeNode *prune(int childActId) {
            TreeNode newRoot = children[childActId];
            newRoot.parent = nullptr;
            children[childActId] = nullptr;
            return newRoot;
        }

        iterator addQEntry(const AGENT &agent) {
            auto [newEntry, wasInserted] = qEntries.try_emplace(agent); // insert uniform quality with zero sample count
            assert(wasInserted);
            return newEntry;
        }

        iterator randomEntry() {
            auto uniformDist = std::uniform_int_distribution<int>(0, qEntries.size() - 1);
            auto it = qEntries.begin();
            std::advance(it, uniformDist(randomGenerator));
            return it;
        }

        bool isRoot() { return parent == *this; }
    };


    // Represents a trajectory through the tree. This requires, in addition to the
    // final node, a record of which agent states (qEntries) within each node were visited.
    class Trajectory {
    public:
        TreeNode *endNode;
        std::vector<typename TreeNode::iterator> qEntries;
        std::vector<double> rewards;
        int endActId;

        Trajectory(TreeNode *rootNode): endNode(rootNode) {
            qEntries.push_back(rootNode->randomEntry());
            qEntries.push_back(rootNode->randomEntry()); // start with both players drawn from the prior
            endActId = qEntries.back()->second.getUCTAction();
        }

        TreeNode::iterator currentPlayerQEntry() { return qEntries.back(); }

        TreeNode::iterator otherPlayerQEntry() {
            return qEntries[qEntries.size()-2];
        }

        // choose an action for the currentPlayerQEntry.
//        int chooseNextAction() {
//            // The sample counts from the current qEntry come from different parents,
//            // so parentSampleCount has no meaning. The sum of counts for all acts in this qEntry
//            // gives the total number of samples through this state. Scale this by the
//            // proportion of samples from the current parent qEntry going into this treeNode.
//            // If we're at the root node, just give the sum of counts in this qEntry.
//            int parentSampleCount = endNode->isRoot()?
//                    currentPlayerQEntry()->second.sampleCounts.sum():
//                    ((int)currentPlayerQEntry()->second.sampleCounts.sum()*
//                    (otherPlayerQEntry()->second.sampleCounts[endNode->actId]*
//                    1.0/otherPlayerQEntry()->second.sampleCounts.sum()));
//            return chooseAction(currentPlayerQEntry()->second, parentSampleCount);
//        }

        // returns true if we're not at a trajectory leaf
        bool addMove() {
            if(endNode->children[endActId] == nullptr) return false; // we're at a tree leaf
            AGENT otherPlayer = otherPlayerQEntry()->first;
            double reward = otherPlayer.transition(endNode->actId, endActId);
            endNode = endNode->children(endActId);
            auto nextQEntry = endNode->qEntries.find(otherPlayer);
            if(nextQEntry == endNode->qEntries.end()) {
                endNode = endNode->parent; // roll back endNode
                return false; // no entry for state, we're at an internal trajectory leaf
            }
            qEntries.push_back(nextQEntry);
            rewards.push_back(reward);
            endActId = nextQEntry->second.getUCTAction();
            return true;
        }

        // take one step forward, creating a new qEntry in the underlying tree
        bool expandTreeOneMove() {
            AGENT otherPlayer = otherPlayerQEntry()->first;
            double reward = otherPlayer.transition(endNode->actId, endActId);
            endNode = endNode->getChildOrCreate(endActId);
            auto newQEntry = endNode->addQEntry(otherPlayer);
            qEntries.push_back(newQEntry);
            rewards.push_back(reward);
            endActId = newQEntry->second.getUCTAction();
        }

        // Selects a leaf node to expand by traversing from root using the Q values
        void moveToLeaf() {
            while(addMove());
        }



        // play out from the current end of the trajectory to the end of the episode
        // using random moves,
        // updates the final 2 entries of the rewards vector to be the cumulative
        // rewards to the end of the episode.
        void playOut(double discount) {
            std::uniform_int_distribution<int> uniformActDist(0, AGENT::Action::size-1);

            AGENT players[2] = { currentPlayerQEntry()->first, otherPlayerQEntry()->first };
            double cumulativeRewards[2] = {0.0, 0.0};
            int otherPlayerIndex = 1;
            int currentPlayerIndex = 0;
            int playerActIds[2] = {endActId, endNode->actId };
            double totalDiscount = 1.0;
            do {
                if(currentPlayerIndex == 0) totalDiscount *= discount;
                double reward = players[otherPlayerIndex].transition(playerActIds[otherPlayerIndex],
                                                                     playerActIds[currentPlayerIndex]);
                cumulativeRewards[otherPlayerIndex] += reward*totalDiscount;
                otherPlayerIndex = currentPlayerIndex;
                currentPlayerIndex = currentPlayerIndex ^ 1;
                playerActIds[currentPlayerIndex] = uniformActDist(randomGenerator);
            } while(!AGENT::isTerminal(playerActIds[otherPlayerIndex], playerActIds[currentPlayerIndex]));
            rewards.back() += cumulativeRewards[0];
            rewards[rewards.size()-2] += cumulativeRewards[1];
        }

        // Update each qEntry in this trajectory by incrementing the sample count and updating the Qvalue
        // in-line with the cumulative reward to the end of the episode.
        // Deletes the values in rewards as it goes.
        void backProp(double discount) {
            TreeNode *node = endNode;
            int actId = endActId;
            while(!rewards.empty()) {
                typename TreeNode::QVector &qEntry = qEntries[rewards.size() - 1]->second;
                qEntry.addSample(actId, rewards.back());
                if(rewards.size() > 2) rewards[rewards.size() - 3] += rewards.back()*discount;
                rewards.pop_back();
                actId = node->actId;
                node = node->parent;
            }
        }
    };

    TreeNode *              rootNode;
    std::function<AGENT()>  priorAgentSampler;

    IncompleteInformationMCTS(std::function<AGENT()> priorAgentSampler, int nRootSamples):
    rootNode(new TreeNode()),
    priorAgentSampler(priorAgentSampler) {
        // get samples for initial state
        for(int i=0; i<nRootSamples; ++i) {
            rootNode->addQEntry(priorAgentSampler());
        }
    }

    ~IncompleteInformationMCTS() {
        delete(rootNode);
    }

    void buildTree(int nSamples, double discount) {
        for(int s = 0; s < nSamples; ++s) {
            Trajectory sample(rootNode);
            sample.moveToLeaf();
            sample.expandTreeOneMove();
            sample.playOut(discount);
            sample.backProp(discount);
        }
    }

    // use UCT to choose which action to explore
//    static int chooseAction(const TreeNode::QVector &qVec, int parentNSamples) {
//        std::vector<double> actUCTScores(qVec.size());
//        for(int actId = 0; actId < qVec.size(); ++actId) {
//            actUCTScores[actId] = qVec.qValues[actId] + exploration*sqrt(log(parentNSamples)/qVec.sampleCounts[actId]);
//        }
//        return std::max_element(actUCTScores.begin(), actUCTScores.end()) - actUCTScores.begin();
//    }

};



#endif //MULTIAGENTGOVERNMENT_INCOMPLETEINFORMATIONMCTS_H
