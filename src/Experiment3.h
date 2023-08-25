// Three-player game where two players play prisoner's dilemma and one watches.
// The watcher can choose whether or not to punish players that defect,
// and so can modify the player's reward.
// All agents are strangers in a society of agents playing this game.
//
// So the act space is always of size 2 but interpreted as cooperate/defect when
// player or punish/don't punish when observer. The state space is simply
// oberver or player.
//
// The playing agents send their moves to the observer, then the observer sends the
// reward to players.
//
// Can the observer learn to punish defection and can this behaviour spread
// to all agents in the society? (i.e. the establishment of ethics as a social convention)?
//
// Once esteblished, how much punishment is actually necessary? (i.e. distinguish
// between the threat of punishment and the application of punishment)
//
// Clearly, if everyone punishes defection, it makes sense to cooperate, but if there
// is any cost to giving out punishment, then is there any motivation to punish?
//
// - What happens if all agents (including players) can punish?
//
// [If an agent can't distinguish the state it ends up in after making two different acts
// and gets the same imediate reward, it cannot distinguish between them. If the agent remembers
// whether it has been punished before, and whether it punished last time, this may be enough for it
// to realise that punishment leads to better personal outcomes...]
//
// Created by daniel on 12/04/23.
//

#ifndef MULTIAGENTGOVERNMENT_EXPERIMENT3_H
#define MULTIAGENTGOVERNMENT_EXPERIMENT3_H

#include <cstdlib>

#include "abm/V0.1/abm.h"
#include "abm/V0.1/agents/agents.h"

class ObservedPrisonersDilemmaAgent {
public:
    // States are
    // - just sonnected to oberver (need to make mve)
    // - just received player moves 00, 01, 10, 11 as observer (need to make reward - choose which point in 3rd reward dimension)
    // so 5 states
    static constexpr size_t NSTATES = 2; // number of possible outputs from the observation function
    static constexpr size_t NACTIONS = 2; // if player then 0=cooperate, 1=defect. If observer 0=no punishment, 1=punish
    static constexpr int punishment = -3; // enough to make it rational for the punished player to cooperate
    static constexpr int playerState = 4;


    typedef ulong time_type;
    typedef abm::QTablePolicy<NSTATES, NACTIONS> policy_type;
    typedef abm::Schedule<time_type> schedule_type;

    static constexpr double REWARD[2][2] = {{3, 0},
                                            {4, 1}};
    static constexpr double QMIN = 0;
    static constexpr double QMAX = REWARD[1][0] /
                                   (1.0 -
                                    policy_type::DEFAULT_DISCOUNT); // Value of Q if all future rewards are max reward

    policy_type policy;
    bool isObserver;
    int player1Move;
    int player2Move;
    abm::CommunicationChannel<abm::Schedule<time_type>, int> player1;
    abm::CommunicationChannel<abm::Schedule<time_type>, int> player2;
    abm::CommunicationChannel<abm::Schedule<time_type>, int> &observer = player1;

    // connect and send move immediately to observer
    void connectToObserverAsPlayer1(ObservedPrisonersDilemmaAgent &observerAgent, time_type time) {
        isObserver = false;
        observer.connectTo(observerAgent, &ObservedPrisonersDilemmaAgent::handlePlayer1Move, 1);
        int myMove = policy.getAction(0); // state is player
        observer.send(myMove, time);
    }

    void connectToObserverAsPlayer2(ObservedPrisonersDilemmaAgent &observerAgent, time_type time) {
        isObserver = false;
        observer.connectTo(observerAgent, &ObservedPrisonersDilemmaAgent::handlePlayer2Move, 1);
        int myMove = policy.getAction(0); // state is player
        observer.send(myMove, time);
    }

    void connectToPlayer1AsOberver(ObservedPrisonersDilemmaAgent &player1Agent) {
        isObserver = true;
        player1.connectTo(player1Agent, &ObservedPrisonersDilemmaAgent::handleObserverReward, 1);
    }

    void connectToPlayer2AsObserver(ObservedPrisonersDilemmaAgent &player2Agent) {
        player2.connectTo(player2Agent, &ObservedPrisonersDilemmaAgent::handleObserverReward, 1);
    }

//    schedule_type start() {
//        myLastMove = 0;
//        policy.trainingStepsSinceLastPolicyChange = 0;
//        policy.pExplore = policy.DEFAULT_INITIAL_EXPLORATION;
//        return opponent.send(myLastMove, 0);
//    }

    void setPolicy(long policyID) {
        policy.setPolicy(policyID, QMIN, QMAX);
    }

    // get reward and learn
    schedule_type handleObserverReward(int reward, time_type time) {
        policy.train();
        return schedule_type();
    }

    schedule_type handlePlayer1Move(int playerMove, time_type time) {
        if(player1Move >=0 && player2Move >=0) player2Move = -1; // reset from last observer moves
        player1Move = playerMove;
        return player2Move>=0?chooseRewards(time):schedule_type();
    }

    schedule_type handlePlayer2Move(int playerMove, time_type time) {
        if(player1Move >=0 && player2Move >=0) player1Move = -1; // reset from last observer moves
        player2Move = playerMove;
        return player1Move>=0?chooseRewards(time):schedule_type();
    }

    schedule_type chooseRewards(time_type time) {
        int state = player1Move*2 + player2Move;
        bool punishReward = policy.getAction(state)>0;
        int player1Reward = REWARD[player1Move][player2Move] - ((punishReward && player1Move==1)?punishment:0);
        int player2Reward = REWARD[player2Move][player1Move] - ((punishReward && player2Move==1)?punishment:0);
        return player1.send(player1Reward, time) + player2.send(player2Reward, time);
    }

};


inline void experiment3() {
    abm::agents::SequentialPairingAgent<AGENT> rootAgent(startPopulatinon);
    while (rootAgent.agents.size() <= endPopulation) {
        std::cout << "Population of " << std::dec << rootAgent.agents.size() << " agents:" << std::endl;
        typename AGENT::schedule_type sim = rootAgent.start();
        sim.execUntil([&sim, &rootAgent, burninTimesteps]() {
            return sim.time() >= burninTimesteps;
        });
        std::cout << "  Policy population at end of burnin: " << std::hex << rootAgent.getPopulationByPolicy() << std::endl;
        rootAgent.resetAllTrainingStats();
        sim.execUntil([&sim, &rootAgent, maxTimesteps = burninTimesteps + simTimesteps]() {
            return sim.time() >= maxTimesteps;
        });

        std::pair<double, double> wellbeingMeanSD = rootAgent.getRewardMeanAndSD();
        std::cout << "  Wellbeing " << wellbeingMeanSD.first << " +- " << wellbeingMeanSD.second << std::endl;
        std::cout << "  Policy population: " << std::hex << rootAgent.getPopulationByPolicy() << std::endl;
//            if(sim.time() < MAX_TIMESTEPS) {
//                std::cout << "converged to " << std::hex << sentinel.getPopulationByPolicy() << std::endl;
//            } else {
//                std::cout << "did not converge. Current population: " << std::hex << sentinel.getPopulationByPolicy() << std::endl;
//            }
        rootAgent.addQEntry();
    }

}

//    class ObservedPrisonersDilemmaInterface {
//    public:
//        typedef uint time_type;
//
//        static constexpr size_t NSTATES = 8; // number of possible outputs from the observation function
//        static constexpr size_t NACTIONS = 2;
//        static constexpr size_t BUFFER_SIZE = 64;
//        static constexpr double  REWARD[2][2] = {{3.0, 0.0},
//                                                {4.0, 1.0}};
//        static constexpr double PUNISHMENT[2] = {0.0, -5.0};
//
//        int     observerPhase = 0;
//
//        // public state
//        abm::ArrayChannel<bool,BUFFER_SIZE>                ownMoves;
//
//        // read channels
//        const abm::ArrayChannel<bool,BUFFER_SIZE> *leftNeighbourMoves; // read-end of the opponent's channel
//        const abm::ArrayChannel<bool,BUFFER_SIZE> *rightNeighbourMoves; // read-end of the opponent's channel
//
//        // observe opponent's move and return <reward,state> pair for Q-agent
//        int observe(time_type time) {
//            int state;
//            switch(phase(time)) {
//                case 0: // is observer
//                    state = 4 + 2*(*leftNeighbourMoves)[time-3] + (*rightNeighbourMoves)[time-3];
//                    break;
//                case 1: // play right neigbour
//                    state = 2*ownMoves[time-3] + (*rightNeighbourMoves)[time-3];
//                    break;
//                case 2: // play left neighbour
//                    state = 2*ownMoves[time-3] + (*leftNeighbourMoves)[time-3];
//            }
//            return state;
//        }
//
//
//        double reward(time_type time)  {
//            double reward;
//            switch(phase(time)) {
//                case 0: // was observer: no reward
//                    reward = 0.0;
//                    break;
//                case 1: // played right neighbour
//                    reward = REWARD[ownMoves[time-1]][(*rightNeighbourMoves)[time-1]];
//                    break;
//                case 2: // played left neighbour, punishment from right neighbour
//                    reward = REWARD[ownMoves[time-1]][(*leftNeighbourMoves)[time-1]] +
//                             PUNISHMENT[(*rightNeighbourMoves)[time-1]];
//                    ;
//            }
//            return reward;
//        }
//
//        void act(time_type time, int action) {
//            assert(action == 0 | action == 1);
//            ownMoves[time] = (action == 1);
//        }
//
//
//        inline int phase(time_type time) {
//            return (time-observerPhase)%3==0;
//        }
//    };
//
//    typedef abm::QAgent<abm::TabularQPolicy<ObservedPrisonersDilemmaInterface>, ObservedPrisonersDilemmaInterface> ObservedPrisonersDilemmaAgent;


//    template<int MAX_TIME>
//    class RoundRobinPrisonersDilemmaView {
//    public:
//        static constexpr size_t BUFFER_SIZE = 100;
//        static constexpr size_t NSTATES = 4;
//        static constexpr size_t NACTIONS = 2;
//        static constexpr double  REWARD[2][2] = {{3, 0},
//                                                {4, 1}};
//
//        // public state
//        abm::ArrayChannel<bool,BUFFER_SIZE>                ownMoves;
//
//        // constant public state
//        RoundRobinPrisonersDilemmaView * nextAgentView; // next channel in ordering
//
//        // internal state
//        RoundRobinPrisonersDilemmaView *opponent;
//
//        std::pair<double,int> observe(size_t time) {
//            int opponentsMove = (opponent->ownMoves)[time-1];
//            int ownMove = ownMoves[time - 1];
//            double r = REWARD[ownMove][opponentsMove];
//            int qstate = 2*ownMove + opponentsMove;
//            return {r, qstate};
//        }
//
//        void act(int action, size_t time) {
//            assert(action == 0 | action == 1);
//            ownMoves[time] = (action == 0);
//            opponent = opponent->nextAgentView;
//        }
//
//    protected:
//    };
//



#endif //MULTIAGENTGOVERNMENT_EXPERIMENT3_H
