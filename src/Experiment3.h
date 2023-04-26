//
// Created by daniel on 12/04/23.
//

#ifndef MULTIAGENTGOVERNMENT_EXPERIMENT3_H
#define MULTIAGENTGOVERNMENT_EXPERIMENT3_H


namespace experiment3 {
//    class ObservedPrisonersDilemmaInterface {
//    public:
//        typedef uint time_type;
//
//        static constexpr size_t NSTATES = 8; // number of possible outputs from the observation function
//        static constexpr size_t NACTIONS = 2;
//        static constexpr size_t BUFFER_SIZE = 64;
//        static constexpr float  REWARD[2][2] = {{3.0, 0.0},
//                                                {4.0, 1.0}};
//        static constexpr float PUNISHMENT[2] = {0.0, -5.0};
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
//        float reward(time_type time)  {
//            float reward;
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
//        static constexpr float  REWARD[2][2] = {{3, 0},
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
//        std::pair<float,int> observe(size_t time) {
//            int opponentsMove = (opponent->ownMoves)[time-1];
//            int ownMove = ownMoves[time - 1];
//            float r = REWARD[ownMove][opponentsMove];
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


};


#endif //MULTIAGENTGOVERNMENT_EXPERIMENT3_H
