#include "mlpack.hpp"
//#include "abm/agents/agents.h"
//#include "Experiment1.h"
//#include "Experiment2.h"
#include "Experiment5.h"
//#include "tests/tests.h"

#include "abm/DQN.h"
#include "tests/mlpacktests.cpp"
#include "tests/DQNtest.cpp"

int main() {
//  --- EXPERIMENTS
////    experiment1();
//    experiment2a();
////    experiment2b();
//    experiment5a();
//    experiment5b();

    // ---- TESTS
//    cartPoleDQNTest();
    tests::DQNCartPole();
//    pingPongTest();

    return 0;
}
