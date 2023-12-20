#include "mlpack.hpp"

//#include "abm/approximators/FNN.h"
//#include "abm/Concepts.h"
//#include "abm/lossFunctions/SumOfLosses.h"

//#include "abm/agents/agents.h"
//#include "Experiment1.h"
//#include "Experiment2.h"
//#include "Experiment5.h"

#include "Experiment6.h"

//#include "abm/societies/RandomEncounterSociety.h"
// #include "abm/PingPongAgent.h"

//#include "tests/PrisonersDilemma.h"
//#include "tests/tests.h"
//#include "tests/mlpacktests.cpp"
//#include "tests/DQNtest.cpp"
//#include "tests/IncompleteInformationMCTSTest.cpp"
//#include "tests/FNN.h"

//#include "tests/mlpacktests.cpp"

//#include "abm/societies/RandomEncounterSociety.h"


int main() {

//  --- EXPERIMENTS
////    experiment1();
//    experiment2a();
////    experiment2b();
//    experiment5a();
//    experiment5b();

//    experiment5::iimctsSugarSpice();

    experiment6::qLearningGuessTheNumber();

    // ---- TESTS
//    cartPoleDQNTest();
//    tests::DQNCartPole();
//    pingPongTest();

//    tests::incompleteInformationMCTSTest();

//    abm::societies::RandomEncounterSociety mySociety(abm::PingPongAgent{}, abm::PingPongAgent{});
//    mySociety.run(2, abm::callbacks::Verbose());
//    tests::zeroIntelligencePrisonersDilemma();
//    tests::tabularQMindPrisonersDilemma();

//    tests::testFNN();

//    tests::DQNCartPole();

//    mlpack::RandomSeed(52386);
//    cartPoleDQNTest();


    return 0;
}
