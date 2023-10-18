//
// Created by daniel on 10/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_TESTS_FNN_H
#define MULTIAGENTGOVERNMENT_TESTS_FNN_H

#include "../abm/approximators/FNN.h"

namespace tests {


    void testFNN() {
        abm::approximators::FNN myFNN(2, mlpack::ConstInitialization(2.0), mlpack::Linear(2));

        class MyLossFunction {
        public:
            static constexpr size_t nPoints() { return 2; }
            void trainingSet(arma::mat &in) { in = arma::mat{1.0,2.0}.t(); }
            void gradientByPrediction(arma::mat &y, arma::mat &grad) { grad = arma::mat{1.0,1.0}.t(); }
        };


        auto grad = myFNN.gradientByParams(MyLossFunction());
//        std::cout << "output = " << out << std::endl;
        std::cout << "Gradient = " << grad << std::endl;

    }

}

#endif //MULTIAGENTGOVERNMENT_TESTS_FNN_H
