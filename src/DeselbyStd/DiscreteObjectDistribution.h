//
// Created by daniel on 07/11/23.
//

#ifndef MULTIAGENTGOVERNMENT_DISCRETEOBJECTDISTRIBUTION_H
#define MULTIAGENTGOVERNMENT_DISCRETEOBJECTDISTRIBUTION_H

#include <vector>
#include <map>
#include <random>
#include <ranges>

/** A distribution over a set of objects of type T. */
template<class T>
class DiscreteObjectDistribution {
public:
    std::discrete_distribution<size_t> discreteDistribution;
    std::vector<T> objects;


    template<std::ranges::sized_range OBJECTS, std::ranges::range PROBABILITIES>
    DiscreteObjectDistribution(const OBJECTS &objectRange, const PROBABILITIES &probabilities) {
        discreteDistribution.param(std::discrete_distribution<size_t>::param_type(probabilities.begin(), probabilities.end()));
        objects.reserve(objectRange.size());
        objects.insert(objects.begin(), objectRange.begin(), objectRange.end());
    }

    template<class KEY, class VAL>
    DiscreteObjectDistribution(const std::map<KEY,VAL> &map) :
        DiscreteObjectDistribution(std::views::keys(map), std::views::values(map)) {
            std::map<int,double> m;
            std::ranges::range_value_t<decltype(std::views::keys(m))> i;
    }

    template<class RNG = decltype(deselby::random::gen)>
    std::unwrap_reference_t<T> &operator()(RNG &randomNumberGenerator = deselby::random::gen) {
        return objects.at(discreteDistribution(randomNumberGenerator));
    }
};

template<class KEY, class VAL>
DiscreteObjectDistribution(const std::map<KEY,VAL> &map) -> DiscreteObjectDistribution<KEY>;

template<std::ranges::sized_range OBJECTS, std::ranges::range PROBABILITIES>
DiscreteObjectDistribution(const OBJECTS &objects, const PROBABILITIES &probabilities) -> DiscreteObjectDistribution<std::ranges::range_value_t<OBJECTS>>;


#endif //MULTIAGENTGOVERNMENT_DISCRETEOBJECTDISTRIBUTION_H
