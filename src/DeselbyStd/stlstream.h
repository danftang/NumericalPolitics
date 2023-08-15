// Allows pretty printing of Standard Template Library containers and classes
//
// Created by daniel on 07/05/2021.
//


#ifndef DESELBY_STLSTREAM_H
#define DESELBY_STLSTREAM_H

#include <chrono>
#include <optional>
#include <utility>
#include <ostream>
#include <ranges>
#include <concepts>


template<typename I, intmax_t UNITS>
std::ostream &operator <<(std::ostream &out, const std::chrono::duration<I,std::ratio<1,UNITS>> &duration) {
    long double count = duration.count();
    intmax_t units = UNITS;
    while(count >= 1000.0 && units >= 1000) {
        count /= 1000.0;
        units /= 1000;
    }
    out << count;
    switch(units) {
        case 1: out << "s"; break;
        case 1000: out << "ms"; break;
        case 1000000: out << "μs"; break;
        case 1000000000: out << "ns"; break;
    }
    return out;
}


// Prior declarations to allow containers of pairs and pairs of containers etc...
template<typename T> std::ostream &operator <<(std::ostream &out, const std::optional<T> &optional);
template<typename T1, typename T2> std::ostream &operator <<(std::ostream &out, const std::pair<T1,T2> &pair);


template<std::ranges::range T> requires(!std::convertible_to<T,std::string>)
std::ostream &operator <<(std::ostream &out, const T &container) {
    out << '{';
    auto it = std::ranges::begin(container);
    if(it != std::ranges::end(container)) {
        out << *it;
        while (++it != container.end()) out << ", " << *it;
    }
    out << '}';
    return out;
}


template<std::ranges::range T> requires requires(T map) {
    typename T::key_type;
    typename T::mapped_type;
    { map.begin() } -> std::convertible_to<std::pair<const typename T::key_type, typename T::mapped_type>>;
}
std::ostream &operator <<(std::ostream &out, const T &container) {
    out << '{';
    auto it = std::ranges::begin(container);
    if(it != std::ranges::end(container)) {
        out << it->first << "->" << it->second;
        while (++it != container.end()) out << ", " << it->first << "->" << it->second;
    }
    out << '}';
    return out;
}


template<typename T>
std::ostream &operator <<(std::ostream &out, const std::optional<T> &optional) {
    if(optional.has_value()) out << optional.value(); else out << "undefined";
    return out;
}


template<typename T1, typename T2>
std::ostream &operator <<(std::ostream &out, const std::pair<T1,T2> &pair) {
    out << "(" << pair.first << ", " << pair.second << ")";
    return out;
}


#endif
