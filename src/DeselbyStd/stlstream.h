// Allows pretty printing of Standard Template Library containers and classes
//
// Created by daniel on 07/05/2021.
//


#ifndef DESELBY_STLSTREAM_H
#define DESELBY_STLSTREAM_H

#include <vector>
#include <list>
#include <iterator>
#include <map>
#include <set>
#include <chrono>
#include <deque>
#include <forward_list>
#include <unordered_set>
#include <ostream>
#include "traits.h"


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


// Prior declarations to allow recursion in containers
template<typename T> std::ostream &operator <<(std::ostream &out, const std::optional<T> &optional);
template<typename T> std::ostream &operator <<(std::ostream &out, const std::valarray<T> &vec);
template<typename T1, typename T2> std::ostream &operator <<(std::ostream &out, const std::pair<T1,T2> &pair);


template<class T, typename = std::enable_if_t<deselby::is_stl_container_v<T>>>
std::ostream &operator <<(std::ostream &out, const T &container) {
    out << "{";
    auto it = container.begin();
    if(it != container.end()) {
            if constexpr (deselby::is_stl_associative_container_v<T>) {
                out << it->first << "->" << it->second;
                while (++it != container.end()) out << ", " << it->first << "->" << it->second;
            } else {
                out << *it;
                while (++it != container.end()) out << ", " << *it;
            }
    }
    out << "}";
    return out;
}


template<typename T>
std::ostream &operator <<(std::ostream &out, const std::valarray<T> &vec) {
    out << "{";
    for(int i=0; i<vec.size()-1; ++i) out << vec[i] << ", ";
    if(vec.size() > 0) out << vec[vec.size()-1];
    out << "}";
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
