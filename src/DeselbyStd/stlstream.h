// Allows pretty printing of Standard Template Library containers and classes
//
// Created by daniel on 07/05/2021.
//


#ifndef DESELBY_STLSTREAM_H
#define DESELBY_STLSTREAM_H

#include <ostream>
#include <vector>
#include <list>
#include <iterator>
#include <map>
#include <set>
#include <chrono>
#include <deque>
#include <forward_list>
#include <unordered_set>
#include <tuple>
#include <ranges>
#include <valarray>
#include <ranges>
#include "typeutils.h"


template<typename I, intmax_t UNITS>
inline std::ostream &operator <<(std::ostream &out, const std::chrono::duration<I,std::ratio<1,UNITS>> &duration) {
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

template<class T, T...indices>
inline std::ostream &operator<<(std::ostream &out, std::integer_sequence<T, indices...> /**/) {
    out << "< ";
    ((out << indices << " "), ...);
    out << ">";
    return out;
}



// Prior declarations to allow recursion both ways (e.g. containers of pairs and pairs of containers)
template<typename T> std::ostream &operator <<(std::ostream &out, const std::optional<T> &optional);
template<typename T1, typename T2> std::ostream &operator <<(std::ostream &out, const std::pair<T1,T2> &pair);
template<class... T> std::ostream &operator <<(std::ostream &out, const std::tuple<T...> &tuple);


namespace deselby {
    /** If a range has no operator << then it will be cast to a RangePrintAdaptor
     * N.B. we need type deletion as the cast won't occur if this is templated */
    class RangePrintAdaptor {
    public:
        std::function<std::ostream &(std::ostream &)> print;

        template<std::ranges::range CONTAINER>
        RangePrintAdaptor(const CONTAINER &container) : print([& container](std::ostream &out) -> std::ostream &{
            out << "{";
            auto it = std::begin(container);
            if(it != std::end(container)) {
                out << *it;
                while (++it != std::end(container)) out << ", " << *it;
            }
            out << "}";
            return out;
        }) {}
    };
}

inline std::ostream &operator <<(std::ostream &out, const deselby::RangePrintAdaptor &container) {
    return container.print(out);
}


template<class T, typename = std::enable_if_t<deselby::is_stl_associative_container_v<T>>>
inline std::ostream &operator <<(std::ostream &out, const T &associativeContainer) {
    out << "{";
    auto it = std::begin(associativeContainer);
    if(it != std::end(associativeContainer)) {
        out << it->first << "->" << it->second;
        while (++it != std::end(associativeContainer)) out << ", " << it->first << "->" << it->second;
    }
    out << "}";
    return out;
}



//template<class T, typename = std::enable_if_t<deselby::is_stl_container_v<T>>>
//inline std::ostream &operator <<(std::ostream &out, const T &container) {
//    out << "{";
//    auto it = std::begin(container);
//    if(it != std::end(container)) {
//            if constexpr (deselby::is_stl_associative_container_v<T>) {
//                out << it->first << "->" << it->second;
//                while (++it != container.end()) out << ", " << it->first << "->" << it->second;
//            } else {
//                out << *it;
//                while (++it != container.end()) out << ", " << *it;
//            }
//    }
//    out << "}";
//    return out;
//}


//template<typename T>
//inline std::ostream &operator <<(std::ostream &out, const std::valarray<T> &vec) {
//    out << "{";
//    for(int i=0; i<vec.size()-1; ++i) out << vec[i] << ", ";
//    if(vec.size() > 0) out << vec[vec.size()-1];
//    out << "}";
//    return out;
//}


template<typename T>
inline std::ostream &operator <<(std::ostream &out, const std::optional<T> &optional) {
    if(optional.has_value()) out << optional.value(); else out << "undefined";
    return out;
}


template<typename T1, typename T2>
inline std::ostream &operator <<(std::ostream &out, const std::pair<T1,T2> &pair) {
    out << "(" << pair.first << ", " << pair.second << ")";
    return out;
}


template<class H, class... T>
inline std::ostream &operator <<(std::ostream &out, const std::tuple<H, T...> &tuple) {
    out << "(" << std::get<0>(tuple);
    [&tuple, &out]<size_t...indices>(std::index_sequence<indices...>) {
        ((out << ", " << std::get<indices+1>(tuple)),...);
    }(std::make_index_sequence<sizeof...(T)>());
    out << ")";
    return out;
}

inline std::ostream &operator <<(std::ostream &out, const std::tuple<> & /*emptyTuple*/) {
    out << "()";
    return out;
}



#endif
