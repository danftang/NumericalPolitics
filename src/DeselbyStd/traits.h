//
// Created by daniel on 20/02/23.
//

#ifndef MULTIAGENTGOVERNMENT_TRAITS_H
#define MULTIAGENTGOVERNMENT_TRAITS_H
#include <vector>
#include <list>
#include <iterator>
#include <map>
#include <set>
#include <chrono>
#include <deque>
#include <forward_list>
#include <unordered_set>

namespace deselby {

    // True if T is an STL container
    template<class T>
    struct is_stl_container : public std::false_type {
    };
    template<class T>
    struct is_stl_container<std::vector<T>> : public std::true_type {
    };
    template<class T>
    struct is_stl_container<std::list<T>> : public std::true_type {
    };
    template<class T>
    struct is_stl_container<std::forward_list<T>> : public std::true_type {
    };
    template<class T>
    struct is_stl_container<std::deque<T>> : public std::true_type {
    };
    template<class T>
    struct is_stl_container<std::set<T>> : public std::true_type {
    };
    template<class T>
    struct is_stl_container<std::unordered_set<T>> : public std::true_type {
    };
    template<class T>
    struct is_stl_container<std::valarray<T>> : public std::true_type {
    };
    template<class T, size_t N>
    struct is_stl_container<std::array<T, N>> : public std::true_type {
    };
    template<class KEY, class VAL>
    struct is_stl_container<std::map<KEY, VAL>> : public std::true_type {
    };
    template<class KEY, class VAL>
    struct is_stl_container<std::multimap<KEY, VAL>> : public std::true_type {
    };
    template<class T> inline constexpr bool is_stl_container_v = is_stl_container<T>::value;


    // True if T is an associative STL container
    template<class T>
    struct is_stl_associative_container : public std::false_type {
    };
    template<class KEY, class VAL>
    struct is_stl_associative_container<std::map<KEY, VAL>> : public std::true_type {
    };
    template<class KEY, class VAL>
    struct is_stl_associative_container<std::multimap<KEY, VAL>> : public std::true_type {
    };
    template<class T> inline constexpr bool is_stl_associative_container_v = is_stl_associative_container<T>::value;


    // True if T is any kind of pair
    template<class T>
    struct is_pair: public std::false_type {};

    template<class A, class B>
    struct is_pair<std::pair<A,B>>: public std::true_type { };
    template<class T> inline constexpr bool is_pair_v = is_pair<T>::value;

}

#endif //MULTIAGENTGOVERNMENT_TRAITS_H
