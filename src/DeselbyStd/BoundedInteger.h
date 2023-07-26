//
// Created by daniel on 26/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_BOUNDEDINTEGER_H
#define MULTIAGENTGOVERNMENT_BOUNDEDINTEGER_H

#include <utility>
#include <cassert>
#include <limits>
#include <concepts>

namespace deselby {

    template<class STORAGE, STORAGE MIN, STORAGE MAX>
    class BoundedInteger {
    public:
        STORAGE v;
        static constexpr STORAGE min = MIN;
        static constexpr STORAGE max = MAX;
        static constexpr STORAGE size = MAX - MIN + 1;

        BoundedInteger(): BoundedInteger(MIN) { }

        BoundedInteger(const STORAGE &value) : v(value) {
            if(value < MIN || value > MAX) throw(std::out_of_range("Bounded integer initialised with out of range value"));
        }

//        BoundedInteger(STORAGE &&value) : v(std::move(value)) {
//            if(value < MIN || value > MAX) throw(std::out_of_range("Bounded integer initialised with out of range value"));
//        }

//        template<class OTHERSTORAGE, OTHERSTORAGE OTHERMIN, OTHERSTORAGE OTHERMAX>
//        BoundedInteger(const BoundedInteger<STORAGE, OTHERMIN, OTHERMAX> &other)
//        requires(OTHERMIN >= MIN && OTHERMAX <= MAX && std::is_convertible<OTHERSTORAGE,STORAGE>()) {
//            v = other.v;
//        }

//        operator const STORAGE &() const { return v; }

        operator STORAGE () const { return v; }

    };

}
#endif //MULTIAGENTGOVERNMENT_BOUNDEDINTEGER_H
