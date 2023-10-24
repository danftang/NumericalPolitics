//
// Created by daniel on 23/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_OPTIONALDOUBLE_H
#define MULTIAGENTGOVERNMENT_OPTIONALDOUBLE_H

#include <cmath>
#include <optional>
#include <cstdint>
#include <ostream>

namespace deselby {
    /** An alternative to std::optional<double> that uses one of the the NaN values to denote unset.
     *  N.B. NaN is not a unique value, and the NaN used for empty is distinct from the NaN produced
     *  by arithmetic operations.
     */
    union OptionalDouble {
    protected:
        double asDouble; // the union with uint64 allows us to distinguish between different NaNs.
        uint64_t asInt;

        static const OptionalDouble empty;
    public:
        OptionalDouble(const OptionalDouble &other) : asInt(other.asInt) {}

        OptionalDouble(const double &val) : asDouble(val) {}

        OptionalDouble(const std::nullopt_t &/* empty */) : OptionalDouble() {}

        OptionalDouble() : asInt(empty.asInt) {}

        bool has_value() const { return asInt != empty.asInt; }

        void reset() { asInt = empty.asInt; }

        double &value() { return asDouble; }

        const double &value() const { return asDouble; }

        double value_or(const double &defaultVal) const { return has_value() ? value() : defaultVal; }

        operator double() const {
            if (!has_value()) throw (std::bad_optional_access());
            return asDouble;
        };


        operator bool() { return has_value(); }

        double *operator->() { return &value(); }

        const double *operator->() const { return &value(); }

        double &operator*() { return value(); }

        const double &operator*() const { return value(); }

        OptionalDouble &operator=(const double &dVal) {
            asDouble = dVal;
            return *this;
        }

        OptionalDouble &operator=(const std::nullopt_t &) {
            reset();
            return *this;
        }


        friend std::ostream &operator<<(std::ostream &out, const OptionalDouble &optDouble) {
            if (optDouble.has_value()) out << optDouble.asDouble; else out << "empty";
            return out;
        }
    };


    inline const OptionalDouble OptionalDouble::empty(std::nan("0x656d707479")); // spells "empty" in ascii.
}



#endif //MULTIAGENTGOVERNMENT_OPTIONALDOUBLE_H
