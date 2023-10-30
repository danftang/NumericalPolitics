//
// Created by daniel on 27/09/23.
//

#ifndef POLYMORPHICCOLLECTION_TYPEUTILS_H
#define POLYMORPHICCOLLECTION_TYPEUTILS_H

#include <cstdlib>
#include <utility>
#include <variant>
#include <vector>
#include <list>
#include <iterator>
#include <map>
#include <set>
#include <deque>
#include <forward_list>
#include <unordered_set>
#include <tuple>
#include <ranges>
#include <valarray>

namespace deselby {

    /** true if T is a type of the from TEMPLATE<class...> for some set of classes */
    template<class T, template<class...> class TEMPLATE>
    struct is_specialization_of {
        static constexpr bool value = false;
    };

    template<template<class...> class TEMPLATEDCLASS, class...TEMPLATES>
    struct is_specialization_of<TEMPLATEDCLASS<TEMPLATES...>, TEMPLATEDCLASS> {
        static constexpr bool value = true;
    };

    template<template<class...> class TEMPLATEDCLASS, class...TEMPLATES>
    struct is_specialization_of<const TEMPLATEDCLASS<TEMPLATES...>, TEMPLATEDCLASS> {
        static constexpr bool value = true;
    };

    template<class T, template<class...> class TEMPLATE>
    static constexpr bool is_specialization_of_v = is_specialization_of<T, TEMPLATE>::value;

    template<class T, template<class...> class TEMPLATE>
    concept IsSpecializationOf = is_specialization_of_v<T, TEMPLATE>;

//    template<class T> inline constexpr bool is_pair_v = is_specialization_of_v<T,std::pair>;
//    template<class T> inline constexpr bool is_optional_v = is_specialization_of_v<T,std::optional>;
//    template<class T> inline constexpr bool is_tuple_v = is_specialization_of_v<T,std::tuple>;

    /** ConvertsToTemplateType is true if a type can be used to call to a function that takes that template type,
     * templated on any templates */
    template<template<class...> class TEMPLATE, class... TS>
    TEMPLATE<TS...> is_uniquely_convertible_to_template_helper(TEMPLATE<TS...> objToTest) { return objToTest; };


    template<class T, template<class...> class TEMPLATE>
    concept IsUniquelyConvertibleToTemplate = requires(T obj) {
        is_uniquely_convertible_to_template_helper<TEMPLATE>(obj); };

    template<class T, template<class...> class TEMPLATE>
    using ConvertsToTemplateType = decltype(is_uniquely_convertible_to_template_helper(std::declval<T>()));

    /** Expresses a number as a sequence of digits in a given base
     *
     * @tparam N        the number to express
     * @tparam digits   how many digits to expand
     * @tparam base     the base to use in expansion
     * @tparam suffix   an optional suffix which will be added to the end of the expansion
     */
    template<size_t N, size_t digits, size_t base, size_t...suffix>
    struct base_n_expansion : base_n_expansion<N / base, digits - 1, base, N % base, suffix...> {
    };

    template<size_t N, size_t base, size_t... suffix>
    struct base_n_expansion<N, 1, base, suffix...> {
        using type = typename std::index_sequence<N % base, suffix...>;
    };

    template<size_t N, size_t digits, size_t base>
    using base_n_expansion_t = typename base_n_expansion<N, digits, base>::type;

    /** True if T is a member of OTHERS */
    template<class T, class... OTHERS>
    concept IsIn = (std::same_as<std::remove_cvref_t<T>, std::remove_cvref_t<OTHERS>> || ...);

    /** AlldifferentAndNotEmpty<class...TS> is true if all classes TS... are distinct and TS is not empty */
    template<class T, class...OTHERS> requires(!IsIn<T, OTHERS...>)
    struct AllDifferentHelper : AllDifferentHelper<OTHERS...> {
    };

    template<class T>
    struct AllDifferentHelper<T> {
    };

    template<class... CLASSES>
    concept AllDifferentAndNotEmpty = requires() { AllDifferentHelper<CLASSES...>(); };

    /** AllSameAndNotEmpty<class...TS> is true if all classes TS... are the same and TS is not empty */
    template<class T, class... CLASSES>
    concept AllSameAndNotEmpty = (std::same_as<T,CLASSES> && ...);

    /** Turns any reference into a reference_wrapper, otherwise returns the original type */
    template<class T>
    using wrap_if_reference = std::conditional_t<
            std::is_reference_v<T>,
            std::reference_wrapper<std::remove_reference_t<T>>,
            T>;


    /** Turn a tuple type into a variant of reference_wrappers to the elements of the tuple */
    template<class TUPLE>
    struct reference_variant_type;
    template<class... TELEMENTS>
    struct reference_variant_type<std::tuple<TELEMENTS...>> {
        using type = std::variant<std::reference_wrapper<std::remove_reference_t<TELEMENTS>>...>;
    };
    template<class... TELEMENTS>
    struct reference_variant_type<const std::tuple<TELEMENTS...>> {
        using type = std::variant<std::reference_wrapper<const std::remove_reference_t<TELEMENTS>>...>;
    };
    template<class TUPLE>
    using reference_variant_type_t = typename reference_variant_type<TUPLE>::type;


    /** Turn a tuple type into a variant type with the same elements, but wrapping any referenes in reference_wrapper */
    template<class TUPLE>
    struct variant_type;
    template<class... TELEMENTS>
    struct variant_type<std::tuple<TELEMENTS...>> {
        using type = std::variant<wrap_if_reference<TELEMENTS>...>;
    };
    template<class... TELEMENTS>
    struct variant_type<const std::tuple<TELEMENTS...>> {
        using type = std::variant<wrap_if_reference<const TELEMENTS>...>;
    };
    template<class TUPLE>
    using variant_type_t = typename variant_type<TUPLE>::type;

    /** Provides a compile-time x^y for integer x and unsigned integer y */
    template<std::integral T>
    consteval T powi(T base, uint index) {
        if (index == 2) return base * base;
        if (index == 0) return 1;
        return (index & 1) ? powi(powi(base, index / 2), 2) * base : powi(powi(base, index / 2), 2);
    }

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



    /** Invokes a function if it is invokable, otherwise does nothing.
     *
     * If you want a return value, use invoke_or to specify a default return value
     * if func is not invokable. (N.B. the return type must be known at compile time.
     * If the return type is dependent on the argument types, either deal with the results
     * inside func or return a std::variant).
     *
     * N.B. a function is invocable if its declaration type requirements are met on the
     * first compiler pass, so cannot be used to filter compilability of the function
     * definition by type. Use constexpr_if or add 'requires' clause to the declaration
     * instead.
     */
    template<class FUNC, class... ARGS>
    inline void invoke_if_invocable(FUNC && /* func */, ARGS &&... /* args */) { }

    template<class FUNC, class... ARGS> requires std::is_invocable_v<FUNC,ARGS...>
    inline void invoke_if_invocable(FUNC &&func, ARGS &&...args) {
        std::invoke(std::forward<FUNC>(func), std::forward<ARGS>(args)...);
    }

    /** Invokes a function if it is invocable, returning its result, otherwise returns a
     * default value.
     *
     * N.B. a function is invocable if its declaration type requirements are met on the
     * first compiler pass, so cannot be used to filter compilability of the function
     * definition by type. Use constexpr_if or add 'requires' clause to the declaration
     * instead.
     */
    template<class FUNC, class RETURN, class... ARGS>
    inline RETURN invoke_or(FUNC && /* func */, RETURN &&defaultValue, ARGS &&... /* args */) {
        return std::forward<RETURN>(defaultValue);
    }

    template<class FUNC, class RETURN, class... ARGS> requires std::is_invocable_v<FUNC,ARGS...>
    inline auto invoke_or(FUNC &&func, RETURN && /* defaultValue */, ARGS &&...args) {
        return std::invoke(std::forward<FUNC>(func), std::forward<ARGS>(args)...);
    }

//
    /** invokes a function if it is invokable, or else it executes an alternative with no arguments
     * The else function should be unconditionally executable, so any args can be lambda captured.
     * If necessary, you can nest another try_invoke inside the else function to make it unconditionally
     * executable */
    template<class FUNC, class ORELSE, class... ARGS> requires std::is_invocable_v<FUNC, ARGS...>
    inline auto invoke_else(FUNC &&invokeFunc, ARGS &&...args, ORELSE && /*orElseRunnable*/) {
        return std::invoke(std::forward<FUNC>(invokeFunc),std::forward<ARGS>(args)...);
    }

    template<class FUNC, class ORELSE, class... ARGS> requires std::is_invocable_v<ORELSE>
    inline auto invoke_else(FUNC && /*invokeFunc*/, ARGS &&.../*args*/, ORELSE &&orElseRunnable) {
        return std::invoke(orElseRunnable);
    }

    /** Static equivalent to (b ? x : y) only x and y don't have to be of the same type
     */
    template<bool CONDITION, class IFTRUE, class IFFALSE> requires CONDITION
    inline IFTRUE choose(IFTRUE &&valueIfTrue, IFFALSE && /* valueIfFalse */) {
        return std::forward<IFTRUE>(valueIfTrue);
    }

    template<bool CONDITION, class IFTRUE, class IFFALSE>
    inline IFFALSE choose(IFTRUE && /* valueIfTrue */, IFFALSE &&valueIfFalse) {
        return std::forward<IFFALSE>(valueIfFalse);
    }


    /** Invokes a function with given arguments if a condition is true.
     *  This replaces
     *  if constexpr(..) {...}
     *  but without the requirement that the expression be compilable
     *  if the constexpr is false.
     *
     *  To use this you must:
     *      - identify the operations that may not be compilable (depending on type assignment)
     *      - identify a set of "type-sensitive" obejcts such that each of these operations involves at least one
     *        member of this set.
     *      - pass a the type-sensitive objects into the function as templated or auto arguments.
     *      - all other objects can be captured.
     *  If in doubt, just pass all objects as auto arguments.
     *
     *  The functionality of if...else can be created using:
     *  constexpr_if<Condition>(...)
     *  constexpr_if<!Condition>(...)
     *  with no runtime overhead.
     */
    template<bool CONDITION, class FUNC, class... ARGS>
    inline void constexpr_if(FUNC && /* function */, ARGS &&... /* arguments */) { }

    template<bool CONDITION, class FUNC, class... ARGS> requires CONDITION
    inline auto constexpr_if(FUNC &&func, ARGS &&...args) {
        return std::invoke(std::forward<FUNC>(func),std::forward<ARGS>(args)...);
    }




    /** True if T can be streamed to a std::ostream using the << operator */
//    template<class T>
//    concept IsStreamable = requires(std::ostream out, T obj) { out << obj; };
    template<class T, class STREAM = std::ostream>
    concept HasInsertStreamOperator = requires(STREAM out, T obj) { out << obj; };

    /** useful lambda for referring to the overloaded stream operator
     *  is not invocable if not streamable */
    constexpr auto streamoperator = []<class S, HasInsertStreamOperator<S> T>(S &out, T &&obj) -> S & { return out << std::forward<T>(obj); };


    /** A simple box that allows native types (such as double) to be used as template parameters
     * which seems not to be supported (as of 2023, clang gives
     * error: sorry, non-type template argument of type 'double' is not yet supported)
     * But ConstExpr<double> is fine!
     */
    template<class T> struct ConstExpr {
        const T value;
        consteval ConstExpr(T val) : value(std::move(val)) {}
        consteval operator const T &() const { return value; }
    };


    /** use call_signature<T> to extract information from call signatures and function pointers */
    namespace impl {
        template<class SIG>
        struct call_signature_helper {
            static constexpr bool not_a_valid_call_signature = true;
        };

        template<class OUTPUT, class... ARGS>
        struct call_signature_helper<OUTPUT(ARGS...)> {
            typedef OUTPUT return_type;

            template<uint N>
            using arg_type = std::tuple_element_t<N, std::tuple<ARGS...>>;

            static constexpr uint n_args = sizeof...(ARGS);
        };

        template<class OUTPUT, class... ARGS>
        struct call_signature_helper<OUTPUT(*)(ARGS...)> {
            typedef OUTPUT return_type;

            template<uint N>
            using arg_type = std::tuple_element_t<N, std::tuple<ARGS...>>;

            static constexpr uint n_args = sizeof...(ARGS);
        };

        template<class OUTPUT, class OBJ, class... ARGS>
        struct call_signature_helper<OUTPUT(OBJ::*)(ARGS...)> {
            typedef OUTPUT return_type;
            typedef OBJ containing_class_type;

            template<uint N>
            using arg_type = std::tuple_element_t<N, std::tuple<ARGS...>>;

            static constexpr uint n_args = sizeof...(ARGS);
        };
    }

    template<class T> using call_signature = impl::call_signature_helper<std::remove_cv_t<T>>;
}

#endif //POLYMORPHICCOLLECTION_TYPEUTILS_H
