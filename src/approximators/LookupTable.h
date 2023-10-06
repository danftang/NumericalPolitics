//
// Created by daniel on 06/09/23.
//

#ifndef MULTIAGENTGOVERNMENT_LOOKUPTABLE_H
#define MULTIAGENTGOVERNMENT_LOOKUPTABLE_H

#include <ostream>
#include <vector>


namespace events {
    template<class INPUT, class OUTPUT>
    struct IOPair {
        INPUT   input;
        OUTPUT  output;
    };
}

/** A function approximator from size_t to any output type.
 * The function can be trained on Input/Evidence pairs.
 * On an IOPair event the table entry is updated according to
 * table[input] = update(table[input], evidence);
 * where update is the supplied update function.
 * Evidence can be any type that update can handle, but
 * is often the same as the table output, i.e. an I/O pair,
 * and the update func is some averaging function.
 *
 */
template<class OUTPUT, class UPDATEFUNC>
class LookupTable {
public:
    UPDATEFUNC update;
    std::vector<OUTPUT>  table;

    LookupTable(UPDATEFUNC updateFunction) : update(std::move(updateFunction)) { }

    auto operator()(size_t in) const { return table[in]; }


    /** Train on Input/Output pair */
    template<class EVIDENCE>
    void on(const events::IOPair<size_t,EVIDENCE> & event) {
        table[event.input] = update(table[event.input], event.output);
    }


    friend std::ostream &operator <<(std::ostream &out, const LookupTable<OUTPUT,UPDATEFUNC> &lookup) {
        for(uint i=0; i<lookup.table.size(); ++i) {
            out << i << " -> " << lookup.table[i] << '\n';
        }
        return out;
    }

};


#endif //MULTIAGENTGOVERNMENT_LOOKUPTABLE_H
