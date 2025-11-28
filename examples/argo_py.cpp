// argo_py.cpp
#include <cstddef>
#include "argo.h"

extern "C" {

void py_argo_init(std::size_t asize, std::size_t csize) {
    argo_init(asize, csize);
}

int py_argo_node_id(void) {
    return argo_node_id();
}

int py_argo_number_of_nodes(void) {
    return argo_number_of_nodes();
}

void *py_collective_alloc(std::size_t size) {
    return collective_alloc(size);
}

void py_argo_barrier(int b) {
    argo_barrier(b);
}

void py_argo_finalize(void) {
    argo_finalize();
}

} // extern "C"
