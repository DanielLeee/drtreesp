from cffi import FFI
ffibuilder = FFI()

ffibuilder.cdef("void fast_parse(double *para_weight, bool one_root, int32_t n, int32_t num_rels, int32_t *result_3d);")

ffibuilder.set_source("_cmst",  # name of the output C extension
"""
#include "stdbool.h"

void fast_parse(double *para_weight, bool one_root, int32_t n, int32_t num_rels, int32_t *result_3d);
""",
    sources=['cmst.c'],   # includes pi.c as additional sources
    libraries=[])    # on Unix, link with the math library

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
