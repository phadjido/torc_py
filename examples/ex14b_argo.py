"""
(C) Copyright IBM Corporation 2020
All rights reserved. This program and the accompanying materials
are made available under the terms of the Eclipse Public License v1.0
which accompanies this distribution, and is available at
http://www.eclipse.org/legal/epl-v10.html
"""

"""
Explicit global memory variable using MPI shared memory
"""
import torcpy as torc
import time
from mpi4py import MPI
import numpy as np
from functools import partial
import ctypes


# argodsm
import os
from ctypes import c_size_t, c_int, c_double, c_void_p, POINTER
import socket

# Ensure current dir in loader path (helpful under mpirun)
os.environ["LD_LIBRARY_PATH"] = os.getcwd() + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# Load shim; it already pulls in libargo + backend
lib = ctypes.CDLL("./libargo_py.so")

# Declare shim signatures
lib.py_argo_init.argtypes = [c_size_t, c_size_t]
lib.py_argo_init.restype  = None

lib.py_argo_node_id.argtypes = []
lib.py_argo_node_id.restype  = c_int

lib.py_argo_number_of_nodes.argtypes = []
lib.py_argo_number_of_nodes.restype  = c_int

lib.py_collective_alloc.argtypes = [c_size_t]
lib.py_collective_alloc.restype  = c_void_p

lib.py_argo_barrier.argtypes = [c_int]
lib.py_argo_barrier.restype  = None

lib.py_argo_finalize.argtypes = []
lib.py_argo_finalize.restype  = None

#

A = None

def alloc_mem(shape, dtype):
    global A

    # Same init as in your C code
    lib.py_argo_init(1024 * 1024, 4 * 1024)

    me   = lib.py_argo_node_id()
    size = lib.py_argo_number_of_nodes()
    host = socket.gethostname()

    n = 64
    raw_ptr = lib.py_collective_alloc(n * ctypes.sizeof(c_double))

    DoubleArray = c_double * n
    c_array_ptr = ctypes.cast(raw_ptr, POINTER(DoubleArray))
    c_array     = c_array_ptr.contents
    A = np.ctypeslib.as_array(c_array)

    print(f"d (rank {me}) = {raw_ptr}")

    lib.py_argo_barrier(1)
    print(f"2 hello from process {me} of {size}")


#    size = np.prod(shape)
#    if torc.node_id() == 0:
#        nbytes = size * itemsize 
#    else: 
#        nbytes = 0
#    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=MPI.COMM_WORLD) 
#    buf, itemsize = win.Shared_query(0) 
#    A = np.ndarray(buffer=buf, dtype=dtype, shape=shape) 
    return


def sdsm_barrier():
    lib.py_argo_barrier(1)


def sdsm_finalize():
    lib.py_argo_finalize()



def foo(x):
    global A

    print("foo begins", flush=True)
    print("foo: A=>", A, flush=True)
    A[x] = A[x] + 99*(x+1)
    print("foo2: A=>", A, flush=True)
    return x + 1


def main():
    global A
    N = 64
    shape=(N,)
    dtype = 'float64'

    torc.spmd(alloc_mem, shape, dtype)

    # primary task initializes array A on rank 0
    for i in range(0, N):
        A[i] = 100*i

    f1 = torc.submit(foo, 1)
    f2 = torc.submit(foo, 2)
    f3 = torc.submit(foo, 3)
    f4 = torc.submit(foo, 4)
    torc.waitall()
    torc.spmd(sdsm_barrier)

    print(f1.result())
    print(f2.result())
    print(f3.result())
    print(f4.result())


    print("main: A=>", A, flush=True)

    torc.spmd(sdsm_barrier)
    torc.spmd(sdsm_finalize)


if __name__ == '__main__':
    torc.start(main)
