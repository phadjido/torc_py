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

# Ensure current dir is searched
os.environ["LD_LIBRARY_PATH"] = os.getcwd() + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# 1) Load core Argo first, lazily, symbols global
libargo = ctypes.CDLL(
    "./libargo.so",
    mode=os.RTLD_LAZY | os.RTLD_GLOBAL
)

# 2) Then load backend, also lazily
libbackend = ctypes.CDLL(
    "./libargobackend-mpi.so",   # name from your error
    mode=os.RTLD_LAZY | os.RTLD_GLOBAL
)

# Now bind the Argo API from libargo
libargo.argo_init.argtypes = [c_size_t, c_size_t]
libargo.argo_init.restype  = None

libargo.argo_node_id.argtypes = []
libargo.argo_node_id.restype  = c_int

libargo.argo_number_of_nodes.argtypes = []
libargo.argo_number_of_nodes.restype  = c_int

libargo.collective_alloc.argtypes = [c_size_t]
libargo.collective_alloc.restype  = c_void_p

libargo.argo_barrier.argtypes = [c_int]
libargo.argo_barrier.restype  = None

libargo.argo_finalize.argtypes = []
libargo.argo_finalize.restype  = None


#

A = None

def alloc_mem(shape, dtype):
    global A

    libargo.argo_init(1024 * 1024, 4 * 1024)

    me   = libargo.argo_node_id()
    size = libargo.argo_number_of_nodes()
    host = socket.gethostname()

    dt = np.dtype(dtype)
    print(dt)
    itemsize = np.dtype(dtype).itemsize

    n = 64
    raw_ptr = libargo.collective_alloc(n * ctypes.sizeof(c_double))

    DoubleArray = c_double * n
    c_array_ptr = ctypes.cast(raw_ptr, POINTER(DoubleArray))
    c_array     = c_array_ptr.contents
    A = np.ctypeslib.as_array(c_array)

    print(f"d (rank {me}) = {raw_ptr}")

    libargo.argo_barrier(1)
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
    libargo.argo_barrier(1)



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


if __name__ == '__main__':
    torc.start(main)
