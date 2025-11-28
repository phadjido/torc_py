export ARGO_NO_MPI_FINALIZE="TRUE"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.
mpirun -n 2 python ex14b_argo.py
