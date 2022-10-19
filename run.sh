#!/bin/bash

mpirun -mca btl vader,self,tcp --mca mpi_cuda_support 0 -map-by node --bind-to none -n 1  -hostfile nodes $1
