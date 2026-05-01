// openmp.h - Conditional OpenMP wrapper macros (internal header)
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_INTERNAL_OPENMP_H
#define BOAT_INTERNAL_OPENMP_H

#ifdef _OPENMP
    #include <omp.h>
    #define BOAT_OMP_PARALLEL_FOR _Pragma("omp parallel for")
    #define BOAT_OMP_PARALLEL_FOR_SCHEDULE(sched) _Pragma("omp parallel for schedule(" #sched ")")
    #define BOAT_OMP_GET_THREAD_NUM() omp_get_thread_num()
    #define BOAT_OMP_GET_NUM_THREADS() omp_get_num_threads()
    #define BOAT_OMP_SET_NUM_THREADS(n) omp_set_num_threads(n)
#else
    #define BOAT_OMP_PARALLEL_FOR
    #define BOAT_OMP_PARALLEL_FOR_SCHEDULE(sched)
    #define BOAT_OMP_GET_THREAD_NUM() 0
    #define BOAT_OMP_GET_NUM_THREADS() 1
    #define BOAT_OMP_SET_NUM_THREADS(n) ((void)0)
#endif

#endif // BOAT_INTERNAL_OPENMP_H
