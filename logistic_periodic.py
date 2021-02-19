import argparse
import taichi as ti
from time import time

import utils as ut
import numpy as np
import tqdm

ti.init(arch=ti.gpu)

a_valid = ti.field(ut.real, ())
a_abort = ti.field(ut.real, ())
a_valid[None] = 0
a_abort[None] = 0


@ti.kernel
def search_periodic(burn_in: ti.i32,
                    target_period: ti.i32,
                    max_orbit_length: ti.i32,
                    a_min: ut.real,
                    a_dt: ut.real,
                    n_threads: ti.i32):
    for tid in range(n_threads):
        # if tid == 0:
        #     print("a: ", a_min, a_min + a_dt * n_threads)
        if a_valid[None] == 0:
            a_curr = a_min + a_dt * (tid - 1 + ti.random(ut.real))
            period_len = ut.test_periodic(a_curr, burn_in, target_period, max_orbit_length)
            if period_len > target_period:
                a_abort[None] = 1
            elif period_len == target_period:
                if (a_abort[None] == 0):
                    print("Tid:", tid,  "| found a period", target_period, "found with a:", a_curr)
                    if (a_valid[None] == 0 or a_valid[None] > a_curr):
                        a_valid[None] = a_curr



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--burn_in", default=1, type=int)
    parser.add_argument("-p", "--target_period", default=2, type=int)
    parser.add_argument("-m", "--max_orbit_length", default=1<<30, type=int)
    parser.add_argument("--a_min", default=2.5, type=float)
    parser.add_argument("--a_max", default=3.5, type=float)
    parser.add_argument("-n", "--n_threads", default=1024, type=int)
    parser.add_argument( "--threads_per_kernel", default=1<<14, type=int)

    args = parser.parse_args()



#     a_vals = [3, # period 1 - 2 bifurcation
#                 3.4494897, # 2-4, from wikipedia
#                 3.5440903, # 4-8, from wikipedia
#                 3.5644073] # 8-16, from wikipedia
    
#     target_period_log_two = 5
#     n_threads = 1<<15
#     # a_range = 5e-3

#     while True:
#         n_groups = (n_threads + args.threads_per_kernel - 1 ) // args.threads_per_kernel

#         a_min = a_vals[-1]
#         # 3.570019227359
#         a_max =  3.569945672
#         a_range = a_max - a_min
#         # a_mid = ut.approx_next_a_val(a_vals[-1], a_vals[-2])
#         # a_max = 2*a_mid -  a_min
#         # a_range = a_max - a_min
#         a_dt = a_range / n_threads
#         # if (a_dt < 1e-11):
#         #     a_range *= 2
#         #     a_max = a_min + a_range
#         #     a_dt = a_range / n_threads
#         target_period = 1<<target_period_log_two
#         print(f"searching in a: [{a_min}-{a_max}] for target period: {target_period} (1<<{target_period_log_two})" )
    
#         a_valid[None] = 0
#         a_abort[None] = 0
#         burn_in = 1<<20

#         for group_id_raw in tqdm.tqdm(range(n_groups)):
#             group_id = n_groups//2 + (group_id_raw%2 *2 - 1)*group_id_raw//2
#             print(group_id)
#             threads_in_group = args.threads_per_kernel if group_id < n_groups - 1 else n_threads % args.threads_per_kernel
#             # print(f"running group {group_id} of {n_groups} searching for period {target_period} (1<<{target_period_log_two}) with 1<<{np.log2(n_threads)} threads")
#             search_periodic(burn_in,
#                             target_period,
#                             target_period,
#                             a_min + a_dt * (group_id * args.threads_per_kernel),
#                             a_dt,
#                             threads_in_group)
#             # if a_valid[None] != 0:
#             #     print(f"Found a period of length {target_period} (1<<{target_period_log_two}) with a: {a_valid[None]}\n\n")
#                 # break
#         if a_valid[None] == 0:
#             print(f"Failed to find a period of length {target_period} (1<<{target_period_log_two}).")
#             n_threads *= 2
#             # if a_dt < 1e-10:
#             #     a_range *= 2
#             print(f"Running again with {n_threads} (1<<{np.log2(n_threads)}) threads")
#         else:
#             print(f"Period {target_period} (1<<{target_period_log_two}) finished")
#             target_period_log_two += 1
#             a_vals.append(a_valid[None])
#             # a_range /= 2
#             # n_threads //= 2



    assert args.max_orbit_length >= args.target_period
    print(args)

    a_range = args.a_max - args.a_min
    a_dt = a_range/args.n_threads

    n_groups = (args.n_threads + args.threads_per_kernel - 1 ) // args.threads_per_kernel


    print(f"Batching computation into {n_groups} kernel launches each"\
        "of {args.threads_per_kernel} threads")
    print("a_dt is: {:E}".format(a_dt))
    t0 = time()
    tq = tqdm.tqdm(range(n_groups))
    # tq = range(n_groups)
    for group_id in tq:
        tq.set_description_str("a_min: %.15f"%(args.a_min +
                                        a_dt * (group_id *args.threads_per_kernel)))
        search_periodic(args.burn_in*args.target_period,
                    args.target_period, args.max_orbit_length,
                    args.a_min + a_dt * (group_id *args.threads_per_kernel),
                    a_dt, args.threads_per_kernel)
        if a_valid[None] != 0:
            break
    t1 = time()
    print(f"Simulations took: {t1-t0}")
    print("a_valid:", a_valid[None])
    print("a_value_as_int:", ut.integer_from_float(a_valid[None]))