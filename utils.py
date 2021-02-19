import struct
import taichi as ti
real = ti.f64

def float_from_integer(integer):
    return struct.unpack('!d', struct.pack('!Q', integer))[0]

def integer_from_float(flt):
    return struct.unpack('!Q', struct.pack('!d', flt))[0]

# tests
assert integer_from_float(float_from_integer(1)) == 1
assert float_from_integer(integer_from_float(1.0)) == 1.0

feigenbaum_const = 4.669201609102990671853203820466 # I know that we have too many digits here for fp64 precision.

def approx_next_a_val(a_curr, a_prev):
    return ((1+feigenbaum_const)* a_curr - a_prev)/feigenbaum_const

@ti.func
def logistic_map(x, a):
    return a * x * (1-x)
    
@ti.func
def test_periodic(a, burn_in, target_orbit_length, max_orbit_length):
    x_nought = ti.random(real)
    x_curr = x_nought

    # allow the initial condition to settle into an orbit
    for idx in range(burn_in):
        x_curr = a * x_curr * (1-x_curr)

    # save first point in the possible orbit
    x_final = x_curr
    period_length = 0
    period_found = False
    # print("x_final", x_final)
    for idx in range(max_orbit_length):
        x_curr = a * x_curr * (1-x_curr)
        if not period_found:
            period_length += 1
        
        if (x_curr == x_final):
            period_found = True
    # if (period_length >= target_orbit_length):
    #     print("Period of length", period_length, "found - a:", a)
    if not period_found:
        period_length = 0
    return period_length
