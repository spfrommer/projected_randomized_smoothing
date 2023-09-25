import pdb
from math import exp, log, pi

def logfact(n):
    # From https://math.stackexchange.com/questions/138194/approximating-log-of-factorial
    sum = 0
    for i in range(1, n+1):
        sum = sum + log(i)
    return sum

def nball_vol_log(n, r):
    assert n % 2 == 0

    return (n // 2) * log(pi) + n * log(r) - logfact(n // 2)

def nball_l1_vol_log(n, r):
    return n * log(2) + n * log(r) - logfact(n)

def nball_linf_vol_log(n, r):
    return n * log(r)

def nellipse_vol_log(n, rs):
    assert n % 2 == 0

    return (n // 2) * log(pi) + rs.log().sum().item() - logfact(n // 2)


def nball_radius(n, vol_log):
    nlogr = vol_log - (n // 2) * log(pi) + logfact(n // 2)
    return exp(nlogr / n)
