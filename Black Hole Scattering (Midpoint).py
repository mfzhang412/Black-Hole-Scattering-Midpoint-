"""
Relativistic scattering off a non-rotating, uniform, and point-like black hole
using the midpoint integration technique.

@author: Michael Zhang
"""
import numpy as np

m_sun = 1.989e30
G = 6.673e-11
M = 10*m_sun
m = 1
c = 299792458
gamma = G*M*m
r_schwarz = 2*G*M/c**2


def to_spher(cartesian):
    if cartesian.ndim != 1:
        r = np.linalg.norm(cartesian, axis=1)
    else:
        r = np.linalg.norm(cartesian)
    theta = np.arctan(np.linalg.norm(cartesian.T[:1], axis=0)/cartesian.T[-1])
    if theta != 0 and theta != np.pi:
        phi = np.arctan(cartesian.T[1] / cartesian.T[0])
    else:
        phi = 0
    return np.array([r,theta,phi])


def force(r, v, b):
    r_magn = np.linalg.norm(r)
    r_hat = r/r_magn
    
    F = -r_hat*(gamma/r_magn**2 + 3*gamma/r_magn**4*b**2*(v/c)**2)
    return F


def acc(f, v):
    A = f * (1-(v/c)**2)**(3/2)/(1+(v/c)**2*(c**2-1))
    return A
    

num_iters = 100000

b = 2*r_schwarz
z_0 = -10*r_schwarz
v_0 = 0.0001*c

r = [np.array([b,0,z_0])]
v = [np.array([0,0,v_0])]
dt = 0.1

for i in range(num_iters):
    r_curr = r[-1]
    v_curr = v[-1]
    v_curr_magn = np.linalg.norm(v_curr)
    F_curr = force(r_curr, v_curr_magn, b)
    a_curr = acc(F_curr, v_curr_magn)

    r_mid = r_curr + v_curr * dt/2
    v_mid = v_curr + a_curr * dt/2
    v_mid_magn = np.linalg.norm(v_mid)
    F_mid = force(r_mid, v_mid_magn, b)
    a_mid = acc(F_mid, v_mid_magn)
    
    dr = v_mid * dt
    dv = a_mid * dt
    
    r.append(r_curr + dr)
    v.append(v_curr + dv)

v_spher_initial = to_spher(v[0])
v_spher_final = to_spher(v[-1])

print("Incident angle:", v_spher_initial[1]/np.pi, "pi")
print("Scattered angle:", v_spher_final[1]/np.pi, "pi")
