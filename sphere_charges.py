#!/usr/bin/env python
# @author: "Colby"
from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

# simple plotting on a sphere: 
# https://stackoverflow.com/questions/31768031/plotting-points-on-the-surface-of-a-sphere-in-pythons-matplotlib

class Charge():
    def __init__(self, q, pos, id):
        self.q = q
        self.pos = np.array(pos)
        self.id = id

def to_spherical(pos):
    r = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
    theta = math.acos(pos[2]/r)
    phi = math.atan2(pos[1],pos[0])
    return np.array([r, theta, phi])

def to_cartesian(pos):
    x = pos[0] * math.sin(pos[1]) * math.cos(pos[2])
    y = pos[0] * math.sin(pos[1]) * math.sin(pos[2])
    z = pos[0] * math.cos(pos[1])
    return np.array([x,y,z])

def energy(charge1, charge2):
    sepvec = charge1.pos - charge2.pos
    sep = np.sqrt(np.dot(sepvec,sepvec))
    return (charge1.q * charge2.q) / sep

def electrostatic_energy(charges):
    W = 0.0
    N = len(charges)
    for i in range(N):
        for j in range(i+1,N):
            W = W + energy(charges[i],charges[j])
    return W

def set_theta(charges, theta):
    for c in charges:
        if c.id!=0:
            spherical_pos = to_spherical(c.pos)
            spherical_pos[1] = theta
            c.pos = to_cartesian(spherical_pos)
    return charges

#-------------------------------------------
# Define old and new sets of charges
#-------------------------------------------
old_charges = [Charge(1.0, [0.0, 0.0, 1.0], 0), Charge(1.0,[np.sqrt(8.0/9.0), 0.0, -1.0/3.0], 1), Charge(1.0, [-np.sqrt(2.0/9.0), np.sqrt(2.0/3.0), -1.0/3.0], 2), Charge(1.0, [-np.sqrt(2.0/9.0), -np.sqrt(2.0/3.0), -1.0/3.0], 3) ]

# the only difference here is the charge of charges[0], q->2e
charges = [Charge(2.0, [0.0, 0.0, 1.0], 0), Charge(1.0,[np.sqrt(8.0/9.0), 0.0, -1.0/3.0], 1), Charge(1.0, [-np.sqrt(2.0/9.0), np.sqrt(2.0/3.0), -1.0/3.0], 2), Charge(1.0, [-np.sqrt(2.0/9.0), -np.sqrt(2.0/3.0), -1.0/3.0], 3) ]

#-------------------------------------------
# Compute W for many reasonable angles
#-------------------------------------------
thetas = np.arange(np.pi/2.0, np.pi, 0.00001)
Ws = []
for th in thetas:
    set_theta(charges, th)
    Ws.append(electrostatic_energy(charges))

Ws = np.array(Ws)

#-------------------------------------------
# Find minimum electrostatic energy
#-------------------------------------------
min_index = np.argmin(Ws)
set_theta(charges, thetas[min_index])

print("The optimal angle is theta = ", str(thetas[min_index]), "radians, measured from the charge at the top of the sphere (z-axis).")

plt.close('all')

#-------------------------------------------
# Create a sphere on which to plot charges
#-------------------------------------------
r = 1.0
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0.0:pi:20j, 0.0:2.0*pi:20j]
x = r*sin(phi)*cos(theta)
y = r*sin(phi)*sin(theta)
z = r*cos(phi)

fig = plt.figure()

#-------------------------------------------
# Plot original configuration
#-------------------------------------------
ax = fig.add_subplot(121, projection='3d')

xx = r * np.array([old_charges[0].pos[0], old_charges[1].pos[0], old_charges[2].pos[0], old_charges[3].pos[0]])
yy = r * np.array([old_charges[0].pos[1], old_charges[1].pos[1], old_charges[2].pos[1], old_charges[3].pos[1]])
zz = r * np.array([old_charges[0].pos[2], old_charges[1].pos[2], old_charges[2].pos[2], old_charges[3].pos[2]])

ax.scatter(xx[1:],yy[1:],zz[1:],c="k", edgecolor='#50f442', s=40)
ax.scatter(xx[0],yy[0],zz[0],c='k', edgecolor='#50f442', s=80)

ax.plot_surface(x, y, z, rstride=1, cstride=1, color='m', alpha=0.2, linewidth=0)

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.set_aspect("equal")
ax.view_init(0,0)
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.yaxis.set_major_formatter(plt.NullFormatter())
ax.zaxis.set_major_formatter(plt.NullFormatter())
ax.set_title("$q_{\mathrm{top}} = -e$")

#-------------------------------------------
# Plot new configuration
#-------------------------------------------
ax = fig.add_subplot(122, projection='3d')

xx = r * np.array([charges[0].pos[0], charges[1].pos[0], charges[2].pos[0], charges[3].pos[0]])
yy = r * np.array([charges[0].pos[1], charges[1].pos[1], charges[2].pos[1], charges[3].pos[1]])
zz = r * np.array([charges[0].pos[2], charges[1].pos[2], charges[2].pos[2], charges[3].pos[2]])

ax.scatter(xx[1:],yy[1:],zz[1:],c="k", edgecolor='#50f442', s=40)
ax.scatter(xx[0],yy[0],zz[0],c='k', edgecolor='#50f442', s=80)

ax.plot_surface(x, y, z, rstride=1, cstride=1, color='m', alpha=0.2, linewidth=0)

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.set_aspect("equal")
ax.view_init(0,0)
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.yaxis.set_major_formatter(plt.NullFormatter())
ax.zaxis.set_major_formatter(plt.NullFormatter())
ax.set_title("$q_{\mathrm{top}} = -2e$")

#-------------------------------------------
# Show, and/or save plot to .pdf
#-------------------------------------------
#fig.savefig("foo.pdf", bbox_inches='tight')
plt.show()
