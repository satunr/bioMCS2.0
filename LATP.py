import matplotlib.pyplot as plt
import random, math
import numpy as np
import networkx as nx

from scipy.spatial.distance import *


def correlate(D, F):

    for k in D.keys():

        dist = np.mean([euclidean(D[k], D[l]) for l in D.keys()])
        plt.scatter(F[k], float(dist))

    plt.show()


def latp(pt, D, a):

    # Available locations
    AL = [k for k in D.keys() if euclidean(D[k], pt) > 0]

    den = np.sum([1.0 / math.pow(float(euclidean(D[k], pt)), a) for k in sorted(AL)])

    plist = [(1.0 / math.pow(float(euclidean(D[k], pt)), a) / den) for k in sorted(AL)]

    next_stop = np.random.choice([k for k in sorted(AL)], p = plist, size = 1)

    # print (plist[next_stop[0]])

    return next_stop[0], D[next_stop[0]]


def SNT(pt, D, G, P, u):

    A = {k: 0 for k in D.keys()}
    for k in A.keys():
        if len([v for v in G.nodes() if P[v] == k]) > 0 and k != pt:
            A[k] = float(len([v for v in G.neighbors(u) if P[v] == k]))/float(len([v for v in G.nodes() if P[v] == k]))

    plist = [float(A[k])/float(sum(A.values())) for k in sorted(D.keys())]
    # print ('Available destination slots:', [i for i, v in enumerate(plist) if v > 0])

    next_stop = np.random.choice([k for k in sorted(A.keys())], p = plist, size = 1)

    return next_stop[0], D[next_stop[0]]


'''
# Simulation range
Xlim = 300.0
Ylim = 300.0

# How many destinations
how_many = 50
how_many_people = 100

# Number of steps to be taken
steps = 50

# Coordinates of destination
D = {i: [random.randint(0, Xlim), (random.randint(0, Ylim))] for i in range(how_many)}
plt.scatter([D[k][0] for k in D.keys()], [D[k][1] for k in D.keys()], s = 25, c = 'red', label = 'waypoints')

# Set of visited positions (optional)
V = []

# Node initial position
new_pt = D[np.random.choice(list(D.keys()))]

# Weighing factor (for LATP)
a = 1.2

# Friendship network
G = nx.erdos_renyi_graph(n = how_many_people, p = 0.1, directed = False)
print (list(G.neighbors(0)))
input('')

# Frequency of selection
F = {k: 0 for k in D.keys()}

i = 0
f = None
while i < steps:

    # Location of each person: person: destination
    P = {u: np.random.choice(list(D.keys())) for u in G.nodes()}
    P[0] = f

    old_pt = new_pt

    # LATP model
    # f, new_pt = latp(old_pt, D, a)

    # SNT model
    f, new_pt = SNT(new_pt, D, G, P, 0)

    print (f, [u for u in G.nodes() if P[u] == f], "\n")

    plt.plot([old_pt[0], new_pt[0]], [old_pt[1], new_pt[1]], linestyle = 'dotted')
    plt.scatter(new_pt[0], new_pt[1], s = 5)

    F[f] += 1.0

    i = i + 1

plt.legend()
plt.savefig('SNT.png', dpi = 300)
plt.show()

# Validation
# print (F)
# correlate(D, F)
'''
