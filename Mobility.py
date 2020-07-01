import matplotlib.pyplot as plt
import math
import random
import numpy as np
import networkx as nx

from matplotlib.patches import Ellipse
from scipy.spatial.distance import *

def mobile():
    # Borough coordinates
    BC = {'Manhattan': (40.7831, 73.9712),
          'Bronx': (40.8448, 73.8648),
          'Brooklyn': (40.6782, 73.9442),
          'Queens': (40.7282, 73.7949),
          'Staten Island': (40.5795, 74.1502)
          }

    # District coordinates
    DC = {'Manhattan': [(40.7163, 74.0086), (40.7336, 74.0027), (40.7150, 73.9843), (40.7465, 74.0014), (40.7549, 73.9840), (40.7571, 73.9719),
                       (40.7870, 73.9754), (40.7736, 73.9566), (40.8253, 73.9476), (40.8089, 73.9482), (40.7957, 73.9389), (40.8677, 73.9212)],

          'Bronx': [(40.8245, 73.9104), (40.8248, 73.8916), (40.8311, 73.9059), (40.8369, 73.9271), (40.8575, 73.9097), (40.8535, 73.8894),
                    (40.8810, 73.8785), (40.8834, 73.9051), (40.8303, 73.8507), (40.8398, 73.8465), (40.8631, 73.8616), (40.8976, 73.8669)],

          'Brooklyn': [(40.7081, 73.9571), (40.6961, 73.9845), (40.6783, 73.9108), (40.6958, 73.9171), (40.6591, 73.8759), (40.6734, 74.0083),
                       (40.6527, 74.0093), (40.6694, 73.9422), (40.6602, 73.9690), (40.6264, 74.0299), (40.6039, 74.0062), (40.6204, 73.9600),
                       (40.5755, 73.9707), (40.6415, 73.9594), (40.6069, 73.9480), (40.6783, 73.9108), (40.6482, 73.9300), (40.6233, 73.9322)],

          'Queens': [(40.7931, 73.8860), (40.7433, 73.9196), (40.7544, 73.8669), (40.7380, 73.8801), (40.7017, 73.8842), (40.7181, 73.8448),
                     (40.7864, 73.8390), (40.7136, 73.7965), (40.7057, 73.8272), (40.6764, 73.8125), (40.7578, 73.7834), (40.6895, 73.7644),
                     (40.7472, 73.7118), (40.6158, 73.8213)],

          'Staten Island': [(40.6323, 74.1651), (40.5890, 74.1915), (40.5434, 74.1976)]
          }

    return DC


def find_dist(x1, y1, x2, y2, lat_dist, lon_dist):
    return math.sqrt(math.pow(x2 - x1, 2) * lat_dist + math.pow(y2 - y1, 2) * lon_dist)


def latp(pt, D, a):

    # Available locations
    AL = [k for k in D.keys() if euclidean(D[k], pt) > 0]
    print (AL)

    den = np.sum([1.0 / math.pow(float(euclidean(D[k], pt)), a) for k in sorted(AL)])

    plist = [(1.0 / math.pow(float(euclidean(D[k], pt)), a) / den) for k in sorted(AL)]

    next_stop = np.random.choice([k for k in sorted(AL)], p = plist, size = 1)

    # print (plist[next_stop[0]])

    return next_stop[0], D[next_stop[0]]


def SNT(pt, D, G, P, u, freqs):

    A = {k: 0 for k in D.keys()}
    for k in A.keys():
        if len([v for v in G.nodes() if P[v] == k]) > 0 and k != pt:
            A[k] = float(len([v for v in G.neighbors(u) if P[v] == k]))/float(len([v for v in G.nodes() if P[v] == k]))

    plist = [float(A[k])/float(sum(A.values())) for k in sorted(D.keys())]
    # print ('Available destination slots:', [i for i, v in enumerate(plist) if v > 0])

    next_stop = np.random.choice([k for k in sorted(A.keys())], p = plist, size = 1)

    freqs[next_stop[0]] += 1
    return next_stop[0], D[next_stop[0]], freqs


def orbit(f, D, pr):

    global rB, lat_dist, lon_dist

    if random.uniform(0, 1) < pr:
        return f, (random.uniform(D[f][0] - rB/lat_dist, D[f][0] + rB/lat_dist),
                   random.uniform(D[f][1] - rB/lat_dist, D[f][1] + rB/lat_dist))

    else:
        f = random.choice(list(D.keys()))
        return f, (random.uniform(D[f][0] - rB/lon_dist, D[f][0] + rB/lon_dist),
                   random.uniform(D[f][1] - rB/lon_dist, D[f][1] + rB/lon_dist))


fig, ax = plt.subplots()

# Area of NYC
A = 302.6

# Number of neighborhoods
nB = 59

# radius of a neighborhood
AB = A/nB
rB = math.sqrt(AB/ math.pi)
print (rB)

# Distance between two latitude and longitudes
lat_dist = 69.0
lon_dist = 54.6

DC = mobile()

# Plot NYC Map
colorlist = ['green', 'red', 'black', 'orange', 'magenta']
boroughs = ['Manhattan', 'Bronx', 'Brooklyn', 'Queens', 'Staten Island']

'''
i = 0
for p in DC.keys():

    L = DC[p]
    ax.scatter([L[j][0] for j in range(len(L))], [L[j][1] for j in range(len(L))], c = colorlist[i], s = 2, label = p)
    print(rB / lat_dist, rB / lon_dist)

    # Create circles around each neighborhood
    for j in range(len(L)):
        circle1 = Ellipse((L[j][0], L[j][1]), width = rB/lat_dist, height = rB/lat_dist, color = 'black', fill = False, alpha = 0.2)
        ax.add_patch(circle1)
    i = i + 1


plt.legend()
plt.xlabel('Latitude', fontsize = 15)
plt.ylabel('Longitude', fontsize = 15)

plt.grid(alpha = 0.2)
plt.tight_layout()
plt.savefig('NYC_Map.png', dpi = 300)
plt.show()
'''

# Generate a sub-list of NYC zones
D = {}
i = 0
for k in DC.keys():
    L = DC[k]

    for z in range(2):
        D[i] = L[z]
        i = i + 1

print (D)


hubs = [1, 2, 3]

# LATP and SNT Mobility
# Number of steps to be taken

steps = 100
new_pt = D[0]
#
# # Friendship network
# how_many_people = 100
# G = nx.erdos_renyi_graph(n = how_many_people, p = 0.15, directed = False)
#
# # Location of each person: person: destination
# P = {u: np.random.choice(list(D.keys())) for u in G.nodes()}
#
# # Weighing factor (for LATP)
a = 1.2

i = 0
dist_list = []

# Probability of intra-zone mobility (for ORBIT)
# pr = 0.9
#
# freqs = {i: 0 for i in D.keys()}
# affty = {i: len([u for u in G.neighbors(0) if P[u] == i]) for i in D.keys()}
# f = 0
#
# Firsts = [True for i in range(len(colorlist))]
#
while i < steps:

    old_pt = new_pt

    # LATP model
    f, new_pt = latp(old_pt, D, a)

    # SNT model
    # f, new_pt, freqs = SNT(old_pt, D, G, P, 0, freqs)

    # Orbit model
    # f, new_pt = orbit(f, D, pr)

    # plt.plot([old_pt[0], new_pt[0]], [old_pt[1], new_pt[1]], linestyle = 'dotted', color = 'black', linewidth = 1)

    # if Firsts[int(f/2)]:
    #     plt.scatter(new_pt[0], new_pt[1], s = 5, c = colorlist[int(f/2)], label = boroughs[int(f/2)])
    #     Firsts[int(f / 2)] = False
    #
    # else:
    #     plt.scatter(new_pt[0], new_pt[1], s = 5, c = colorlist[int(f/2)])

    dist_list.append(find_dist(old_pt[0], old_pt[1], new_pt[0], new_pt[1], lat_dist, lon_dist))

    i = i + 1

# print (freqs)
# print (affty)
#
# # print (dist_list)
# # plt.scatter([D[j][0] for j in D.keys()], [D[j][1] for j in D.keys()], s = 10, c = 'black')
# # y = [freqs[j] for j in freqs.keys()]
# # y = [v/sum(y) for v in y]
# #
# # x = [affty[j] for j in freqs.keys()]
# #
# # plt.scatter(x, y, s = 10, c = 'black')
# #
# # z = np.polyfit(x, y, 1)
# # p = np.poly1d(z)
# # ym = [p(v) for v in x]
#
weights = np.ones_like(dist_list)/len(dist_list)
plt.hist(dist_list, weights = weights, bins = 10, edgecolor='black')  # `density=False` would make counts
plt.ylabel('Probability', fontsize = 15)
plt.xlabel('Distance in miles', fontsize = 15)
#
# # plt.legend()
# plt.xlabel('Latitude', fontsize = 15)
# plt.ylabel('Longitude', fontsize = 15)
# #
# # plt.grid(alpha = 0.2)
# # plt.tight_layout()
#
# # plt.plot(x, ym, linewidth = 2, c = 'black')
# # plt.ylabel('Probability', fontsize = 15)
# # plt.xlabel('Social Affinity', fontsize = 15)
# plt.legend()
plt.tight_layout()
plt.savefig('NYC_Mobility_LATP.png', dpi = 300)
plt.show()
#
