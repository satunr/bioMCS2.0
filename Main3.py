import networkx as nx
import simpy
import numpy as np
import pickle
import random
import math
import matplotlib.pyplot as plt
import operator

from scipy.spatial.distance import *


def find_pdr(Tot, Rec):

    num = []
    den = []
    for e in Tot:
        e = e[:-1]
        if e not in den:
            den.append(e)

    for e in Rec:
        e = e[:-1]
        if e not in num:
            num.append(e)

    return float(len(num))/float(len(den))


def rw_angle(f, D):

    global rB, lat_dist, lon_dist, pr

    '''
    new = (random.uniform(Xlim[0], Xlim[1]), random.uniform(Ylim[0], Ylim[1]))
    old = (x, y)

    angle = np.rad2deg(np.arctan2(new[1] - y, new[0] - x))

    disp_x = step * math.cos(math.radians(angle))
    disp_y = step * math.sin(math.radians(angle))

    x = x + disp_x
    y = y + disp_y

    if (x > Xlim[1] or x < Xlim[0]) or (y > Ylim[1] or y < Ylim[0]):
        x = old[0]
        y = old[1]

    return x, y
    '''

    if random.uniform(0, 1) < 0.5:
        return f, (random.uniform(D[f][0] - rB/lat_dist, D[f][0] + rB/lat_dist),
                   random.uniform(D[f][1] - rB/lat_dist, D[f][1] + rB/lat_dist))

    else:
        f = random.choice([i for i in range(len(D))])
        return f, (random.uniform(D[f][0] - rB/lon_dist, D[f][0] + rB/lon_dist),
                   random.uniform(D[f][1] - rB/lon_dist, D[f][1] + rB/lon_dist))


def latp(pt, D, cur_zone):

    global aLATP

    # Available locations
    AL = [i for i in range(len(D)) if euclidean(D[i], pt) > 0]

    den = np.sum([1.0 / math.pow(float(euclidean(D[k], pt)), aLATP) for k in sorted(AL)])
    plist = [(1.0 / math.pow(float(euclidean(D[k], pt)), aLATP) / den) for k in sorted(AL)]

    next_stop = np.random.choice([k for k in sorted(AL)], p = plist, size = 1)

    return next_stop[0], place_node_in_zone(D, next_stop[0])


def SNT(pt, D, G, P, u):

    A = [0 for i in range(len(D))]

    for i in range(len(A)):
        if len([v for v in G.nodes() if P[v] == i]) > 0:
            A[i] = float(len([v for v in G.neighbors(u) if P[v] == i]))/float(len([v for v in G.nodes() if P[v] == i]))

    plist = [float(A[i])/float(sum(A)) for i in range(len(A))]

    next_stop = np.random.choice([i for i in range(len(A))], p = plist, size = 1)
    return next_stop[0], place_node_in_zone(D, next_stop[0])


def orbit(f, D):

    global rB, lat_dist, lon_dist, pr

    if random.uniform(0, 1) < pr:
        return f, (random.uniform(D[f][0] - rB/lat_dist, D[f][0] + rB/lat_dist),
                   random.uniform(D[f][1] - rB/lat_dist, D[f][1] + rB/lat_dist))

    else:
        f = random.choice([i for i in range(len(D))])
        return f, (random.uniform(D[f][0] - rB/lon_dist, D[f][0] + rB/lon_dist),
                   random.uniform(D[f][1] - rB/lon_dist, D[f][1] + rB/lon_dist))


def place_node_in_zone(D, f):

    global lat_dist, lon_dist, rB
    return (random.uniform(D[f][0] - rB / lat_dist, D[f][0] + rB / lat_dist),
            random.uniform(D[f][1] - rB / lon_dist, D[f][1] + rB / lon_dist))


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


def cutoff(AL, W, pt):

    global DF

    if DF == None:
        return AL

    AL_new = []
    for k in AL:

        dd = math.sqrt(math.pow((W[k][0] - pt[0]), 2) + math.pow((W[k][1] - pt[1]), 2))

        if dd <= DF:
            AL_new.append(k)

    return AL_new


def waypoints(Xlim, Ylim, how_many):

    D = {i: (random.uniform(0, Xlim), random.uniform(0, Ylim)) for i in range(how_many)}
    return D


def find_dist(x1, y1, x2, y2):

    global mTOm, lat_dist, lon_dist
    return math.sqrt(math.pow(x2 - x1, 2) * lat_dist + math.pow(y2 - y1, 2) * lon_dist) * mTOm


class Node(object):

    def __init__(self, env, ID, waypoints, my_coor):

        global T

        self.ID = ID
        self.env = env

        # Neighbor list
        self.nlist = []

        self.old_coor = None

        self.waypoints = waypoints

        self.my_coor = my_coor

        self.start = True

        self.recBuf = simpy.Store(env, capacity = recBufferCapacity)

        # Time instant
        self.ti = random.randint(0, PT - 1)

        # List of events detected by system
        self.buffer = []

        # Fog's view of the network
        self.myG = nx.Graph()

        # Dictionary of fog node coordinates
        self.FC = {u: None for u in range(eG)}

        # Proximity Fitness
        self.PC = {}

        # Node motif centrality
        self.NMC = {int(self.ID[2:]): 0}

        # Find neighbors initially
        self.scan_neighbors()

        # List of new nodes in the neighborhood
        self.JReqs = []

        # Score for data forwarding
        self.SD = {u: 0 for u in range(eG)}

        # List of events detected by system
        self.events = []

        if 'E-' in self.ID:
            self.my_waypoint = waypoints[int(self.ID[2:])]
            # self.rE = 1000.0
            self.rE = 150.0

            self.temp = None
            self.next_hop = None
            self.env.process(self.send())
            self.env.process(self.receive())

        if self.ID == 'E-1':
            self.globalG = nx.Graph()
            self.env.process(self.time_increment())

        if 'EG' in self.ID:
            self.env.process(self.genEvent())

        if 'M-' in self.ID:
            self.my_waypoint = waypoints[int(self.ID[2:])]
            self.rE = 250.0
            self.active = int(self.ID[2:]) % 5 + 1
            self.prompt = 1

            self.preference = 0
            self.observations = 1

            self.env.process(self.sense())
            self.env.process(self.send())

    def sense(self):

        global baseE, L, sensing_range_mobile, eG, mG, senseE, prompt_incentive, Tot
        while self.rE > baseE:

            if T % frequencyEvent == 0:
                self.rE = self.rE - senseE
                # Sense event in the vicinity
                for each in entities[eG + mG].events:
                    if find_dist(each[1][0], each[1][1], self.my_coor[0], self.my_coor[1]) <= sensing_range_mobile:
                        # if random.uniform(0, 1) < 1.0 - np.random.power(3, 1):
                        if random.choice([0, 1]) == 1:
                            self.recBuf.put(each)
                            Tot.append(each)
                            self.prompt += prompt_incentive
                        else:
                            self.prompt -= prompt_incentive
                            self.prompt = max(1, self.prompt)

                # Remove old events (i.e. which has been in the system for at least 'frequencyEvent' time) from list
                self.events = self.updateEventList(self.events)
                self.move()

            yield self.env.timeout(minimumWaitingTime)

    def scan_neighbors(self):

        global eG, sensing_range, entities, D, scanE, T

        self.nlist = []

        if 'E-' in self.ID:

            if T > 1:
                self.rE = self.rE - scanE

            self.myG = nx.Graph()
            self.myG.add_node(int(self.ID[2:]))

            if self.start:
                for u in range(eG):
                    next_coor = place_node_in_zone(D, self.waypoints[u])
                    if find_dist(self.my_coor[0], self.my_coor[1], next_coor[0], next_coor[1]) <= sensing_range:
                        self.nlist.append(u)
                        self.myG.add_edge(int(self.ID[2:]), u)

                self.start = False

            else:
                for u in range(eG):
                    if find_dist(self.my_coor[0], self.my_coor[1], entities[u].my_coor[0], entities[u].my_coor[1]) <= sensing_range:
                        self.nlist.append(u)
                        self.myG.add_edge(int(self.ID[2:]), u)

            self.nlist = [u for u in self.nlist if u != int(self.ID[2:])]

    def find_NMC(self):

        self.NMC = {u: 0.0 for u in self.myG.nodes()}
        self.NMC[int(self.ID[2:])] = 0

        for u in self.myG.nodes():
            for v in self.myG.nodes():
                if v <= u:
                    continue

                for w in self.myG.nodes():
                    if w <= v:
                        continue
                    if self.myG.has_edge(u, v) and self.myG.has_edge(v, w) and self.myG.has_edge(u, w):
                        self.NMC[u] += 1
                        self.NMC[v] += 1
                        self.NMC[w] += 1

    def proximity_fitness(self):

        self.PC = {u: 0 for u in range(eG)}

        if len(self.myG.nodes()) > 1:
            L1 = [u for u in self.myG.nodes() if self.myG.has_edge(int(self.ID[2:]), u)]
            for u in L1:
                self.PC[u] += 1.0 if find_dist(entities[u].my_coor[0], entities[u].my_coor[1], BC[0], BC[1]) <= sensing_range else 0.0

                L2 = [w for w in entities[u].myG.nodes() if self.myG.has_edge(u, w)]
                for w in L2:
                    if w == u:
                        continue
                    self.PC[u] += WF * float(sensing_range)/float(max(sensing_range, find_dist(entities[u].my_coor[0], entities[u].my_coor[1], BC[0], BC[1])))

    def send(self):

        global T, pr_fg_E, fg_fg_E, baseE, V1, V2, Rec

        while True:

            if 'E-' in self.ID and self.rE > baseE:
                # Send a leave message (LM) to all current neighbors asking
                # them to remove me from their neighbor lists
                if T % PT == 0 and not self.start:
                    for u in self.nlist:
                        entities[u].buffer.append(['LM', int(self.ID[2:])])
                        self.myG = nx.Graph()
                        self.myG.add_node(int(self.ID[2:]))
                        self.nlist = []

                    self.move()

                # Send a join request (JReq) to the new neighbor nodes
                # informing them of my existence
                if T % PT == (self.ti + 1) % PT:
                    self.scan_neighbors()
                    for u in self.nlist:
                        entities[u].buffer.append(['JReq', (int(self.ID[2:]), self.my_coor)])

                # Send a join response (JRes) to the new neighbor node
                # informing them about my neighbors
                if T % PT == (self.ti + 2) % PT:
                    for m in self.JReqs:
                        entities[m[1][0]].buffer.append(['JRes', int(self.ID[2:]), [(u, entities[u].my_coor) for u in self.nlist]])

                # Send a update request message (UM) to all current neighbors
                # informing them of my neighbors
                if T % PT == (self.ti + 3) % PT:
                    for u in self.nlist:
                        entities[u].buffer.append(['UM', int(self.ID[2:]), [(u, entities[u].my_coor) for u in self.nlist]])

                # Filter out redundant event data and send to next hop
                L = []

                while len(self.recBuf.items) > 0:
                    item = yield self.recBuf.get()
                    if item not in L:
                        L.append(item)

                # Send data to next gop or base station
                if len(list(self.SD.keys())) > 0:

                    if find_dist(self.my_coor[0], self.my_coor[1], BC[0], BC[1]) <= sensing_range:
                        for item in L:
                            entities[self.next_hop].recBuf.put(item)
                            self.rE -= fg_fg_E
                            Rec.append(item)
                        self.next_hop = eG + mG + 1
                    else:
                        # self.next_hop = random.choice(list(self.SD.keys()))
                        self.next_hop = max(self.SD, key = self.SD.get)

                        # Next hop based on global graph
                        # self.next_hop = None
                        # for u in entities[1].globalG.nodes():
                        #     if u == 2000:
                        #         continue
                        #     if entities[1].globalG.has_edge(int(self.ID[2:]), u):
                        #         if nx.has_path(entities[1].globalG, u, 2000):
                        #             d = nx.shortest_path_length(entities[1].globalG, u, 2000)
                        #             if self.next_hop is None or d < nx.shortest_path_length(entities[1].globalG, self.next_hop, 2000):
                        #                 self.next_hop = u
                        #
                        # if self.next_hop is None:
                        #     self.next_hop = random.choice(list(self.SD.keys()))

                    # # Result: proximity vs distance
                    # if T == 45:
                    #
                    #     d = []
                    #     for w in entities[self.next_hop].myG.nodes():
                    #         d.append(find_dist(entities[w].my_coor[0], entities[w].my_coor[1], BC[0], BC[1]))
                    #
                    #     V1[self.next_hop] += 1
                    #     V2[self.next_hop].append(np.mean([v/max(d) for v in d]))
                    #
                    #     for u in self.myG.nodes():
                    #         if u != self.next_hop:
                    #
                    #             d = []
                    #             for w in entities[self.next_hop].myG.nodes():
                    #                 d.append(find_dist(entities[w].my_coor[0], entities[w].my_coor[1], BC[0], BC[1]))
                    #
                    #             V2[u].append(np.mean([v/max(d) for v in d]))

                    for item in L:
                        entities[self.next_hop].recBuf.put(item)
                        self.rE -= fg_fg_E

            if 'M-' in self.ID and self.rE > baseE:

                # Sense fog nodes in vicinity
                nearest_fog = None
                for i in range(eG):
                    if int(self.ID[2:]) == i:
                        continue
                    if nearest_fog is None or find_dist(entities[eG].my_coor[0], entities[eG].my_coor[1], self.my_coor[0], self.my_coor[1]) <= find_dist(entities[nearest_fog].my_coor[0], entities[nearest_fog].my_coor[1], self.my_coor[0], self.my_coor[1]):
                        nearest_fog = i

                if nearest_fog is not None:
                    while len(self.recBuf.items) > 0:
                        item = yield self.recBuf.get()
                        item = item[:-1] + [T]
                        entities[nearest_fog].recBuf.put(item)
                        self.rE -= pr_fg_E

                self.check_knockout()

            yield self.env.timeout(minimumWaitingTime)

    def data_forwarding_score(self):

        global w_e, w_p, w_m, entities

        self.SD = {}
        self.nlist = [u for u in self.myG.nodes() if self.myG.has_edge(int(self.ID[2:]), u) and entities[u].rE > 0 and u != int(self.ID[2:])]

        for u in self.nlist:
            self.SD[u] = 0
            self.SD[u] += w_e * (entities[u].rE + 0.01)/(np.sum([entities[v].rE + 0.01 for v in self.nlist]))
            self.SD[u] += w_m * (self.NMC[u] + 0.01)/(np.sum([self.NMC[v] + 0.01 for v in self.nlist]))
            self.SD[u] += w_p * (self.PC[u] + 0.01)/(np.sum([self.PC[v] + 0.01 for v in self.nlist]))

            # Random
            # self.SD[u] += w_m * (1.0 + 0.01)/(np.sum([1.0 + 0.01 for v in self.nlist]))

    def receive(self):

        global T, eT, V1

        while True:

            # When you receive a leave message
            # update your neighbor list
            LMs = [m for m in self.buffer if 'LM' in m]
            self.myG.remove_nodes_from([m[1] for m in LMs if m[1] in self.myG.nodes()])

            self.buffer = [m for m in self.buffer if m not in LMs]

            # Accept a join request (JReq) from a new neighbor
            # and update G
            self.JReqs = [m for m in self.buffer if 'JReq' in m]

            for m in self.JReqs:
                self.myG.add_edge(int(self.ID[2:]), m[1][0])
                self.FC[m[1][0]] = m[1][1]

            self.buffer = [m for m in self.buffer if m not in self.JReqs]

            # When I receive a join response (JRes) message from a neighbor
            # containing his neighbor information, I update my graph
            JRess = [m for m in self.buffer if 'JRes' in m]
            for m in JRess:
                self.myG.add_edges_from([(m[1], u) for (u, c) in m[2]])

                for (u, c) in m[2]:
                    self.FC[u] = c

            self.buffer = [m for m in self.buffer if m not in JRess]

            # When I receive a update request message (UM) from a neighbor
            # I delete all his prior neighbors of that node and add his current neighbors to the graph
            UMs = [m for m in self.buffer if 'UM' in m]
            for m in UMs:

                # Update graph with new 2 hop neighbor information
                if m[1] in self.myG.nodes():
                    self.myG.remove_node(m[1])
                self.myG.add_node(m[1])

                for (u, c) in m[2]:
                    self.myG.add_edge(m[1], u)
                    self.FC[u] = c

            self.buffer = [m for m in self.buffer if m not in UMs]

            if self.rE > baseE:

                self.proximity_fitness()
                self.find_NMC()
                self.data_forwarding_score()

            yield self.env.timeout(minimumWaitingTime)

    def move(self):

        global Xlim, Ylim, D, a, G, step, baseE

        if self.rE > baseE:

            if 'E-' in self.ID:
                self.old_coor = self.my_coor

                # LATP
                self.my_waypoint, self.my_coor = latp(self.my_coor, D, self.my_waypoint)

                # SNT
                # self.my_waypoint, self.my_coor = SNT(self.my_coor, D, G, self.waypoints, int(self.ID[2:]))

                # Orbit
                # self.my_waypoint, self.my_coor = orbit(self.my_waypoint, D)

            else:
                # new_x, new_y = rw_angle(self.my_coor[0], self.my_coor[1], Xlim, Ylim, step)
                self.my_waypoint, self.my_coor = rw_angle(self.my_waypoint, D)

        self.scan_neighbors()

    def time_increment(self):

        global T, Tracker, eG, Correlate, G, tm, V1, V2, eT

        while True:

            T = T + 1
            tm = tm + 0.002
            self.find_global_graph()

            if T > 2:
                Tracker.append([(entities[u].my_coor[0], entities[u].my_coor[1], entities[u].NMC[u]) for u in range(eG)])

            # Motif Centrality
            # L = list(entities[0].NMC.values())
            # try:
            #     V2.append(entities[0].NMC[entities[0].next_hop])
            #     V1.append(np.mean(L))
            # except:
            #     V1.append(np.mean(L))
            #     V2.append(np.mean(L))

            # print (T)

            # Energy aware
            # L = [1000.0 - entities[u].rE for u in range(eG)]
            # V1.append((np.mean(L), np.std(L)))

            # Latency
            Z = list(entities[eG + mG + 1].recBuf.items)
            t = len(Z)
            Z = Z[eT + 1:]
            V1.extend([T - m[-1] for m in Z])
            eT = t - 1

            # Data filtering
            # if T > 1:
            #     V1.append(sum([entities[u].temp[0] for u in range(eG)]))
            #     V2.append(sum([entities[u].temp[1] for u in range(eG)]))

            yield self.env.timeout(minimumWaitingTime)

    def genEvent(self):

        global T, Duration, frequencyEvent, globalEventCounter, Xlim, Ylim

        while True:

            # Generate new events
            if T % frequencyEvent == 0:

                for i in range(how_many_events):

                    # Random event
                    new_event = [globalEventCounter, (random.uniform(Xlim[0], Xlim[1]), random.uniform(Ylim[0], Ylim[1])), T]

                    # Saved event
                    # new_event = EVENTS[int(T/frequencyEvent)][i]

                    self.events.append(new_event)
                    globalEventCounter += 1

            # Remove old events (i.e. which has been in the system for at least 'frequencyEvent' time) from list
            self.events = self.updateEventList(self.events)

            yield self.env.timeout(minimumWaitingTime)

    def updateEventList(self, L):

        global frequencyEvent, T, recur

        remove_indices = []
        for each in L:
            if each[2] <= T - recur:
                remove_indices.append(L.index(each))

        return [i for j, i in enumerate(L) if j not in remove_indices]

    def check_knockout(self):

        global w_ac, w_en, w_pr, tm, T

        self.preference = w_ac * self.active/max([entities[u].active for u in range(eG, eG + mG)]) + w_en * self.rE / max([entities[u].rE for u in range(eG, eG + mG)]) + w_pr * self.prompt / sum([entities[u].prompt for u in range(eG, eG + mG)])
        # k = 4.0/(self.observations + 1)
        # self.preference = k * increase + (1 - k) * self.preference
        # self.observations += 1

        if self.preference < tm:
            self.rE = 0

    def find_global_graph(self):

        global sensing_range, eG, baseE
        self.globalG = nx.Graph()
        self.globalG.add_node(2000)
        L = [u for u in range(eG) if entities[u].rE > baseE]
        self.globalG.add_nodes_from(L)

        for u in L:
            for v in L:
                if v <= u:
                    continue
                if find_dist(entities[u].my_coor[0], entities[u].my_coor[1], entities[v].my_coor[0], entities[v].my_coor[1]) <= sensing_range:
                    self.globalG.add_edge(u, v)

            if find_dist(entities[u].my_coor[0], entities[u].my_coor[1], BC[0], BC[1]) <= sensing_range:
                self.globalG.add_edge(u, 2000)

iterate = 1
PDR = []
EE = []
LAT = []
for z in range(iterate):
    print ('Iterate ', z)

    # Number of fog nodes
    eG = 100

    # base station event data tracker
    eT = -1

    # Friendship network
    # G = nx.erdos_renyi_graph(n = eG, p = 0.1, directed = False)
    # nx.write_gml(G, 'friendship.gml')

    G = nx.read_gml('friendship.gml')

    mapping = {u: int(u) for u in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    # Visualization purposes
    Tracker = []

    # Number of mobile nodes
    mG = 500

    # List of activeness of mobile devices
    active_list = {}

    T = 0

    # Move how often
    mho = 3

    # Mobility velocity
    min_step = 1
    max_step = 10
    step = 5

    # Base energy level
    baseE = 10.0

    # Mobility weighing parameter and LATP waypoint cutoff
    a = 2
    DF = 100

    frequencyEvent = 2
    globalEventCounter = 0

    # Unit for device memory
    recBufferCapacity = 1000

    # Simulation Duration
    Duration = 90

    # Pause time
    PT = 5

    # Fog sensing range
    sensing_range = 500.0

    # Mobile sensing range
    sensing_range_mobile = 100.0

    # Simulation range
    Xlim = [40.0, 41.0]
    Ylim = [73.0, 74.5]

    # Define waypoints
    how_many = 50
    minimumWaitingTime = 1

    E = mobile()
    D = []
    for p in E.keys():
        L = E[p]
        D.extend(L)

    # Location of Base station
    # BC = (np.sum(Xlim)/2, np.sum(Ylim)/2)
    BC = np.mean(D, axis = 0)
    # Proximity weighing factor
    WF = 0.5

    # Miles to meters
    mTOm = 1609.34

    # Probability of intra-zone mobility (for ORBIT)
    pr = 0.9

    # Promptness increment/decrement
    prompt_incentive = 5

    # Distance between two latitude and longitudes
    lat_dist = 69.0
    lon_dist = 54.6

    # Correlation between the proximity and score
    Correlate = []

    # How many events
    how_many_events = 100

    # How often should event stay in system
    recur = 10

    # Weighing factor (for LATP)
    aLATP = 1.2

    # Area of NYC
    A = 302.6

    # Number of neighborhoods
    nB = 59

    # Data forwarding weight (FOG)
    w_e, w_p, w_m = 0.33, 0.33, 0.33

    # Smart device weight (FOG)
    w_ac, w_pr, w_en = 0.2, 0.6, 0.2

    # radius of a neighborhood
    AB = A/nB
    rB = math.sqrt(AB/math.pi)

    # Create Simpy environment and assign nodes to it.
    env = simpy.Environment()

    # Choice of waypoint
    Coor = [int(i % 59) for i in range(eG + mG + 2)]

    minimumWaitingTime = 1

    # Number of mobile nodes
    mG = 500

    # Sense event data energy
    senseE = 0.05

    # Scan event data energy
    scanE = 3.68

    # Peer to FOG data transfer energy
    pr_fg_E = 0.137

    # Peer to FOG data transfer energy
    fg_fg_E = 0.37

    # Threshold knockout mobile smart device
    tm = 0.1

    # V1 = [0 for u in range(eG)]
    # V2 = [[] for u in range(eG)]

    V1 = []
    V2 = []

    # Total Generated
    Tot = []
    Rec = []

    # Create Simpy environment and assign nodes to it.
    env = simpy.Environment()
    entities = []

    for i in range(eG + mG + 2):

        if i < eG:
            # Edge device
            entities.append(Node(env, 'E-' + str(i), Coor, place_node_in_zone(D, Coor[i])))

        elif i < eG + mG:
            # Mobile device
            entities.append(Node(env, 'M-' + str(i), Coor, place_node_in_zone(D, Coor[i])))

        elif i == eG + mG:
            # Event generator
            entities.append(Node(env, 'EG-' + str(i), None, None))

        else:
            # Base station
            entities.append(Node(env, 'BS-' + str(i), Coor, BC))

    env.run(until = Duration)

    pdr = find_pdr(Tot, Rec)
    PDR.append(pdr)

    lat = np.mean(V1)
    LAT.append(lat)

    ee = np.mean([150.0 - entities[u].rE for u in range(eG)])
    EE.append(ee)

    print (len(Tot), len(Rec), np.mean(PDR), np.mean(EE), np.mean(LAT))
    pickle.dump(Tracker, open('Tracker.p', 'wb'))
    # print (V1)
print ('Packet Delivery Ratio:', np.mean(PDR), np.std(PDR))
print ('Consumed Energy:', np.mean(EE), np.std(EE))
print ('Average latency:', np.mean(LAT), np.std(LAT))

