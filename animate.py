import numpy as np
import random, pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


Xlim = [40.5, 41.0]
Ylim = [73.75, 74.15]

eG = 50
rad = 150.0

lat_dist = 69.0

# Miles to meters
mTOm = 1609.34
factor_node = 0.002
factor = lat_dist/mTOm * factor_node

fig = plt.figure()
#fig.set_dpi(100)
fig.set_size_inches(7, 7)

ax = plt.axes(xlim = (Xlim[0], Xlim[1]), ylim = (Ylim[0], Ylim[1]))
Tracker = pickle.load(open("Tracker.p", "rb"))

p = []
colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(eG)]

for i in range(2 * eG):
    patch = plt.Circle((5, -5), factor, fc = 'r', alpha = 1.0)
    p.append(patch)

def init():

    for i in range(2 * eG):
        #patch.center = (5, 5)
        #ax.add_patch(patch)
        p[i].center = (i, i)
        p[i].radius = factor
        ax.add_patch(p[i])
    return p

def animate(frame):

    print ('***', len(frame))

    maxNMC = max([pt[2] for pt in frame])
    minNMC = min([pt[2] for pt in frame])

    for i in range(0, 2 * eG):

        # Plot the sensing range around node
        if i < eG:
            x = frame[i][0]
            y = frame[i][1]

            p[i].center = (x, y)
            p[i].radius = rad * factor
            p[i].fill = False
            p[i].set_color('gray')
            p[i].set_alpha(1.0)

        # Plot the node with its size = motif centrality
        else:
            x = frame[i - eG][0]
            y = frame[i - eG][1]

            p[i].center = (x, y)
            if maxNMC == minNMC:
                p[i].radius = factor_node
            else:
                p[i].radius = float(frame[i - eG][2] - minNMC)/float(maxNMC - minNMC) * 3.0 + 1.0
                p[i].radius *= factor_node
            p[i].set_color(colors[i - eG])

    return p


Tracker = pickle.load(open('Tracker.p', 'rb'))
print (Tracker)

anim = FuncAnimation(fig, animate, init_func = init, frames = [each for each in Tracker], blit = True, repeat = False, interval = 200)
# anim.save('bio_MCS.mp4', writer='ffmpeg', dpi=200)
# anim.save('bioMCS.gif', writer='imagemagick', fps=30)
# plt.show()

writergif = anim.FFMpegWriter(fps=30)
anim.save('bio_MCS.mp4', writer=writergif)