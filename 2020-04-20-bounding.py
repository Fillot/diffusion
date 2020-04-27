"""
behavior is taken from https://doi.org/10.1016/j.bpj.2017.12.037
Amitai 2018, with binding site having to radii:
    capture radius a :  if the particle ends up at less than a of the binding site, 
                        its position becomes the binding site
    release radius e : upon release, particle gets put at distance e from the site,
                        with uniform angular probability

"""
from diffusion import simulator as sim
import numpy as np
import pandas as pd
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import matplotlib.pyplot as plt
import math
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from diffusion import utils
from diffusion import plots
from diffusion import analyse
from matplotlib.ticker import FormatStrFormatter
import pims
import imageio
import statsmodels.api as sm
import trackpy as tp

def isBound(position, bindingSiteList):
    """Returns the binding site to which the particle is bound
    Returns None if not bound"""
    point = Point(position)
    for site in bindingSiteList:
        #if any of the listed trap contains the point
        if (site.contains(point)):
            return site
    return None

def binding(pos, pLib, site, maxFrames):
    X = []
    Y = []
    xsite = site.centroid.x
    ysite = site.centroid.y
    #put in the first coordinates
    X.append(xsite)
    Y.append(ysite)

    currentFrame = 1
    while (currentFrame <= maxFrames):
        if (np.random.rand() < pLib):
            Xe, Ye = escape(site, 0.5)
            X.append(Xe)
            Y.append(Ye)  
            return X, Y
        else:
            X.append(xsite)
            Y.append(ysite)
        currentFrame += 1
    return X, Y

#TODO:verify that this gives a uniform distrib of alpha
def escape(site, escapeRadius):
    alpha = 2 * math.pi * np.random.rand()
    x = site.centroid.x + math.cos(alpha)
    y = site.centroid.y + math.sin(alpha)
    return x, y

#https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib
def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def getDiffusibityMap(fileName):
    Dmap = imageio.imread(fileName, pilmode='RGB')
    return Dmap[:,:,2]

def inverseVerticalIndex(index, numberOfPixels=256):
    new = (numberOfPixels - index)
    new = new.astype(int)
    return new

def translatePosToArray(position, x_max, x_min, res):
    #position 20,20 returns an index error of course
    #predicted but potentially annoying still
    increment = (x_max - x_min) / 256
    i_index = np.floor(position[1]/increment).astype(int)
    j_index = np.floor(position[0]/increment).astype(int)
    i_index = inverseVerticalIndex(i_index)
    return i_index, j_index

def getDiffusivity(position, map, x_max, x_min):
    i_index, j_index = translatePosToArray(position, x_max, x_min, 512)
    D = map[i_index, j_index]
    return D

bindingSiteList = []

for coord1 in range(5,16,3):
    for coord2 in range(5,16,3):
        bindingSiteList.append(sim.createCircle(0.3, (coord1,coord2), 10))

trapList = []

for coord1 in range(6,17,3):
    for coord2 in range(6,176,3):
        trapList.append(sim.createCircle(0.4, (coord1,coord2), 10))


x_min, y_min, x_max, y_max = 0,0,20,20
x_middle, y_middle = (x_max - x_min) / 2, (y_max - y_min) / 2
radius = x_max - x_middle
ROI= sim.createCircle(radius-1, (x_middle,y_middle), 100)
map = getDiffusibityMap('./diffusivity60.png')




# X = []
# Y = []
# startingPos = (10,10)
# X.append(startingPos[0])
# Y.append(startingPos[1])
# length = 100
# i=1
# while (i<length):
#     i += 1
#     #draw the displacement

#     D = getDiffusivity((X[-1], Y[-1]), map, x_max, x_min)
#     x, y = sim.displace(diffusivity=D)

#     if (ROI.contains(Point(X[-1] + x, Y[-1] + y))):
#         X.append(X[-1] + x)
#         Y.append(Y[-1] + y)
#     else:
#         #if we are not, we add the inverse of the displacement
#         X.append(X[-1] - x)
#         Y.append(Y[-1] - y)




#     site = isBound((X[-1], Y[-1]), bindingSiteList)
#     if (site is not None):
#         XinTrap, YinTrap = binding([X[-1], Y[-1]], 0.01, site, length - 1 - i)
#         X.extend(XinTrap)
#         Y.extend(YinTrap)
#         i += len(XinTrap)

# pos = np.column_stack((X,Y))

traj = sim.singleTraj(10, map, meanLength = 5, ROI = ROI, trapList = trapList, bindingSiteList = bindingSiteList)

# frame = np.arange(1,len(pos)+1)
# particle = np.ones(len(pos))
# traj = pd.DataFrame({'frame': frame,
#                     'x':pos[:,0], 
#                     'y':pos[:,1], 
#                     'particle': particle})

# for site in bindingSiteList:
#     x, y = site.exterior.xy
#     plt.plot(x,y, color='red')

# ROIx, ROIy = ROI.exterior.xy
# plt.plot(ROIx, ROIy)
# axes = plt.gca()
sim.drawGeometry(ROI=ROI, bindingSiteList=bindingSiteList, trapList=trapList)
# colorline(traj['x'], traj['y'], cmap=plt.get_cmap('viridis'), linewidth=1)
tp.plot_traj(traj)
# axes.set_ylim(-10,10)
# axes.set_xlim(-10,10)
# plt.show()
# plots.displacementHistogram(traj, bins=100)
# plots.HansenHeatMap(traj, 0, 1)

#THIS was to verify that having a constant proba of escape each frame results
#in a decreasing exponantial for the duration of the binding.
# trappedFor = []
# pLib = 0.5
# for i in range(100):
#     istrapped = True
#     trapped = 0
#     while (istrapped):
#         if (np.random.rand() < pLib):
#             istrapped = False
#         trapped += 1
#     trappedFor.append(trapped)

# plt.hist(trappedFor)

#2020-04-21: porting hansen's style of heatmap to this project.
# """Creation of a numpy array of the jump length
# Note: this is for a single continuous track. Code
# has to be adapted for multiple short tracks eventually"""
# vectors = utils.get_vectors(traj)
# jumpLength = utils.displacements(traj)
# angles = utils.angles(vectors)
# jumpLength = np.array(jumpLength) #required for digitized to work
# #problem (or not ?), angle_between outputs NaN when computing the angle 
# # between the 0 vector and something else
# # test = utils.angle_between((1,1),(0,0))

# """Creation of a 2D array used for the heat map. What the columns and lines
# represent in terms of actual dimension is set by the number of bins we want,
# and their range. So far, only linear segmentation"""
# bins = np.linspace(0, 10, 11)
# digitized = np.digitize(jumpLength, bins)
# bin_means = [jumpLength[digitized == i].mean() for i in range(1, len(bins))]

# heatmap30_150 = np.zeros((len(bins)+1,len(bins)+1))
# heatmapAllJumps = np.zeros((len(bins)+1,len(bins)+1))

# #iterating through the digitized array of jumps
# #if, for the current jump, angle(jump) anisotrope
# #then we'll tick the box corresponding to
# #line : bin to which this jump belongs,
# #column : bin to which the next jump belongs
# for jmp in range(len(digitized)-1):
#     if (angles[jmp]>150 or angles[jmp]<30):
#         heatmap30_150[digitized[jmp], digitized[jmp+1]] += 1
#         heatmapAllJumps[digitized[jmp], digitized[jmp+1]] +=1
#     else:
#         heatmapAllJumps[digitized[jmp], digitized[jmp+1]] += 1


# #this reads as follows : element-wise division of heatmap30_150 by heatmapAllJumps
# #                       out = where the results will be stored
# #                       where = do the division only where heatmapAllJumps is not 
# #                               0, otherwise just copy number from heatmap30_150
# heatmapAnisotropy = np.divide(heatmap30_150, heatmapAllJumps, out=np.zeros_like(heatmap30_150), where=heatmapAllJumps!=0)

# """
# Plot creation of the heatmap 
# """
# fig, ax = plt.subplots()
# im = ax.imshow(heatmapAnisotropy)
# fig.tight_layout()
# # We want to show all ticks...
# ax.set_xticks(np.arange(len(bins)))
# ax.set_yticks(np.arange(len(bins)))
# # ... and label them with the respective list entries
# #if we take numerical values for the labels, matplotlib 
# # wants to scale them to the axes
# labels = [str(bin) for bin in bins] 
# ax.set_xticklabels(labels)
# ax.set_yticklabels(labels)
# #Show only one decimal place
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")

# # Loop over data dimensions and create text annotations.
# # for i in range(len(bins)):
# #     for j in range(len(bins)):
# #         text = ax.text(j, i, heatmapAnisotropy[i, j],
# #                        ha="center", va="center", color="w")

# ax.set_title("Anisotropy of the jump depending on the jump length")
# fig.tight_layout()
# plt.show()


#Translate into index in array. For now we assume that bottom left
#corner is 0,0, but it should be able to translate regardless of
#the actual axes of the figure

results = analyse.emsd(traj)
# ax = results.plot(x="tau", y="msds", logx=True, logy=True, legend=False)
# ax.fill_between(results['tau'], results['msds'] - results['msds_std'], results['msds'] + results['msds_std'], alpha=0.2)
# plt.show()

minimum_obs = 900
# filtered = results[results['observations'] > minimum_obs]
filtered = results[:20]

# ax = filtered.plot(x="tau", y="msds", logx=True, logy=True, legend=False)
# ax.fill_between(filtered['tau'], filtered['msds'] - filtered['msds_std'], filtered['msds'] + filtered['msds_std'], alpha=0.2)
# plt.show()

#linear regression
X = filtered['tau']
model = sm.OLS(filtered['msds'], X).fit()
model.summary()

#calcul of D
msds = filtered['msds'].values
tau = filtered['tau'].values
Ds = msds / tau
Ds = Ds / 0.004
D_calcul = np.mean(Ds)
print(D_calcul)

#1) make it so the simulator doesn't crash if it doesn't have ROI or trap list or binding site list
#2) adapt all of this so that it work on multiple trajectories.
#3) notebook pour faire une démonstration