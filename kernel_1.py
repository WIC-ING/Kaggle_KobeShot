import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Rectangle, Arc
from sklearn import mixture
from sklearn import ensemble
from sklearn import model_selection
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import log_loss
import time
import itertools
import operator

allData = pd.read_csv('data.csv')
data = allData[allData['shot_made_flag'].notnull()].reset_index()

# print(data['game_date'])



data['game_date_DT'] = pd.to_datetime(data['game_date'])
data['dayOfWeek']    = data['game_date_DT'].dt.dayofweek
data['dayOfYear']    = data['game_date_DT'].dt.dayofyear

#特征为：这一节所余下的秒数
data['secondsFromPeriodEnd'] = 60*data['minutes_remaining'] + data['seconds_remaining']
#特征为：这一节所走过的秒数
data['secondsFromPeriodStart'] = 60*(11-data['minutes_remaining']) + (60-data['seconds_remaining'])

#特征为：比赛开始走过秒数(加时赛另起时间)
data['secondsFromGameStart'] = (data['period'] <= 4).astype(int) * (data['period']-1)*12*60 \
                               + (data['period'] > 4).astype(int)*(data['period']-4)*5*60 \
                               + data['secondsFromPeriodStart']

# print(data.loc[:10, ['period', 'minutes_remaining', 'seconds_remaining', 'secondsFromPeriodEnd', 'secondsFromPeriodStart', 'secondsFromGameStart']])
# # print(data['game_date_DT'].dt)






#对老大的出手位置通过高斯混合模型进行聚类  -- 暂时注释掉
numGaussians = 13
gaussianMixtureModel = mixture.GaussianMixture(n_components=numGaussians,
                                               covariance_type='full',
                                               init_params='kmeans', n_init=50,
                                               verbose=0, random_state=5)

gaussianMixtureModel.fit(data.loc[:, ['loc_x', 'loc_y']])

#add the GMM cluster as a field in the dataset
data['shotLocationCluster'] = gaussianMixtureModel.predict(data.loc[:, ['loc_x', 'loc_y']])
# print(data['shotLocationCluster'])



def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
     ax = plt.gca()

    #Create the various parts of an NBA basketball court

    #Create the basketball hoop
    #Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0,0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    #The paint
    #Create the outer box of the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)

    #Create the inner box of the paint, width=15ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)

    #Create free throw bottom arc
    bottom_free_throw = Arc((0,142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')

    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0,0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    #Three point line
    # Create the side 3pt lines, they are 14ft long before the begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)

    # 3pt arc - center of arc will be the hoop, arc is 23'9"away from hoop
    # I just played around with the theta values until they lined up with the
    # threes
    three_arc = Arc((0,0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

        # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax


def Draw2DGaussians(gaussianMixtureModel, ellipseColors, ellipseTextMessages):
    fig, h = plt.subplots()
    for i, (mean, covarianceMatrix) in enumerate(zip(gaussianMixtureModel.means_, gaussianMixtureModel.covariances_)):
        # get the eigen vectors and eigen values of the covariance matrix
        v, w = np.linalg.eigh(covarianceMatrix)
        v = 2.5 * np.sqrt(v)  # go to units of standard deviation instead of variance

        # calculate the ellipse angle and two axis length and draw it
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        currEllipse = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=ellipseColors[i])
        currEllipse.set_alpha(0.5)
        h.add_artist(currEllipse)
        h.text(mean[0] + 7, mean[1] - 1, ellipseTextMessages[i], fontsize=13, color='blue')
#
# plt.rcParams['figure.figsize'] = (13, 10)
# plt.rcParams['font.size'] = 15
#
# ellipseTestMessages = [str(100*gaussianMixtureModel.weights_[x])[:4]+'%' for x in range(numGaussians)]
#
ellipseColors = ['red', 'green', 'purple', 'cyan', 'magenta', 'yellow', 'blue', 'orange', 'silver',
                 'maroon', 'lime', 'olive', 'brown', 'darkblue']
#
# Draw2DGaussians(gaussianMixtureModel, ellipseColors, ellipseTestMessages)
#
# draw_court(outer_lines=True); plt.ylim(-60, 440); plt.xlim(270, -270); plt.title('shot attempts')
#
# plt.show()


#画出投篮点分布图（散点图）
# plt.rcParams['figure.figsize'] = (13, 10)
# plt.rcParams['font.size']      = 15
#
# plt.figure(); draw_court(outer_lines=True); plt.ylim(-60, 440); plt.xlim(270, -270); plt.title('cluster assignment')
#
# plt.scatter(x=data['loc_x'], y=data['loc_y'],c=data['shotLocationCluster'], s=40, cmap='hsv', alpha=0.1)
# plt.show()

#画出每个出手点分布聚类的命中率
# plt.rcParams['figure.figsize'] = (13, 10)
# plt.rcParams['font.size'] = 15
#
variableCategories = data['shotLocationCluster'].value_counts().index.tolist()

clusterAccuracy = {}

for category in variableCategories:
    shotsAttempted = np.array(data['shotLocationCluster'] == category).sum()
    shotsMade = np.array(data.loc[data['shotLocationCluster'] == category, 'shot_made_flag'] == 1).sum()
    clusterAccuracy[category] = float(shotsMade)/shotsAttempted

# ellipseTestMessages = [str(100*clusterAccuracy[x])[:4]+'%' for x in range(numGaussians)]
# Draw2DGaussians(gaussianMixtureModel, ellipseColors, ellipseTestMessages)
# draw_court(outer_lines=True); plt.ylim(-60, 440); plt.xlim(270, -270);plt.title('shot accuracy')
# plt.show()

plt.rcParams['figure.figsize'] = (18, 10)
plt.rcParams['font.size'] = 18

sortedClustersByAccuracyTuple = sorted(clusterAccuracy.items(), key=operator.itemgetter(1), reverse=True)
sortedClustersByAccuracy = [ x[0] for x in sortedClustersByAccuracyTuple ]

binSizeInSeconds = 12
timeInUnitsOfBins = ((data['secondsFromGameStart']+0.0001)/binSizeInSeconds).astype(int)
# print(timeInUnitsOfBins)

locationInUintsOfClusters = np.array([sortedClustersByAccuracy.index(data.loc[x, 'shotLocationCluster']) for x in range(data.shape[0])])

# build a spatio-temporal histog
shotsAttempts = np.zeros((gaussianMixtureModel.n_components, 1+max(timeInUnitsOfBins)))
# print(shotsAttempts.shape)
for shot in range(data.shape[0]):
    shotsAttempts[locationInUintsOfClusters[shot], timeInUnitsOfBins[shot]] += 1

# make the y-axis have larger area so it will be more visible
shotsAttempts = np.kron(shotsAttempts, np.ones((5,1)))
# the locations of the period ends
vlinesList = 0.5001 + np.array([0, 12*60, 2*12*60, 3*12*60, 4*12*60, 4*12*60+5*60]).astype(int)/binSizeInSeconds


# plt.figure()
# plt.imshow(shotsAttempts, cmap='copper', interpolation='nearest'); plt.xlim(0, float(4*12*60+6*60)/binSizeInSeconds)
# plt.vlines(x=vlinesList, ymin=-0.5, ymax=shotsAttempts.shape[0]-0.5, colors='r')
# plt.xlabel('time from start of game [sec]'); plt.ylabel('cluster (sorted by accuracy)')
# plt.show()

def FactorizeCategoricalVariable(inputDB, categoricalVarName):
    opponentCategories = inputDB[categoricalVarName].value_counts().index.tolist()

    outputDB = pd.DataFrame()
    for category in opponentCategories:
        featureName = categoricalVarName + ': ' + str(category)
        outputDB[featureName] = (inputDB[categoricalVarName] == category).astype(int)

    return outputDB


featuresDB = pd.DataFrame()
featuresDB['homeGame'] = data['matchup'].apply(lambda x: 1 if (x.find('@') < 0) else 0)
featuresDB = pd.concat([featuresDB, FactorizeCategoricalVariable(data, 'opponent')], axis=1)
featuresDB = pd.concat([featuresDB, FactorizeCategoricalVariable(data, 'action_type')], axis=1)
featuresDB = pd.concat([featuresDB, FactorizeCategoricalVariable(data, 'shot_type')], axis=1)
featuresDB = pd.concat([featuresDB, FactorizeCategoricalVariable(data, 'combined_shot_type')], axis=1)
featuresDB = pd.concat([featuresDB, FactorizeCategoricalVariable(data, 'shot_zone_basic')], axis=1)
featuresDB = pd.concat([featuresDB, FactorizeCategoricalVariable(data, 'shot_zone_area')], axis=1)
featuresDB = pd.concat([featuresDB, FactorizeCategoricalVariable(data, 'shot_zone_range')], axis=1)
featuresDB = pd.concat([featuresDB, FactorizeCategoricalVariable(data, 'shotLocationCluster')], axis=1)

featuresDB['playoffGame'] = data['playoffs']
featuresDB['locX'] = data['loc_x']
featuresDB['locY'] = data['loc_y']
featuresDB['distanceFromBasket'] = data['shot_distance']
featuresDB['secondsFromPeriodEnd'] = data['secondsFromPeriodEnd']

featuresDB['dayOfWeek_cycX'] = np.sin(2 * np.pi * (data['dayOfWeek'] / 7))
featuresDB['dayOfWeek_cycY'] = np.cos(2 * np.pi * (data['dayOfWeek'] / 7))
featuresDB['timeOfYear_cycX'] = np.sin(2 * np.pi * (data['dayOfYear'] / 365))
featuresDB['timeOfYear_cycY'] = np.cos(2 * np.pi * (data['dayOfYear'] / 365))

labelsDB = data['shot_made_flag']




#作图观察科比在比赛期间出手频率以及命中率的分布和对比
# plt.rcParams['figure.figsize'] = (15, 10)
# plt.rcParams['font.size'] = 16
#
# binSizeInSeconds = 20
# timeBins = np.arange(0, 60*(4*12+3*5), binSizeInSeconds) + 0.01
#
# attemptsAsFunctionOfTime, b = np.histogram(data['secondsFromGameStart'], bins=timeBins)
#
# # print(data['shot_made_flag'])
#
# madeAttemptsAsFunctionOfTime, b = np.histogram(data.loc[data['shot_made_flag']==1, 'secondsFromGameStart'], bins=timeBins)
# attemptsAsFunctionOfTime[attemptsAsFunctionOfTime < 1] = 1
#
# # #命中率
# accuracyAsFunctionOfTime = madeAttemptsAsFunctionOfTime.astype(float)/attemptsAsFunctionOfTime
# accuracyAsFunctionOfTime[attemptsAsFunctionOfTime <= 50] = 0
#
# maxHeight = max(attemptsAsFunctionOfTime) + 30
# barWidth = 0.999*(timeBins[1] - timeBins[0])
#
# plt.figure()
#
# plt.subplot(2,1,1); plt.bar(timeBins[:-1], attemptsAsFunctionOfTime, align='edge', width=barWidth)
# plt.xlim((-20,3200)); plt.ylim((0,maxHeight)); plt.ylabel('attempts');plt.title(str(binSizeInSeconds) + ' second time bins')
# plt.vlines(x=[0, 12*60, 2*12*60, 3*12*60, 4*12*60, 4*12*60+5*60, 4*12*60+2*5*60, 4*12*60+3*5*60 ], ymin=0, ymax=maxHeight, colors='r')
#
# plt.subplot(2,1,2); plt.bar(timeBins[:-1],accuracyAsFunctionOfTime, align='edge', width=barWidth)
# plt.xlim((-20,3200)); plt.ylabel('accuracy');plt.xlabel('time [seconds from start of game]')
# plt.vlines(x=[0, 12*60, 2*12*60, 3*12*60, 4*12*60, 4*12*60+5*60, 4*12*60+2*5*60, 4*12*60+3*5*60 ], ymin=0, ymax=0.7, colors='r')
# plt.show()


#作图观察科比在比赛期间的出手频率分布图
#
# plt.rcParams['figure.figsize'] = (16, 16)
# plt.rcParams['font.size'] = 16
#
# binsSizes = [24, 12, 6]
# plt.figure()
# for k, binSizeInSeconds in enumerate(binsSizes):
#     timeBins = np.arange(0, 60*(4*12+3*5), binSizeInSeconds) + 0.01
#     attempsAsFunctionOfTime, b = np.histogram(data['secondsFromGameStart'], bins=timeBins)
#
#     maxHeight = max(attempsAsFunctionOfTime) + 30
#
#     barWidth = 0.999*(timeBins[1]-timeBins[0])
#     plt.subplot(len(binsSizes),1, k+1)
#     plt.bar(timeBins[:-1], attempsAsFunctionOfTime, align='edge', width=barWidth)
#     plt.title(str(binSizeInSeconds) + ' second time bins')
#     plt.vlines(x=[0, 12*60, 2*12*60, 3*12*60, 4*12*60, 4*12*60+5*60, 4*12*60+2*5*60, 4*12*60+3*5*60], ymin=0,ymax=maxHeight, colors='r')
#     plt.xlim((-20, 3200))
#     plt.ylim((0, maxHeight))
#     plt.ylabel('attempts')
#     plt.xlabel('time [seconds from start of game]')
#     plt.show()