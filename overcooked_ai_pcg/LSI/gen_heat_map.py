import matplotlib
matplotlib.use("agg")
matplotlib.rcParams.update({'font.size': 12})
import matplotlib.pyplot as plt

import numpy as np
import os
import cv2
import csv
import glob
import toml
import argparse
import seaborn as sns
import pandas as pd
from itertools import product
from overcooked_ai_pcg import LSI_IMAGE_DIR, LSI_LOG_DIR
from overcooked_ai_pcg.helper import read_in_lsi_config

# handled by command line argument parser
FEATURE1_LABEL = None # label of the first feature to plot
FEATURE2_LABEL = None # label of the second feature to plot
IMAGE_TITLE = None # title of the image, aka name of the algorithm
STEP_SIZE = None # step size of the animation to generate
LOG_FILE_NAME = None # filepath to the elite map log file
NUM_FEATURES = None  # total number of features (bc) used in the experiment
ROW_INDEX = None  # index of feature 1 to plot
COL_INDEX = None  # index of feature 2 to plot

# max and min value of fitness
FITNESS_MIN = -100
FITNESS_MAX = 100


def createRecordList(mapData, mapDims):
    recordList = []
    indexPairs = [(x, y)
                  for x, y in product(range(mapDims[0]), range(mapDims[1]))]
    for cellData in mapData:
        # splite data from csv file
        splitedData = cellData.split(":")
        cellRow = int(splitedData[ROW_INDEX])
        cellCol = int(splitedData[COL_INDEX])
        nonFeatureIdx = NUM_FEATURES
        indID = int(splitedData[nonFeatureIdx])
        fitness = float(splitedData[nonFeatureIdx + 1])
        f1 = float(splitedData[nonFeatureIdx + 2 + ROW_INDEX])
        f2 = float(splitedData[nonFeatureIdx + 2 + COL_INDEX])

        data = [cellRow, cellCol, indID, fitness, f1, f2]
        recordList.append(data)
        indexPairs.remove((cellRow, cellCol))

    # Put in the blank cells
    for x, y in indexPairs:
        recordList.append([x, y, 0, np.nan, 0, 0])

    return recordList


def createRecordMap(dataLabels, recordList):
    dataDict = {label: [] for label in dataLabels}
    for recordDatum in recordList:
        for i in range(len(dataLabels)):
            dataDict[dataLabels[i]].append(recordDatum[i])
    return dataDict


def createImage(rowData, filename):
    mapDims = tuple(map(int, rowData[0].split('x')))
    mapData = rowData[1:]

    dataLabels = [
        'CellRow',
        'CellCol',
        # 'CellSize',
        'IndividualId',
        # 'WinCount',
        'Fitness',
        'Feature1',
        'Feature2',
    ]

    recordList = createRecordList(mapData, mapDims)
    dataDict = createRecordMap(dataLabels, recordList)

    recordFrame = pd.DataFrame(dataDict)

    # Write the map for the cell fitness
    fitnessMap = recordFrame.pivot(index=dataLabels[1],
                                   columns=dataLabels[0],
                                   values='Fitness')
    fitnessMap.sort_index(level=1, ascending=False, inplace=True)
    with sns.axes_style("white"):
        numTicks = 5  #11
        numTicksX = mapDims[0] // numTicks + 1
        numTicksY = mapDims[1] // numTicks + 1
        plt.figure(figsize=(3, 3))
        g = sns.heatmap(
            fitnessMap,
            annot=False,
            fmt=".0f",
            xticklabels=numTicksX,
            yticklabels=numTicksY,
            vmin=FITNESS_MIN,
            vmax=FITNESS_MAX,
        )
        fig = g.get_figure()
        # plt.axis('off')
        g.set(title=IMAGE_TITLE, xlabel=FEATURE1_LABEL, ylabel=FEATURE2_LABEL)
        plt.tight_layout()
        fig.savefig(filename)
    plt.close('all')


def createImages(stepSize, rows, filenameTemplate):
    for endInterval in range(stepSize, len(rows), stepSize):
        print('Generating : {}'.format(endInterval))
        filename = filenameTemplate.format(endInterval)
        createImage(rows[endInterval], filename)


def createMovie(folderPath, filename):
    globStr = os.path.join(folderPath, '*.png')
    imageFiles = sorted(glob.glob(globStr))

    # Grab the dimensions of the image
    img = cv2.imread(imageFiles[0])
    imageDims = img.shape[:2][::-1]

    # Create a video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frameRate = 30
    video = cv2.VideoWriter(os.path.join(folderPath, filename), fourcc,
                            frameRate, imageDims)

    for imgFilename in imageFiles:
        img = cv2.imread(imgFilename)
        video.write(img)

    video.release()


def generateAll(logPath):
    with open(logPath, 'r') as csvfile:
        # Read all the data from the csv file
        allRows = list(csv.reader(csvfile, delimiter=','))

        # Clear out the previous images
        tmpImageFolder = LSI_IMAGE_DIR
        if not os.path.exists(tmpImageFolder):
            os.mkdir(tmpImageFolder)
        for curFile in glob.glob(tmpImageFolder + '/*'):
            os.remove(curFile)

        # generate the movie
        template = os.path.join(tmpImageFolder, 'grid_{:05d}.png')
        createImages(STEP_SIZE, allRows[1:], template)
        movieFilename = 'fitness.avi'
        createMovie(tmpImageFolder, movieFilename)

        # Create the final image we need
        imageFilename = 'fitnessMap.png'
        createImage(allRows[-1], os.path.join(LSI_IMAGE_DIR, imageFilename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        help='path to the experiment config file',
                        required=True)
    parser.add_argument('-f1',
                        '--feature1_idx',
                        help='index of the first feature to plot',
                        required=False,
                        default=0)
    parser.add_argument('-f2',
                        '--feature2_idx',
                        help='index of the second feature to plot',
                        required=False,
                        default=1)
    parser.add_argument('-s',
                        '--step_size',
                        help='step size of the animation to generate',
                        required=False,
                        default=5)
    parser.add_argument('-l',
                        '--log_file',
                        help='filepath to the elite map log file',
                        required=False,
                        default=os.path.join(LSI_LOG_DIR, "elite_map.csv"))
    opt = parser.parse_args()

    # read in the name of the algorithm and features to plot
    experiment_config, algorithm_config, elite_map_config = read_in_lsi_config(
        opt.config)
    features = elite_map_config['Map']['Features']

    # read in parameters
    NUM_FEATURES = len(features)
    ROW_INDEX = int(opt.feature1_idx)
    COL_INDEX = int(opt.feature2_idx)
    STEP_SIZE = int(opt.step_size)
    IMAGE_TITLE = algorithm_config['name']
    FEATURE1_LABEL = features[ROW_INDEX]['name']
    FEATURE2_LABEL = features[COL_INDEX]['name']
    LOG_FILE_NAME = opt.log_file

    generateAll(LOG_FILE_NAME)
