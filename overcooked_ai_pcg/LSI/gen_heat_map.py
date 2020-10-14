import matplotlib
matplotlib.use("agg")
matplotlib.rcParams.update({'font.size': 12})
import matplotlib.pyplot as plt

import numpy as np
import os
import cv2
import csv
import glob
import seaborn as sns
import pandas as pd
from itertools import product
from overcooked_ai_pcg import LSI_IMAGE_DIR, LSI_LOG_DIR

feature1Label = 'pot_onion_shortest_dist'
feature2Label = 'pot_serve_shortest_dist'
image_title = 'MAP-Elites'
# image_title = 'CMA-ME'

stepSize = 5

logFilename = os.path.join(LSI_LOG_DIR, "elite_map.csv")


def createRecordList(mapData, mapDims):
    recordList = []
    indexPairs = [(x, y)
                  for x, y in product(range(mapDims[0]), range(mapDims[1]))]
    for cellData in mapData:
        splitedData = cellData.split(":")
        cellRow = int(splitedData[0])
        cellCol = int(splitedData[1])
        indID = int(splitedData[2])
        fitness = float(splitedData[3])
        f1 = float(splitedData[4])
        f2 = float(splitedData[5])

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
        g = sns.heatmap(fitnessMap,
                        annot=False,
                        fmt=".0f",
                        xticklabels=numTicksX,
                        yticklabels=numTicksY,
                        vmin=0,
                        vmax=40)
        fig = g.get_figure()
        # plt.axis('off')
        g.set(title=image_title, xlabel=feature2Label, ylabel=feature1Label)
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
        createImages(stepSize, allRows[1:], template)
        movieFilename = 'fitness.avi'
        createMovie(tmpImageFolder, movieFilename)

        # Create the final image we need
        imageFilename = 'fitnessMap.png'
        createImage(allRows[-1], os.path.join(LSI_IMAGE_DIR, imageFilename))


if __name__ == "__main__":
    generateAll(logFilename)
