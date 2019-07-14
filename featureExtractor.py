from PIL import Image, ImageFile
import glob
import  numpy as np, time
import cv2
import os
import matplotlib.pyplot as plt

def ImageDirectory():
    data = {}
    labels1 = open("data/colors.txt", "r+").readlines()
    for i in labels1:
        d = i.split("-")
        labels = d[0];
        images = d[1].split(",")
        data[labels] = images
    return data

def featureExtractor(path,k=" "):
    source_image = cv2.imread(path)
    chans = cv2.split(source_image)
    colors = ('r', 'g', 'b')
    features = []
    data_rgb = ''
    G = "";R = "";B = ""
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)
        Number = np.argmax(hist)
        if counter == 1:  ## opencv reads the images as BGR instead of RGB
            B = str(Number)
        elif counter == 2:
            G = str(Number)
        elif counter == 3:
            R = str(Number)
            if k != " ":
                data_rgb = R + ',' + G + ',' + B + ',' + k + '\n'
            else:
                data_rgb = R + ',' + G + ',' + B
            return data_rgb



def main(stat,test_image =" "):
    f_data=[]
    if stat == 1: ## for training set
        data = ImageDirectory()
        for k,v in data.items():
            for image in v :
                data = k.split()[0]+image
                path = "data/"+data.strip()
                f_data.append(featureExtractor(path,k)) # image pathway and color  as parameters
                with open('features.data', 'w+') as myfile:
                    for i in f_data:
                        myfile.write(i)

    elif stat == 2: # for single images test
        feature = featureExtractor(test_image)
        with open('test.data', 'w+') as myfile:
            myfile.write(feature)

main(1)
