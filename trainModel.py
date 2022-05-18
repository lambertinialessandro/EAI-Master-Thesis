# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:36:00 2022

@author: lambe
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gc
import time
import random

import torch
import numpy as np
import matplotlib.pyplot as plt

from loadData import DataGeneretor, RandomDataGeneretor, DataLoader, attach2Torch
from buildModel import buildModel

import params
from EnumPreproc import EnumPreproc
from utility import PM, bcolors


def trainEpochPreprocessed():
    loss_train = {key: {"tot": [], "pose": [], "rot": []} for key in params.trainingSeries+["tot"]}

    PM.printI(bcolors.LIGHTYELLOW+"TRAINING"+bcolors.ENDC)
    model.train()
    model.training = True

    train_initT = time.time()
    for sequence in params.trainingSeries:
        PM.printI(bcolors.LIGHTGREEN+"Sequence: {}".format(sequence)+bcolors.ENDC)
        X, y = DataLoader(params.dir_Dataset, attach=False, suffixType=params.suffixType, sequence=sequence)
        train_numOfBatch = int(len(X)/params.BACH_SIZE)-1
        outputs = []

        for i in range(train_numOfBatch):
            inputs, labels = attach2Torch(
                    X[i*params.BACH_SIZE:(i+1)*params.BACH_SIZE],
                    y[i*params.BACH_SIZE:(i+1)*params.BACH_SIZE]
                )

            model.zero_grad()

            app_outputs = model(inputs[0])
            if params.DEVICE.type == 'cuda':
                outputs.append(app_outputs.cpu().detach().numpy())
            else:
                outputs.append(app_outputs.detach().numpy())

            totLoss = criterion(app_outputs, labels[0])
            poseLoss = criterion(app_outputs[0:3], labels[0][0:3]).item()
            rotLoss = criterion(app_outputs[3:6], labels[0][3:6]).item()

            totLoss.backward()
            optimizer.step()

            loss_train[sequence]["tot"].append(totLoss.item())
            loss_train[sequence]["pose"].append(poseLoss)
            loss_train[sequence]["rot"].append(rotLoss)
            PM.printProgressBarI(i, train_numOfBatch)

            del inputs, labels, app_outputs, totLoss, poseLoss, rotLoss
            gc.collect()
            torch.cuda.empty_cache()
        PM.printI("Loss Sequence: [tot: %.5f, pose: %.5f, rot: %.5f]"%(
            np.mean(loss_train[sequence]["tot"]),
            np.mean(loss_train[sequence]["pose"]),
            np.mean(loss_train[sequence]["rot"])
            ))
        del X
        gc.collect()
        torch.cuda.empty_cache()

        pts_yTrain = np.array([[0, 0, 0, 0, 0, 0]])
        pts_out = np.array([[0, 0, 0, 0, 0, 0]])
        for i in range(0, len(outputs)):
          for j in range(0, params.BACH_SIZE):
            pos = i*params.BACH_SIZE+j
            pts_yTrain = np.append(pts_yTrain, [pts_yTrain[pos] + y[pos]], axis=0)
            pts_out = np.append(pts_out, [pts_out[pos] + outputs[i][j]], axis=0)

        del outputs, y
        gc.collect()
        torch.cuda.empty_cache()

        plt.plot(pts_out[:, 0], pts_out[:, 2], color='red')
        plt.plot(pts_yTrain[:, 0], pts_yTrain[:, 2], color='blue')
        plt.legend(['out', 'yTest'])
        plt.show()

        ax = plt.axes(projection='3d')
        ax.plot3D(pts_out[:, 0], pts_out[:, 1], pts_out[:, 2], color='red')
        ax.plot3D(pts_yTrain[:, 0], pts_yTrain[:, 1], pts_yTrain[:, 2], color='blue')
        plt.legend(['out', 'yTest'])
        plt.show()
        del pts_yTrain, pts_out
        gc.collect()
        torch.cuda.empty_cache()
    train_elapsedT = time.time() - train_initT
    loss_train["tot"]["tot"].append(sum(
            [np.mean(loss_train[seq]["tot"]) for seq in params.trainingSeries]
        )/len(params.trainingSeries))
    loss_train["tot"]["pose"].append(sum(
        [np.mean(loss_train[seq]["pose"]) for seq in params.trainingSeries]
        )/len(params.trainingSeries))
    loss_train["tot"]["rot"].append(sum(
        [np.mean(loss_train[seq]["rot"]) for seq in params.trainingSeries]
        )/len(params.trainingSeries))
    PM.printI("Loss Train: [tot: %.5f, pose: %.5f, rot: %.5f] , time %.2fs"%(
        loss_train["tot"]["tot"][-1], loss_train["tot"]["pose"][-1], loss_train["tot"]["rot"][-1], train_elapsedT))

    return loss_train, train_elapsedT

def testEpochPreprocessed():
    loss_test = {key: {"tot": [], "pose": [], "rot": []} for key in params.testingSeries+["tot"]}

    PM.printI(bcolors.LIGHTYELLOW+"TESTING"+bcolors.ENDC)
    model.eval()
    model.training = False

    test_initT = time.time()
    for sequence in params.testingSeries:
        PM.printI(bcolors.LIGHTGREEN+"Sequence: {}".format(sequence)+bcolors.ENDC)
        X, y = DataLoader(params.dir_Dataset, attach=False, suffixType=params.suffixType, sequence=sequence)
        test_numOfBatch = int(len(X)/params.BACH_SIZE)-1
        outputs = []

        for i in range(test_numOfBatch):
            inputs, labels = attach2Torch(
                    X[i*params.BACH_SIZE:(i+1)*params.BACH_SIZE],
                    y[i*params.BACH_SIZE:(i+1)*params.BACH_SIZE]
                )

            #model.zero_grad()

            app_outputs = model(inputs[0])
            if params.DEVICE.type == 'cuda':
              outputs.append(app_outputs.cpu().detach().numpy())
            else:
              outputs.append(app_outputs.detach().numpy())

            totLoss = criterion(app_outputs, labels[0]).item()
            poseLoss = criterion(app_outputs[:, 0:3], labels[0][:, 0:3]).item()
            rotLoss = criterion(app_outputs[:, 3:6], labels[0][:, 3:6]).item()

            loss_test[sequence]["tot"].append(totLoss)
            loss_test[sequence]["pose"].append(poseLoss)
            loss_test[sequence]["rot"].append(rotLoss)
            PM.printProgressBarI(i, test_numOfBatch)

            del inputs, labels, app_outputs, totLoss, poseLoss, rotLoss
            gc.collect()
            torch.cuda.empty_cache()
        PM.printI("Loss Sequence: [tot: %.5f, pose: %.5f, rot: %.5f]"%(
            np.mean(loss_test[sequence]["tot"]),
            np.mean(loss_test[sequence]["pose"]),
            np.mean(loss_test[sequence]["rot"])
            ))
        del X
        gc.collect()
        torch.cuda.empty_cache()

        #print(y_test_det.shape)
        #print(outputs.shape)

        pts_yTest = np.array([[0, 0, 0, 0, 0, 0]])
        pts_out = np.array([[0, 0, 0, 0, 0, 0]])
        for i in range(0, len(outputs)):
            for j in range(0, params.BACH_SIZE):
                pos = i*params.BACH_SIZE+j
                pts_yTest = np.append(pts_yTest, [pts_yTest[pos] + y[pos]], axis=0)
                pts_out = np.append(pts_out, [pts_out[pos] + outputs[i][j]], axis=0)

        del outputs, y
        gc.collect()
        torch.cuda.empty_cache()

        plt.plot(pts_out[:, 0], pts_out[:, 2], color='red')
        plt.plot(pts_yTest[:, 0], pts_yTest[:, 2], color='blue')
        plt.legend(['out', 'yTest'])
        plt.show()

        ax = plt.axes(projection='3d')
        ax.plot3D(pts_out[:, 0], pts_out[:, 1], pts_out[:, 2], color='red')
        ax.plot3D(pts_yTest[:, 0], pts_yTest[:, 1], pts_yTest[:, 2], color='blue')
        plt.legend(['out', 'yTest'])
        plt.show()
        del pts_yTest, pts_out
        gc.collect()
        torch.cuda.empty_cache()
    test_elapsedT = time.time() - test_initT
    loss_test["tot"]["tot"].append(sum(
        [np.mean(loss_test[seq]["tot"]) for seq in params.testingSeries]
        )/len(params.testingSeries))
    loss_test["tot"]["pose"].append(sum(
        [np.mean(loss_test[seq]["pose"]) for seq in params.testingSeries]
        )/len(params.testingSeries))
    loss_test["tot"]["rot"].append(sum(
        [np.mean(loss_test[seq]["rot"]) for seq in params.testingSeries]
        )/len(params.testingSeries))
    PM.printI("Loss Test: [tot: %.5f, pose: %.5f, rot: %.5f] , time %.2fs"%(
        loss_test["tot"]["tot"][-1], loss_test["tot"]["pose"][-1], loss_test["tot"]["rot"][-1], test_elapsedT))

    return loss_test, test_elapsedT



def trainEpoch(imageDir, prepreocF):
    loss_train = {key: {"tot": [], "pose": [], "rot": []} for key in params.trainingSeries+["tot"]}

    PM.printI(bcolors.LIGHTYELLOW+"TRAINING"+bcolors.ENDC)
    model.train()
    model.training = True

    train_initT = time.time()
    for sequence in params.trainingSeries:
        PM.printI(bcolors.LIGHTGREEN+"Sequence: {}".format(sequence)+bcolors.ENDC)
        dg = DataGeneretor(sequence, imageDir, prepreocF, attach=True)
        train_numOfBatch = dg.numBatchImgs
        outputs = []
        pts_yTrain = np.array([[0, 0, 0, 0, 0, 0]])
        pts_out = np.array([[0, 0, 0, 0, 0, 0]])

        for inputs, labels, pos, nb in dg:
            for i in range(nb):
                torch.cuda.empty_cache()

                model.zero_grad()

                outputs = model(inputs[i])
                if params.DEVICE.type == 'cuda':
                    det_outputs = outputs.cpu().detach().numpy()
                    det_labels = labels[i].cpu().detach().numpy()
                else:
                    det_outputs = outputs.detach().numpy()
                    det_labels = labels[i].detach().numpy()

                totLoss = criterion(outputs, labels[i])
                poseLoss = criterion(outputs[0:3], labels[i][0:3]).item()
                rotLoss = criterion(outputs[3:6], labels[i][3:6]).item()

                totLoss.backward()
                optimizer.step()

                loss_train[sequence]["tot"].append(totLoss.item())
                loss_train[sequence]["pose"].append(poseLoss)
                loss_train[sequence]["rot"].append(rotLoss)

                for j in range(params.BACH_SIZE):
                    pts_yTrain = np.append(pts_yTrain, [pts_yTrain[-1] + det_labels[j]], axis=0)
                    pts_out = np.append(pts_out, [pts_out[-1] + det_outputs[j]], axis=0)
            del inputs, labels, det_outputs, det_labels, totLoss, poseLoss, rotLoss
            gc.collect()
            torch.cuda.empty_cache()
            PM.printProgressBarI(i, train_numOfBatch)
        del dg
        gc.collect()
        torch.cuda.empty_cache()

        PM.printI("Loss Sequence: [tot: %.5f, pose: %.5f, rot: %.5f]"%(
            np.mean(loss_train[sequence]["tot"]),
            np.mean(loss_train[sequence]["pose"]),
            np.mean(loss_train[sequence]["rot"])
            ))

        plt.plot(pts_out[:, 0], pts_out[:, 2], color='red')
        plt.plot(pts_yTrain[:, 0], pts_yTrain[:, 2], color='blue')
        plt.legend(['out', 'yTest'])
        plt.show()

        ax = plt.axes(projection='3d')
        ax.plot3D(pts_out[:, 0], pts_out[:, 1], pts_out[:, 2], color='red')
        ax.plot3D(pts_yTrain[:, 0], pts_yTrain[:, 1], pts_yTrain[:, 2], color='blue')
        plt.legend(['out', 'yTest'])
        plt.show()
        del pts_yTrain, pts_out
        gc.collect()
        torch.cuda.empty_cache()
    train_elapsedT = time.time() - train_initT
    loss_train["tot"]["tot"].append(sum(
        [np.mean(loss_train[seq]["tot"]) for seq in params.trainingSeries]
        )/len(params.trainingSeries))
    loss_train["tot"]["pose"].append(sum(
        [np.mean(loss_train[seq]["pose"]) for seq in params.trainingSeries]
        )/len(params.trainingSeries))
    loss_train["tot"]["rot"].append(sum(
        [np.mean(loss_train[seq]["rot"]) for seq in params.trainingSeries]
        )/len(params.trainingSeries))
    PM.printI("Loss Train: [tot: %.5f, pose: %.5f, rot: %.5f] , time %.2fs"%(
        loss_train["tot"]["tot"][-1], loss_train["tot"]["pose"][-1], loss_train["tot"]["rot"][-1], train_elapsedT))

    return loss_train, train_elapsedT

def testEpoch(imageDir, prepreocF):
    loss_test = {key: {"tot": [], "pose": [], "rot": []} for key in params.testingSeries+["tot"]}

    PM.printI(bcolors.LIGHTYELLOW+"TESTING"+bcolors.ENDC)
    model.eval()
    model.training = False

    test_initT = time.time()
    for sequence in params.testingSeries:
        PM.printI(bcolors.LIGHTGREEN+"Sequence: {}".format(sequence)+bcolors.ENDC)
        dg = DataGeneretor(sequence, imageDir, prepreocF, attach=True)
        test_numOfBatch = dg.numBatchImgs - 1

        pts_yTest = np.array([[0, 0, 0, 0, 0, 0]])
        pts_out = np.array([[0, 0, 0, 0, 0, 0]])

        for inputs, labels, pos, nb in dg:
            for i in range(nb):

                outputs = model(inputs[i])
                if params.DEVICE.type == 'cuda':
                    det_outputs = outputs.cpu().detach().numpy()
                    det_labels = labels[i].cpu().detach().numpy()
                else:
                    det_outputs = outputs.detach().numpy()
                    det_labels = labels[i].detach().numpy()

                totLoss = criterion(outputs, labels[i]).item()
                poseLoss = criterion(outputs[:, 0:3], labels[i][:, 0:3]).item()
                rotLoss = criterion(outputs[:, 3:6], labels[i][:, 3:6]).item()

                loss_test[sequence]["tot"].append(totLoss)
                loss_test[sequence]["pose"].append(poseLoss)
                loss_test[sequence]["rot"].append(rotLoss)

                for j in range(params.BACH_SIZE):
                    pts_yTest = np.append(pts_yTest, [pts_yTest[-1] + det_labels[j]], axis=0)
                    pts_out = np.append(pts_out, [pts_out[-1] + det_outputs[j]], axis=0)
            del inputs, labels, det_outputs, det_labels, totLoss, poseLoss, rotLoss
            gc.collect()
            torch.cuda.empty_cache()
            PM.printProgressBarI(pos, test_numOfBatch)
        del dg
        gc.collect()
        torch.cuda.empty_cache()

        PM.printI("Loss Sequence: [tot: %.5f, pose: %.5f, rot: %.5f]"%(
            np.mean(loss_test[sequence]["tot"]),
            np.mean(loss_test[sequence]["pose"]),
            np.mean(loss_test[sequence]["rot"])
            ))

        plt.plot(pts_out[:, 0], pts_out[:, 2], color='red')
        plt.plot(pts_yTest[:, 0], pts_yTest[:, 2], color='blue')
        plt.legend(['out', 'yTest'])
        plt.show()

        ax = plt.axes(projection='3d')
        ax.plot3D(pts_out[:, 0], pts_out[:, 1], pts_out[:, 2], color='red')
        ax.plot3D(pts_yTest[:, 0], pts_yTest[:, 1], pts_yTest[:, 2], color='blue')
        plt.legend(['out', 'yTest'])
        plt.show()
        del pts_yTest, pts_out
        gc.collect()
        torch.cuda.empty_cache()
    test_elapsedT = time.time() - test_initT
    loss_test["tot"]["tot"].append(sum(
        [np.mean(loss_test[seq]["tot"]) for seq in params.testingSeries]
        )/len(params.testingSeries))
    loss_test["tot"]["pose"].append(sum(
        [np.mean(loss_test[seq]["pose"]) for seq in params.testingSeries]
        )/len(params.testingSeries))
    loss_test["tot"]["rot"].append(sum(
        [np.mean(loss_test[seq]["rot"]) for seq in params.testingSeries]
        )/len(params.testingSeries))
    PM.printI("Loss Test: [tot: %.5f, pose: %.5f, rot: %.5f] , time %.2fs"%(
        loss_test["tot"]["tot"][-1], loss_test["tot"]["pose"][-1], loss_test["tot"]["rot"][-1], test_elapsedT),
        head="\n")

    return loss_test, test_elapsedT



def trainEpochRandom(imageDir, prepreocF):
    loss_train = {key: {"tot": [], "pose": [], "rot": []} for key in params.trainingSeries+["tot"]}

    PM.printI(bcolors.LIGHTYELLOW+"TRAINING"+bcolors.ENDC)
    model.train()
    model.training = True

    train_initT = time.time()

    dataGens = []
    outDisps = []
    for s in params.trainingSeries:
        dataGens.append(DataGeneretor(s, imageDir, prepreocF, attach=True))
        PM.printI(bcolors.LIGHTGREEN+f"sequence: {s}"+bcolors.ENDC)
        outDisps.append(PM.HTMLProgressBarI(0, dataGens[-1].numBatchImgs-1))

    while len(dataGens) > 0:
        pos_dg = random.randint(0, len(dataGens)-1)

        try:
            inputs, labels, pos, nb = dataGens[pos_dg].__next__()
            outputs = []
            outDisps[pos_dg].update(pos)

            for i in range(nb):
                torch.cuda.empty_cache()

                model.zero_grad()

                outputs = model(inputs[i])

                totLoss = criterion(outputs, labels[i])
                poseLoss = criterion(outputs[0:3], labels[i][0:3]).item()
                rotLoss = criterion(outputs[3:6], labels[i][3:6]).item()

                totLoss.backward()
                optimizer.step()

                loss_train[dataGens[pos_dg].sequence]["tot"].append(totLoss.item())
                loss_train[dataGens[pos_dg].sequence]["pose"].append(poseLoss)
                loss_train[dataGens[pos_dg].sequence]["rot"].append(rotLoss)

            del inputs, labels, totLoss, poseLoss, rotLoss
            gc.collect()
            torch.cuda.empty_cache()
        except StopIteration:
            PM.printI("Loss Sequence[{dataGens[pos_dg].sequence}]: [tot: %.5f, pose: %.5f, rot: %.5f]"%(
                np.mean(loss_train[dataGens[pos_dg].sequence]["tot"]),
                np.mean(loss_train[dataGens[pos_dg].sequence]["pose"]),
                np.mean(loss_train[dataGens[pos_dg].sequence]["rot"])
                ))

            dataGens.remove(dataGens[pos_dg])
            outDisps.remove(outDisps[pos_dg])

    train_elapsedT = time.time() - train_initT
    loss_train["tot"]["tot"].append(sum(
        [np.mean(loss_train[seq]["tot"]) for seq in params.trainingSeries]
        )/len(params.trainingSeries))
    loss_train["tot"]["pose"].append(sum(
        [np.mean(loss_train[seq]["pose"]) for seq in params.trainingSeries]
        )/len(params.trainingSeries))
    loss_train["tot"]["rot"].append(sum(
        [np.mean(loss_train[seq]["rot"]) for seq in params.trainingSeries]
        )/len(params.trainingSeries))
    PM.printI("Loss Train: [tot: %.5f, pose: %.5f, rot: %.5f] , time %.2fs"%(
        loss_train["tot"]["tot"][-1], loss_train["tot"]["pose"][-1], loss_train["tot"]["rot"][-1], train_elapsedT))

    return loss_train, train_elapsedT



def trainEpochRandom_RDG(imageDir, prepreocF):
    loss_train = {key: {"tot": [], "pose": [], "rot": []} for key in params.trainingSeries+["tot"]}

    PM.printI(bcolors.LIGHTYELLOW+"TRAINING"+bcolors.ENDC)
    model.train()
    model.training = True

    train_initT = time.time()
    rdg = RandomDataGeneretor(params.trainingSeries, imageDir, prepreocF, attach=True)
    train_numOfBatch = rdg.maxIters-1
    outputs = []

    for inputs, labels, pos, seq, nb in rdg:
        for i in range(nb):
            torch.cuda.empty_cache()

            model.zero_grad()

            outputs = model(inputs[i])

            totLoss = criterion(outputs, labels[i])
            poseLoss = criterion(outputs[0:3], labels[i][0:3]).item()
            rotLoss = criterion(outputs[3:6], labels[i][3:6]).item()

            totLoss.backward()
            optimizer.step()

            loss_train[seq[i]]["tot"].append(totLoss.item())
            loss_train[seq[i]]["pose"].append(poseLoss)
            loss_train[seq[i]]["rot"].append(rotLoss)
        del inputs, labels, totLoss, poseLoss, rotLoss
        gc.collect()
        torch.cuda.empty_cache()
        PM.printProgressBarI(pos, train_numOfBatch)
    del rdg
    gc.collect()
    torch.cuda.empty_cache()

    for s in params.trainingSeries:
        PM.printI(f"Loss Sequence[{bcolors.LIGHTGREEN}{s}{bcolors.ENDC}]: [tot: %.5f, pose: %.5f, rot: %.5f]"%(
            np.mean(loss_train[s]["tot"]),
            np.mean(loss_train[s]["pose"]),
            np.mean(loss_train[s]["rot"])
            ))

    train_elapsedT = time.time() - train_initT
    loss_train["tot"]["tot"].append(sum(
        [np.mean(loss_train[seq]["tot"]) for seq in params.trainingSeries]
        )/len(params.trainingSeries))
    loss_train["tot"]["pose"].append(sum(
        [np.mean(loss_train[seq]["pose"]) for seq in params.trainingSeries]
        )/len(params.trainingSeries))
    loss_train["tot"]["rot"].append(sum(
        [np.mean(loss_train[seq]["rot"]) for seq in params.trainingSeries]
        )/len(params.trainingSeries))
    PM.printI("Loss Train: [tot: %.5f, pose: %.5f, rot: %.5f] , time %.2fs"%(
        loss_train["tot"]["tot"][-1], loss_train["tot"]["pose"][-1], loss_train["tot"]["rot"][-1], train_elapsedT),
        head="\n")

    return loss_train, train_elapsedT



FLAG_LOAD = False #@param {type:"boolean"}
FLAG_SAVE_LOG = False #@param {type:"boolean"}
FLAG_SAVE = False #@param {type:"boolean"}

BASE_EPOCH = 1 #@param {type:"raw"} # 1 starting epoch
NUM_EPOCHS = 200 - BASE_EPOCH #@param {type:"raw"} # 10 how many epoch
PM.printD(f"[{BASE_EPOCH}-{BASE_EPOCH+NUM_EPOCHS-1}]\n")

fileNameFormat = "medium"

prefixFileNameLoad = "DeepVO_epoch_"
suffixFileNameLoad = "medium[1-2]" #@param {type:"string"}

prefixFileNameLosses = "loss_"
suffixFileNameLosses = "{}[{}]"

prefixFileNameSave = "DeepVO_epoch_"
suffixFileNameSave = "{}[{}-{}]"

imageDir = "image_2"
prepreocF = EnumPreproc.UNCHANGED((params.WIDTH, params.HEIGHT))
type_train = "online_random_RDG" # online, preprocessed, online_random, online_random_RDG


try:
  del model, criterion, optimizer
  gc.collect()
  torch.cuda.empty_cache()
except NameError:
  pass

model, criterion, optimizer = buildModel(typeModel=params.typeModel,
                                        typeCriterion=params.typeCriterion,
                                        typeOptimizer=params.typeOptimizer)

#Load the model
if FLAG_LOAD:
    fileName = os.path.join(params.dir_Model, f"{prefixFileNameLoad}{suffixFileNameLoad}.pt")
    checkpoint = torch.load(fileName)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    PM.printI(bcolors.LIGHTGREEN+"Loaded {}\n".format(fileName)+bcolors.ENDC)
else:
    PM.printI(bcolors.LIGHTRED+"Model not Loaded\n"+bcolors.ENDC)



for epoch in range(BASE_EPOCH, BASE_EPOCH+NUM_EPOCHS):
    #trainEpoch(imageDir, prepreocF)
    PM.printI(bcolors.LIGHTRED+"EPOCH {}/{}\n".format(epoch, BASE_EPOCH+NUM_EPOCHS-1)+bcolors.ENDC)


    if type_train == "online":
        loss_train, train_elapsedT = trainEpoch(imageDir, prepreocF)
    elif type_train == "preprocessed":
        loss_train, train_elapsedT = trainEpochPreprocessed()
    elif type_train == "online_random":
        loss_train, train_elapsedT = trainEpochRandom(imageDir, prepreocF)
    elif type_train == "online_random_RDG":
        loss_train, train_elapsedT = trainEpochRandom_RDG(imageDir, prepreocF)
    else:
        raise NotImplementedError


    if FLAG_SAVE_LOG:
        suffix = suffixFileNameLosses.format(fileNameFormat, epoch)
        fileName = os.path.join(params.dir_History, f"{prefixFileNameLosses}{suffix}.txt")
        with open(fileName, "w") as f:
            f.write("{}\n".format(str(loss_train)))
        PM.printD(bcolors.LIGHTGREEN+"saved on file {}.txt".format("{}/loss{}".format(params.dir_History, suffix))+bcolors.ENDC)
    else:
        PM.printD(bcolors.LIGHTRED+"History not Saved\n"+bcolors.ENDC)


    #Save the model
    if FLAG_SAVE:
        suffix = suffixFileNameSave.format(fileNameFormat, BASE_EPOCH, epoch)
        fileName = os.path.join(params.dir_Model, f"{prefixFileNameSave}{suffix}.pt")
        torch.save({
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  }, fileName)
        PM.printI(bcolors.LIGHTGREEN+"Saved {}\n".format(fileName)+bcolors.ENDC)

        if epoch != BASE_EPOCH:
            suffix = suffixFileNameSave.format(fileNameFormat, BASE_EPOCH, epoch-1)
            fileName = os.path.join(params.dir_Model, f"{prefixFileNameSave}{suffix}.pt")
            os.remove(fileName)
            PM.printI(bcolors.LIGHTRED+"Removed {}\n".format(fileName)+bcolors.ENDC)
    else:
        PM.printI(bcolors.LIGHTRED+"Model not Saved\n"+bcolors.ENDC)



    if type_train == "online" or type_train == "online_random" or type_train == "online_random_RDG":
        loss_test, test_elapsedT = testEpoch(imageDir, prepreocF)
    elif type_train == "preprocessed":
        loss_test, test_elapsedT = testEpochPreprocessed()
    else:
        raise NotImplementedError


    if FLAG_SAVE_LOG:
        suffix = suffixFileNameLosses.format(fileNameFormat, epoch)
        fileName = os.path.join(params.dir_History, f"{prefixFileNameLosses}{suffix}.txt")
        with open(fileName, "a") as f:
            f.write("{}\n".format(str(loss_test)))
        PM.printD(bcolors.LIGHTGREEN+"saved on file {}.txt".format("{}/loss{}".format(params.dir_History, suffix))+bcolors.ENDC)
    else:
        PM.printI(bcolors.LIGHTRED+"History not Saved\n"+bcolors.ENDC)


    PM.printI("epoch %d"%(epoch), head="\n")
    PM.printI("Loss Train: [tot: %.5f, pose: %.5f, rot: %.5f] , time %.2fs"%(
        loss_train["tot"]["tot"][-1], loss_train["tot"]["pose"][-1], loss_train["tot"]["rot"][-1], train_elapsedT))
    PM.printI("Loss Test: [tot: %.5f, pose: %.5f, rot: %.5f] , time %.2fs\n"%(
        loss_test["tot"]["tot"][-1], loss_test["tot"]["pose"][-1], loss_test["tot"]["rot"][-1], test_elapsedT))

    del loss_train, loss_test
    gc.collect()
    torch.cuda.empty_cache()


