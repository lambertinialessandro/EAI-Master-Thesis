
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from enum import Enum

import gc
import time
import random

import torch
import numpy as np
import matplotlib.pyplot as plt

from loadData import DataGeneretorPreprocessed, DataGeneretorOnline, RandomDataGeneretor

import params
from modules.utility import PM, bcolors


def trainEP(model, criterion, optimizer, imageDir, prepreocF, sequences=params.trainingSeries):
    loss_train = {key: {"tot": [], "pose": [], "rot": []} for key in sequences+["tot"]}

    PM.printI(bcolors.LIGHTYELLOW+"TRAINING"+bcolors.ENDC)
    model.train()
    model.training = True

    train_initT = time.time()
    for sequence in sequences:
        PM.printI(bcolors.LIGHTGREEN+"Sequence: {}".format(sequence)+bcolors.ENDC)
        dg = DataGeneretorPreprocessed(prepreocF, sequence, imageDir, attach=True)
        train_numOfBatch = dg.numBatchImgs - 1
        PB = PM.printProgressBarI(0, train_numOfBatch)
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
                    pts_yTrain = np.append(pts_yTrain,
                                           [pts_yTrain[-1] + det_labels[j]],
                                           axis=0)
                    pts_out = np.append(pts_out,
                                        [pts_out[-1] + det_outputs[j]],
                                        axis=0)
            del inputs, labels, outputs, det_outputs, det_labels, totLoss, poseLoss, rotLoss
            gc.collect()
            torch.cuda.empty_cache()
            PB.update(i)
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
        [np.mean(loss_train[seq]["tot"]) for seq in sequences]
        )/len(sequences))
    loss_train["tot"]["pose"].append(sum(
        [np.mean(loss_train[seq]["pose"]) for seq in sequences]
        )/len(sequences))
    loss_train["tot"]["rot"].append(sum(
        [np.mean(loss_train[seq]["rot"]) for seq in sequences]
        )/len(sequences))
    PM.printI("Loss Train: [tot: %.5f, pose: %.5f, rot: %.5f] , time %.2fs"%(
        loss_train["tot"]["tot"][-1], loss_train["tot"]["pose"][-1], loss_train["tot"]["rot"][-1], train_elapsedT))

    return loss_train, train_elapsedT

def trainEO(model, criterion, optimizer, imageDir, prepreocF, sequences=params.trainingSeries):
    loss_train = {key: {"tot": [], "pose": [], "rot": []} for key in sequences+["tot"]}

    PM.printI(bcolors.LIGHTYELLOW+"TRAINING"+bcolors.ENDC)
    model.train()
    model.training = True

    train_initT = time.time()
    for sequence in sequences:
        PM.printI(bcolors.LIGHTGREEN+"Sequence: {}".format(sequence)+bcolors.ENDC)
        dg = DataGeneretorOnline(prepreocF, sequence, imageDir, attach=True)
        train_numOfBatch = dg.numBatchImgs - 1
        PB = PM.printProgressBarI(0, train_numOfBatch)
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
            PB.update(i)
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
        [np.mean(loss_train[seq]["tot"]) for seq in sequences]
        )/len(sequences))
    loss_train["tot"]["pose"].append(sum(
        [np.mean(loss_train[seq]["pose"]) for seq in sequences]
        )/len(sequences))
    loss_train["tot"]["rot"].append(sum(
        [np.mean(loss_train[seq]["rot"]) for seq in sequences]
        )/len(sequences))
    PM.printI("Loss Train: [tot: %.5f, pose: %.5f, rot: %.5f] , time %.2fs"%(
        loss_train["tot"]["tot"][-1], loss_train["tot"]["pose"][-1], loss_train["tot"]["rot"][-1], train_elapsedT))

    return loss_train, train_elapsedT


def trainEPR(model, criterion, optimizer, imageDir, prepreocF, sequences=params.trainingSeries):
    loss_train = {key: {"tot": [], "pose": [], "rot": []} for key in sequences+["tot"]}

    PM.printI(bcolors.LIGHTYELLOW+"TRAINING"+bcolors.ENDC)
    model.train()
    model.training = True

    train_initT = time.time()

    dataGens = []
    outDisps = []
    for sequence in sequences:
        dataGens.append(DataGeneretorPreprocessed(prepreocF, sequence, imageDir, attach=True))
        PM.printI(bcolors.LIGHTGREEN+f"sequence: {sequence}"+bcolors.ENDC)
        outDisps.append(PM.printProgressBarI(0, dataGens[-1].numBatchImgs-1))

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
        [np.mean(loss_train[seq]["tot"]) for seq in sequences]
        )/len(sequences))
    loss_train["tot"]["pose"].append(sum(
        [np.mean(loss_train[seq]["pose"]) for seq in sequences]
        )/len(sequences))
    loss_train["tot"]["rot"].append(sum(
        [np.mean(loss_train[seq]["rot"]) for seq in sequences]
        )/len(sequences))
    PM.printI("Loss Train: [tot: %.5f, pose: %.5f, rot: %.5f] , time %.2fs"%(
        loss_train["tot"]["tot"][-1], loss_train["tot"]["pose"][-1], loss_train["tot"]["rot"][-1], train_elapsedT))

    return loss_train, train_elapsedT

def trainEOR(model, criterion, optimizer, imageDir, prepreocF, sequences=params.trainingSeries):
    loss_train = {key: {"tot": [], "pose": [], "rot": []} for key in sequences+["tot"]}

    PM.printI(bcolors.LIGHTYELLOW+"TRAINING"+bcolors.ENDC)
    model.train()
    model.training = True

    train_initT = time.time()

    dataGens = []
    outDisps = []
    for sequence in sequences:
        dataGens.append(DataGeneretorOnline(prepreocF, sequence, imageDir, attach=True))
        PM.printI(bcolors.LIGHTGREEN+f"sequence: {sequence}"+bcolors.ENDC)
        outDisps.append(PM.printProgressBarI(0, dataGens[-1].numBatchImgs-1))

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
        [np.mean(loss_train[seq]["tot"]) for seq in sequences]
        )/len(sequences))
    loss_train["tot"]["pose"].append(sum(
        [np.mean(loss_train[seq]["pose"]) for seq in sequences]
        )/len(sequences))
    loss_train["tot"]["rot"].append(sum(
        [np.mean(loss_train[seq]["rot"]) for seq in sequences]
        )/len(sequences))
    PM.printI("Loss Train: [tot: %.5f, pose: %.5f, rot: %.5f] , time %.2fs"%(
        loss_train["tot"]["tot"][-1], loss_train["tot"]["pose"][-1], loss_train["tot"]["rot"][-1], train_elapsedT))

    return loss_train, train_elapsedT


def trainEPR_RDG(model, criterion, optimizer, imageDir, prepreocF, sequences=params.trainingSeries):
    loss_train = {key: {"tot": [], "pose": [], "rot": []} for key in sequences+["tot"]}

    PM.printI(bcolors.LIGHTYELLOW+"TRAINING"+bcolors.ENDC)
    model.train()
    model.training = True

    train_initT = time.time()
    rdg = RandomDataGeneretor(sequences, imageDir,
                              RandomDataGeneretor.GeneratorType.PREPROCESS,
                              prepreocF, attach=True)
    train_numOfBatch = rdg.maxIters-1
    PB = PM.printProgressBarI(0, train_numOfBatch)
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
        PB.update(pos)
    del rdg
    gc.collect()
    torch.cuda.empty_cache()

    for s in sequences:
        PM.printI(f"Loss Sequence[{bcolors.LIGHTGREEN}{s}{bcolors.ENDC}]: [tot: %.5f, pose: %.5f, rot: %.5f]"%(
            np.mean(loss_train[s]["tot"]),
            np.mean(loss_train[s]["pose"]),
            np.mean(loss_train[s]["rot"])
            ))

    train_elapsedT = time.time() - train_initT
    loss_train["tot"]["tot"].append(sum(
            [np.mean(loss_train[seq]["tot"]) for seq in sequences]
        )/len(sequences))
    loss_train["tot"]["pose"].append(sum(
        [np.mean(loss_train[seq]["pose"]) for seq in sequences]
        )/len(sequences))
    loss_train["tot"]["rot"].append(sum(
        [np.mean(loss_train[seq]["rot"]) for seq in sequences]
        )/len(sequences))
    PM.printI("Loss Train: [tot: %.5f, pose: %.5f, rot: %.5f] , time %.2fs"%(
        loss_train["tot"]["tot"][-1], loss_train["tot"]["pose"][-1], loss_train["tot"]["rot"][-1], train_elapsedT))

    return loss_train, train_elapsedT

def trainEOR_RDG(model, criterion, optimizer, imageDir, prepreocF, sequences=params.trainingSeries):
    loss_train = {key: {"tot": [], "pose": [], "rot": []} for key in sequences+["tot"]}

    PM.printI(bcolors.LIGHTYELLOW+"TRAINING"+bcolors.ENDC)
    model.train()
    model.training = True

    train_initT = time.time()
    rdg = RandomDataGeneretor(sequences, imageDir,
                              RandomDataGeneretor.GeneratorType.ONLINE,
                              prepreocF, attach=True)
    train_numOfBatch = rdg.maxIters-1
    PB = PM.printProgressBarI(0, train_numOfBatch)
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
        PB.update(pos)
    del rdg
    gc.collect()
    torch.cuda.empty_cache()

    for s in sequences:
        PM.printI(f"Loss Sequence[{bcolors.LIGHTGREEN}{s}{bcolors.ENDC}]: [tot: %.5f, pose: %.5f, rot: %.5f]"%(
            np.mean(loss_train[s]["tot"]),
            np.mean(loss_train[s]["pose"]),
            np.mean(loss_train[s]["rot"])
            ))

    train_elapsedT = time.time() - train_initT
    loss_train["tot"]["tot"].append(sum(
        [np.mean(loss_train[seq]["tot"]) for seq in sequences]
        )/len(sequences))
    loss_train["tot"]["pose"].append(sum(
        [np.mean(loss_train[seq]["pose"]) for seq in sequences]
        )/len(sequences))
    loss_train["tot"]["rot"].append(sum(
        [np.mean(loss_train[seq]["rot"]) for seq in sequences]
        )/len(sequences))
    PM.printI("Loss Train: [tot: %.5f, pose: %.5f, rot: %.5f] , time %.2fs"%(
        loss_train["tot"]["tot"][-1], loss_train["tot"]["pose"][-1], loss_train["tot"]["rot"][-1], train_elapsedT),
        head="\n")

    return loss_train, train_elapsedT


def testEP(model, criterion, optimizer, imageDir, prepreocF, sequences=params.testingSeries,
           plot=True):
    loss_test = {key: {"tot": [], "pose": [], "rot": []} for key in sequences+["tot"]}

    PM.printI(bcolors.LIGHTYELLOW+"TESTING"+bcolors.ENDC)
    model.eval()
    model.training = False

    test_initT = time.time()
    for sequence in sequences:
        PM.printI(bcolors.LIGHTGREEN+"Sequence: {}".format(sequence)+bcolors.ENDC)
        dg = DataGeneretorPreprocessed(prepreocF, sequence, imageDir, attach=True)
        test_numOfBatch = dg.numBatchImgs - 1
        PB = PM.printProgressBarI(0, test_numOfBatch)
        outputs = []
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

                totLoss = criterion(outputs, labels[0]).item()
                poseLoss = criterion(outputs[0:3], labels[0][0:3]).item()
                rotLoss = criterion(outputs[3:6], labels[0][3:6]).item()

                loss_test[sequence]["tot"].append(totLoss)
                loss_test[sequence]["pose"].append(poseLoss)
                loss_test[sequence]["rot"].append(rotLoss)

                if plot:
                    for j in range(params.BACH_SIZE):
                        pts_yTest = np.append(pts_yTest,
                                               [pts_yTest[-1] + det_labels[j]],
                                               axis=0)
                        pts_out = np.append(pts_out,
                                            [pts_out[-1] + det_outputs[j]],
                                            axis=0)
            PB.update(pos)

            del inputs, labels, outputs, det_outputs, det_labels, totLoss, poseLoss, rotLoss
            gc.collect()
            torch.cuda.empty_cache()
        PM.printI("Loss Sequence: [tot: %.5f, pose: %.5f, rot: %.5f]"%(
            np.mean(loss_test[sequence]["tot"]),
            np.mean(loss_test[sequence]["pose"]),
            np.mean(loss_test[sequence]["rot"])
            ))
        del dg
        gc.collect()
        torch.cuda.empty_cache()

        #print(y_test_det.shape)
        #print(outputs.shape)

        if plot:
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
        [np.mean(loss_test[seq]["tot"]) for seq in sequences]
        )/len(sequences))
    loss_test["tot"]["pose"].append(sum(
        [np.mean(loss_test[seq]["pose"]) for seq in sequences]
        )/len(sequences))
    loss_test["tot"]["rot"].append(sum(
        [np.mean(loss_test[seq]["rot"]) for seq in sequences]
        )/len(sequences))
    PM.printI("Loss Test: [tot: %.5f, pose: %.5f, rot: %.5f] , time %.2fs"%(
        loss_test["tot"]["tot"][-1], loss_test["tot"]["pose"][-1], loss_test["tot"]["rot"][-1], test_elapsedT))

    return loss_test, test_elapsedT

def testEO(model, criterion, optimizer, imageDir, prepreocF, sequences=params.testingSeries,
           plot=True):
    loss_test = {key: {"tot": [], "pose": [], "rot": []} for key in sequences+["tot"]}

    PM.printI(bcolors.LIGHTYELLOW+"TESTING"+bcolors.ENDC)
    model.eval()
    model.training = False

    test_initT = time.time()
    for sequence in sequences:
        PM.printI(bcolors.LIGHTGREEN+"Sequence: {}".format(sequence)+bcolors.ENDC)
        dg = DataGeneretorOnline(prepreocF, sequence, imageDir, attach=True)
        test_numOfBatch = dg.numBatchImgs - 1
        PB = PM.printProgressBarI(0, test_numOfBatch)

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

                if plot:
                    for j in range(params.BACH_SIZE):
                        pts_yTest = np.append(pts_yTest, [pts_yTest[-1] + det_labels[j]], axis=0)

                        if j % params.STEP == 0:
                            pts_out = np.append(pts_out, [pts_out[-1] + det_outputs[j]], axis=0)
            del inputs, labels, det_outputs, det_labels, totLoss, poseLoss, rotLoss
            gc.collect()
            torch.cuda.empty_cache()
            PB.update(pos)
        del dg
        gc.collect()
        torch.cuda.empty_cache()

        PM.printI("Loss Sequence: [tot: %.5f, pose: %.5f, rot: %.5f]"%(
            np.mean(loss_test[sequence]["tot"]),
            np.mean(loss_test[sequence]["pose"]),
            np.mean(loss_test[sequence]["rot"])
            ))

        if plot:
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
        [np.mean(loss_test[seq]["tot"]) for seq in sequences]
        )/len(sequences))
    loss_test["tot"]["pose"].append(sum(
        [np.mean(loss_test[seq]["pose"]) for seq in sequences]
        )/len(sequences))
    loss_test["tot"]["rot"].append(sum(
        [np.mean(loss_test[seq]["rot"]) for seq in sequences]
        )/len(sequences))
    PM.printI("Loss Test: [tot: %.5f, pose: %.5f, rot: %.5f] , time %.2fs"%(
        loss_test["tot"]["tot"][-1], loss_test["tot"]["pose"][-1], loss_test["tot"]["rot"][-1], test_elapsedT),
        head="\n")

    return loss_test, test_elapsedT


class enumTrain(Enum):
    preprocessed = "preprocessed"
    online = "online"

    preprocessed_random = "preprocessed_random"
    online_random = "online_random"

    preprocessed_random_RDG = "preprocessed_random_RDG"
    online_random_RDG = "online_random_RDG"


def trainModel(model, criterion, optimizer, imageDir, prepreocF, type_train,\
               sequences_train=params.trainingSeries, sequences_test=params.testingSeries,
               plot_test=False):
    # Load the model
    if params.FLAG_LOAD:
        fileName = os.path.join(params.dir_Model,
                                f"{params.prefixFileNameLoad}{params.suffixFileNameLoad}.pt")
        checkpoint = torch.load(fileName)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        PM.printI(bcolors.LIGHTGREEN+"Loaded {}\n".format(fileName)+bcolors.ENDC)
    else:
        PM.printI(bcolors.LIGHTRED+"Model not Loaded\n"+bcolors.ENDC)



    loss_train_epochs = {"tot":[], "pose":[], "rot":[]}
    loss_test_epochs = {"tot":[], "pose":[], "rot":[]}
    for epoch in range(params.BASE_EPOCH, params.BASE_EPOCH + params.NUM_EPOCHS):
        PM.printI(bcolors.LIGHTRED+"EPOCH {}/{}\n".format(epoch, params.BASE_EPOCH+params.NUM_EPOCHS-1)+bcolors.ENDC)


        # Train the model
        if type_train == enumTrain.preprocessed:
            loss_train, train_elapsedT = trainEP(model, criterion, optimizer,
                                                    imageDir, prepreocF,
                                                    sequences=sequences_train)
        elif type_train == enumTrain.online:
            loss_train, train_elapsedT = trainEO(model, criterion, optimizer,
                                                    imageDir, prepreocF,
                                                    sequences=sequences_train)
        elif type_train == enumTrain.preprocessed_random:
            loss_train, train_elapsedT = trainEPR(model, criterion, optimizer,
                                                    imageDir, prepreocF,
                                                    sequences=sequences_train)
        elif type_train == enumTrain.online_random:
            loss_train, train_elapsedT = trainEOR(model, criterion, optimizer,
                                                    imageDir, prepreocF,
                                                    sequences=sequences_train)
        elif type_train == enumTrain.preprocessed_random_RDG:
            loss_train, train_elapsedT = trainEPR_RDG(model, criterion, optimizer,
                                                        imageDir, prepreocF,
                                                        sequences=sequences_train)
        elif type_train == enumTrain.online_random_RDG:
            loss_train, train_elapsedT = trainEOR_RDG(model, criterion, optimizer,
                                                        imageDir, prepreocF,
                                                        sequences=sequences_train)
        else:
            raise ValueError


        # Save the log file
        if params.FLAG_SAVE_LOG:
            suffix = params.suffixFileNameLosses.format(params.fileNameFormat, epoch)
            fileName = os.path.join(params.dir_History,
                                    f"{params.prefixFileNameLosses}{suffix}.txt")
            with open(fileName, "w") as f:
                f.write("{}\n".format(str(loss_train)))
            PM.printD(bcolors.LIGHTGREEN+"saved on file {}.txt".format("{}/loss_{}".format(params.dir_History, suffix))+bcolors.ENDC)
        else:
            PM.printD(bcolors.LIGHTRED+"History not Saved\n"+bcolors.ENDC)


        # Save the model
        if params.FLAG_SAVE:
            suffix = params.suffixFileNameSave.format(params.fileNameFormat, params.BASE_EPOCH, epoch)
            fileName = os.path.join(params.dir_Model,
                                    f"{params.prefixFileNameSave}{suffix}.pt")
            torch.save({
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      }, fileName)
            PM.printI(bcolors.LIGHTGREEN+"Saved {}\n".format(fileName)+bcolors.ENDC)

            if (epoch-params.BASE_EPOCH) % params.SAVE_STEP != 0:
                suffix = params.suffixFileNameSave.format(params.fileNameFormat, params.BASE_EPOCH, epoch-1)
                fileName = os.path.join(params.dir_Model,
                                        f"{params.prefixFileNameSave}{suffix}.pt")
                os.remove(fileName)
                PM.printI(bcolors.LIGHTRED+"Removed {}\n".format(fileName)+bcolors.ENDC)
        else:
            PM.printI(bcolors.LIGHTRED+"Model not Saved\n"+bcolors.ENDC)


        # Test the model
        if type_train == enumTrain.preprocessed or \
                type_train == enumTrain.preprocessed_random or \
                type_train == enumTrain.preprocessed_random_RDG:
            loss_test, test_elapsedT = testEP(model, criterion, optimizer,
                                                             imageDir, prepreocF,
                                                             sequences=sequences_test,
                                                             plot=plot_test)
        elif type_train == enumTrain.online or \
                type_train == enumTrain.online_random or \
                type_train == enumTrain.online_random_RDG:
            loss_test, test_elapsedT = testEO(model, criterion, optimizer,
                                                 imageDir, prepreocF,
                                                 sequences=sequences_test,
                                                 plot=plot_test)
        else:
            raise NotImplementedError


        # Save the log file
        if params.FLAG_SAVE_LOG:
            suffix = params.suffixFileNameLosses.format(params.fileNameFormat, epoch)
            fileName = os.path.join(params.dir_History,
                                    f"{params.prefixFileNameLosses}{suffix}.txt")
            with open(fileName, "a") as f:
                f.write("{}\n".format(str(loss_test)))
            PM.printD(bcolors.LIGHTGREEN+"saved on file {}.txt".format("{}/loss_{}".format(params.dir_History, suffix))+bcolors.ENDC)
        else:
            PM.printI(bcolors.LIGHTRED+"History not Saved\n"+bcolors.ENDC)


        # Epoch summary and plot
        PM.printI("epoch %d"%(epoch), head="\n")
        PM.printI("Loss Train: [tot: %.5f, pose: %.5f, rot: %.5f] , time %.2fs"%(
            loss_train["tot"]["tot"][-1], loss_train["tot"]["pose"][-1], loss_train["tot"]["rot"][-1], train_elapsedT))
        PM.printI("Loss Test: [tot: %.5f, pose: %.5f, rot: %.5f] , time %.2fs\n"%(
            loss_test["tot"]["tot"][-1], loss_test["tot"]["pose"][-1], loss_test["tot"]["rot"][-1], test_elapsedT))

        for k in loss_train_epochs.keys():
            loss_train_epochs[k].append(loss_train["tot"][k])
            loss_test_epochs[k].append(loss_test["tot"][k])

        del loss_train, loss_test
        gc.collect()
        torch.cuda.empty_cache()


        dimX = len(loss_train_epochs['tot'])
        x = np.linspace(1, dimX, dimX)
        plt.figure(figsize=(8, 6), dpi=80)
        plt.plot(x, loss_train_epochs['tot'], color='red')
        plt.plot(x, loss_train_epochs['pose'], color='blue')
        plt.plot(x, loss_train_epochs['rot'], color='green')
        plt.legend(['total loss', 'position loss', 'rotation loss'])
        plt.show()

        dimX = len(loss_test_epochs['tot'])
        x = np.linspace(1, dimX, dimX)
        plt.figure(figsize=(8, 6), dpi=80)
        plt.plot(x, loss_test_epochs['tot'], color='red')
        plt.plot(x, loss_test_epochs['pose'], color='blue')
        plt.plot(x, loss_test_epochs['rot'], color='green')
        plt.legend(['total loss', 'position loss', 'rotation loss'])
        plt.show()



if __name__ == "__main__":
    import math
    from modules.network.NetworkModule import NetworkFactory
    from modules.preprocess.PreprocessModule import PreprocessFactory
    import argparse

    parser = argparse.ArgumentParser(description='Training DeepVO')

    parser.add_argument('--type', default='4', type=int, help=' int [1-2]: SOBEL/UNCHANGED')
    parser.add_argument('--size', default='1', type=int, help=' int [1-3]: small/medium/big')
    parser.add_argument('--name', default='docker_test', type=str, help=' str: name network')
    parser.add_argument('--epochs', default=200, type=int, help=' int: number of epochs')
    parser.add_argument('--dim_LSTM', default=1000, type=int, help=' int: number of epochs')
    parser.add_argument('--online', default=1, type=int,
                        help=' int: online dataset preprocessing [0=False, 1=True]')
    args = parser.parse_args()

    type_net = args.type
    size_net = args.size
    name_net = args.name
    num_epochs = args.epochs
    dim_LSTM = args.dim_LSTM
    online = args.online

    params.BASE_EPOCH = 1
    params.NUM_EPOCHS = num_epochs
    imageDir = "image_2"

    if size_net == 1:
        params.WIDTH = 320
        params.HEIGHT = 96
    elif size_net == 2:
        params.WIDTH = 640
        params.HEIGHT = 192
    elif size_net == 3:
        params.WIDTH = 1280
        params.HEIGHT = 384
    else:
        raise ValueError

    if online == 1:
        type_train = enumTrain.online_random_RDG
        # online  online_random  online_random_RDG
    elif online == 0:
        type_train = enumTrain.preprocessed_random_RDG
        # preprocessed  preprocessed_random  preprocessed_random_RDG
    else:
        raise ValueError

    PM.printI(bcolors.LIGHTGREEN+"Info training:"+bcolors.ENDC)
    PM.printI(bcolors.LIGHTYELLOW+"name Network: {}".format(name_net)+bcolors.ENDC)
    PM.printI(bcolors.LIGHTYELLOW+"number epochs: {}".format(num_epochs)+bcolors.ENDC)
    PM.printI(bcolors.LIGHTYELLOW+"size images: {}x{}".format(params.WIDTH, params.HEIGHT)+bcolors.ENDC)
    PM.printI(bcolors.LIGHTYELLOW+"Dimension LSTM: {}".format(dim_LSTM)+bcolors.ENDC)
    PM.printI(bcolors.LIGHTYELLOW+"online preprocessing: {}".format(online)+bcolors.ENDC)
    PM.printI(bcolors.LIGHTYELLOW+"type preprocessing: {}".format(type_train.value)+bcolors.ENDC+"\n")


    if type_net == 1: # SOBEL
        typeModel = NetworkFactory.ModelEnum.SmallDeepVONet
        params.CHANNELS = 2
        params.suffixType = "SOBEL"
        params.DIM_LSTM = 384 * math.ceil(params.WIDTH/2**6) * math.ceil(params.HEIGHT/2**6)
        prepreocF = PreprocessFactory.build(PreprocessFactory.PreprocessEnum.SOBEL,
                                            (params.WIDTH, params.HEIGHT))
        params.DEVICE = torch.device("cuda")
    elif type_net == 2: # UNCHANGED
        typeModel = NetworkFactory.ModelEnum.DeepVONet
        params.CHANNELS = 6
        params.suffixType = "UNCHANGED"
        params.DIM_LSTM = 1024 * math.ceil(params.WIDTH/2**6) * math.ceil(params.HEIGHT/2**6)
        prepreocF = PreprocessFactory.build(PreprocessFactory.PreprocessEnum.UNCHANGED,
                                            (params.WIDTH, params.HEIGHT))
        params.DEVICE = torch.device("cpu") # TODO
    elif type_net == 3: # UNCHANGED_FSM
        typeModel = NetworkFactory.ModelEnum.DeepVONet_FSM
        params.CHANNELS = 6
        params.suffixType = "UNCHANGED"
        params.DIM_LSTM = 1024 * math.ceil(params.WIDTH/2**6) * math.ceil(params.HEIGHT/2**6)
        prepreocF = PreprocessFactory.build(PreprocessFactory.PreprocessEnum.UNCHANGED,
                                            (params.WIDTH, params.HEIGHT))
        params.DEVICE = torch.device("cuda")
    elif type_net == 4: # UNCHANGED
        typeModel = NetworkFactory.ModelEnum.DSC_VONet
        params.CHANNELS = 6
        params.suffixType = "UNCHANGED"
        params.DIM_LSTM = 1024 * math.ceil(params.WIDTH/2**6) * math.ceil(params.HEIGHT/2**6)
        prepreocF = PreprocessFactory.build(PreprocessFactory.PreprocessEnum.UNCHANGED,
                                            (params.WIDTH, params.HEIGHT))

    else:
        raise ValueError

    params.HIDDEN_SIZE_LSTM = dim_LSTM

    params.FLAG_LOAD = False
    params.fileNameFormat = name_net


    PM.printD(params.BASE_EPOCH)
    PM.printD(params.NUM_EPOCHS)
    PM.printD(params.suffixType)
    PM.printD(params.DIM_LSTM)

    try:
        del model, criterion, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    except NameError:
        pass

    typeCriterion = NetworkFactory.CriterionEnum.MSELoss
    typeOptimizer = NetworkFactory.OptimizerEnum.Adam

    model, criterion, optimizer = \
        NetworkFactory.build(typeModel, params.DIM_LSTM, params.HIDDEN_SIZE_LSTM, params.DEVICE,
                             typeCriterion,
                             typeOptimizer)

    # for parameter in model.parameters():
    #     PM.printI(str(parameter.size()))

    trainModel(model, criterion, optimizer, imageDir, prepreocF, type_train, plot_test=False)

    print("Train Terminated !!")


