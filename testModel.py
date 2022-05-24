
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gc
import time

import torch
import numpy as np
import matplotlib.pyplot as plt

from loadData import DataGeneretor
from buildModel import buildModel

import params
from EnumPreproc import EnumPreproc
from utility import PM, bcolors

def testEpoch(imageDir, prepreocF, seq):
    loss_test = {key: {"tot": [], "pose": [], "rot": []} for key in seq+["tot"]}

    PM.printI(bcolors.LIGHTYELLOW+"TESTING"+bcolors.ENDC)
    model.eval()
    model.training = False

    test_initT = time.time()
    for sequence in seq:
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
        [np.mean(loss_test[seq]["tot"]) for seq in seq]
        )/len(seq))
    loss_test["tot"]["pose"].append(sum(
        [np.mean(loss_test[seq]["pose"]) for seq in seq]
        )/len(seq))
    loss_test["tot"]["rot"].append(sum(
        [np.mean(loss_test[seq]["rot"]) for seq in seq]
        )/len(seq))
    PM.printI("Loss Test: [tot: %.5f, pose: %.5f, rot: %.5f] , time %.2fs"%(
        loss_test["tot"]["tot"][-1], loss_test["tot"]["pose"][-1], loss_test["tot"]["rot"][-1], test_elapsedT),
        head="\n")

    return loss_test, test_elapsedT



fileNameFormat = "medium"

prefixFileNameLoad = "DeepVO_epoch_"
suffixFileNameLoad = "medium[1-105]" #@param {type:"string"}

imageDir = "image_2"
prepreocF = EnumPreproc.UNCHANGED((params.WIDTH, params.HEIGHT))


try:
  del model, criterion, optimizer
  gc.collect()
  torch.cuda.empty_cache()
except NameError:
  pass

model, criterion, optimizer = buildModel(typeModel=params.typeModel,
                                        typeCriterion=params.typeCriterion,
                                        typeOptimizer=params.typeOptimizer)

# Load the model
fileName = os.path.join(params.dir_Model, f"{prefixFileNameLoad}{suffixFileNameLoad}.pt")
checkpoint = torch.load(fileName, map_location=torch.device('cpu'))

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

PM.printI(bcolors.LIGHTGREEN+"Loaded {}\n".format(fileName)+bcolors.ENDC)

"""
 00: 4541  01: 1101  02: 4661  03: 801  04: 271  05: 2761  06: 1101
 07: 1101  08: 4071  09: 1591  10: 1201  11: 921

 trainingSeries = ["00", "01", "02", "08", "09"]
 testingSeries = ["03", "04", "05", "06", "07", "10"]
"""

seqs = [['04'], ['01']]

for seq in seqs:
    loss_test, test_elapsedT = testEpoch(imageDir, prepreocF, seq)









