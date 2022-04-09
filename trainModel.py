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

from loadData import DataGeneretor
from buildModel import buildModel

import params
from EnumPreproc import EnumPreproc
from utility import PM, bcolors

model, criterion, optimizer = buildModel(typeModel=params.typeModel,
                                               typeCriterion=params.typeCriterion,
                                               typeOptimizer=params.typeOptimizer)

imageDir = "image_2"
prepreocF = EnumPreproc.UNCHANGED((params.WIDTH, params.HEIGHT))

suffixFileNameLoad = "[1-10]"
suffixFileNameLosses = "[1-10]"
suffixFileNameSave = "[1-10]"

#fileName = "{}DeepVO_epoch{}.pt".format(params.dir_Model, suffixFileNameLoad)
#model.load_state_dict(torch.load(fileName))
#print(print("\x1b[1;31;10mLoaded {}\x1b[0m\n".format(fileName)))

BASE_EPOCH = 1 # 1 starting epoch
NUM_EPOCHS = 10 # 10 how many epoch

for epoch in range(BASE_EPOCH, BASE_EPOCH+NUM_EPOCHS):
  loss_train = {key: {"tot": [], "pose": [], "rot": []} for key in params.trainingSeries+["tot"]}
  loss_test = {key: {"tot": [], "pose": [], "rot": []} for key in params.testingSeries+["tot"]}

  print(bcolors.LIGHTRED+"EPOCH {}/{}\n".format(epoch, BASE_EPOCH+NUM_EPOCHS-1)+bcolors.ENDC)
  print(bcolors.LIGHTYELLOW+"TRAINING"+bcolors.ENDC)
  model.train()
  model.training = True

  train_initT = time.time()
  for sequence in params.trainingSeries:
    print(bcolors.LIGHTGREEN+"Sequence: {}".format(sequence)+bcolors.ENDC)
    # X, y = DataLoader(params.dir_Dataset, attach=False, suffixType=params.suffixType, sequence=sequence)

    dg = DataGeneretor(sequence, imageDir, prepreocF, attach=True)
    train_numOfBatch = dg.numImgs

    for inputs, labels, pos in dg:
        PM.printProgressBarI(pos, train_numOfBatch)
        for i in range(params.NUM_BACH):
          torch.cuda.empty_cache()

          model.zero_grad()
          # model.reset_hidden_states(sizeHidden=params.BACH_SIZE, zero=True)

          outputs = model(inputs[i])

          totLoss = criterion(outputs, labels[i])
          poseLoss = criterion(outputs[0:3], labels[i][0:3]).item()
          rotLoss = criterion(outputs[3:6], labels[i][3:6]).item()

          totLoss.backward()
          optimizer.step()

          loss_train[sequence]["tot"].append(totLoss.item())
          loss_train[sequence]["pose"].append(poseLoss)
          loss_train[sequence]["rot"].append(rotLoss)
        del inputs, labels, outputs, totLoss, poseLoss, rotLoss
        gc.collect()
        torch.cuda.empty_cache()
    del dg
    gc.collect()
    torch.cuda.empty_cache()

    PM.printI("Loss Sequence: [tot: %.5f, pose: %.5f, rot: %.5f]"%(
        np.mean(loss_train[sequence]["tot"][-1]),
        np.mean(loss_train[sequence]["pose"][-1]),
        np.mean(loss_train[sequence]["rot"][-1])
        ))
  train_elapsedT = time.time() - train_initT

  loss_train["tot"]["tot"].append(sum([np.mean(loss_train[seq]["tot"]) for seq in trainingSeries])/len(trainingSeries))
  loss_train["tot"]["pose"].append(sum([np.mean(loss_train[seq]["pose"]) for seq in trainingSeries])/len(trainingSeries))
  loss_train["tot"]["rot"].append(sum([np.mean(loss_train[seq]["rot"]) for seq in trainingSeries])/len(trainingSeries))
  with open("loss{}".format(suffixFileNameLosses), "a") as f:
      f.write("{}\n".format(str(loss_train)))
  
  PM.printD("saved on file {}.txt".format("{}loss{}".format(dir_History, suffixFileNameLosses)))
  for sequence in trainingSeries:
      del loss_train[sequence]
  gc.collect()
  torch.cuda.empty_cache()


  # TODO
  print(bcolors.LIGHTYELLOW+"TESTING"+bcolors.ENDC)
  model.eval()
  model.training = False
  sequence = random.choice(params.testingSeries)
  print(bcolors.LIGHTGREEN+"sequence: {}".format(sequence)+bcolors.ENDC)
  X, y = DataLoader(params.dir_Dataset, suffixType=params.suffixType,
                    sequence=sequence)
  test_numOfBatch = len(X)# np.min([40, len(X)])
  app_loss_test = []

  test_initT = time.time()
  outputs = []
  for i in range(test_numOfBatch):
    PM.printProgressBarI(i, test_numOfBatch)
    inputs = X[i]
    labels = y[i]

    app_outputs = model(inputs)

    totLoss = criterion(app_outputs, labels[0]).item()
    poseLoss = criterion(app_outputs, labels[0]).item()
    rotLoss = criterion(app_outputs, labels[0]).item()

    loss_test[sequence]["tot"].append(totLoss)
    loss_test[sequence]["pose"].append(poseLoss)
    loss_test[sequence]["rot"].append(rotLoss)

    outputs.append(app_outputs.detach().numpy())
  del X, inputs, labels
  gc.collect()
  torch.cuda.empty_cache()

  PM.printProgressBarI(test_numOfBatch, test_numOfBatch)
  test_elapsedT = time.time() - test_initT

  loss_test["tot"]["tot"] = sum([np.mean(loss_test[seq]["tot"]) for seq in params.testingSeries])/len(params.testingSeries)
  loss_test["tot"]["pose"] = sum([np.mean(loss_test[seq]["pose"]) for seq in params.testingSeries])/len(params.testingSeries)
  loss_test["tot"]["rot"] = sum([np.mean(loss_test[seq]["rot"]) for seq in params.testingSeries])/len(params.testingSeries)
  with open("loss{}".format(suffixFileNameLosses), "a") as f:
      f.write("{}\n".format(str(loss_train)))

  y_test_det = []
  for i in range(len(y)):
    y_test_det.append(y[i].detach().numpy())
  y_test_det = np.array(y_test_det)
  del y
  gc.collect()
  torch.cuda.empty_cache()
  outputs = np.array(outputs)

  print(y_test_det.shape)
  print(outputs.shape)

  pts_yTest = np.array([[0, 0, 0, 0, 0, 0]])
  pts_out = np.array([[0, 0, 0, 0, 0, 0]])
  for i in range(0, len(outputs)):
    for j in range(0, params.BACH_SIZE):
      pos = i*params.BACH_SIZE+j
      pts_yTest = np.append(pts_yTest, [pts_yTest[pos] + y_test_det[i, j]], axis=0)
      pts_out = np.append(pts_out, [pts_out[pos] + outputs[i, j]], axis=0)

  del outputs, y_test_det
  gc.collect()
  torch.cuda.empty_cache()
  print(pts_yTest.shape)
  print(pts_out.shape)

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

  loss_train.append(sum(app_loss_train)/len(app_loss_train))
  loss_test.append(sum(app_loss_test)/len(app_loss_test))

  print("\nepoch %d"%(epoch))
  print("loss_train %.5f, time %.2fs"%(loss_train[-1], train_elapsedT))
  print("loss_test %.5f, time %.2fs"%(loss_test[-1], test_elapsedT))



  #Save the model
  #fileName = "{}DeepVO_epoch{}.pt".format(params.dir_Model, suffixFileNameSave)
  #torch.save(model.state_dict(), fileName)
  #print(print("\x1b[1;31;10mSaved {}\x1b[0m\n".format(fileName)))


