# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:36:00 2022

@author: lambe
"""

# from keras_video import SlidingFrameGenerator

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gc
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch

from loadData import DataLoader
import buildModel

import params
from utility import PrintManager, bcolors

pm = PrintManager(params.FLAG_DEBUG_PRINT, params.FLAG_INFO_PRINT)

model, criterion, optimizer = buildModel.main()

#fileName = params.dir_Model+"DeepVO_epoch3.pt"
#model.load_state_dict(torch.load(fileName))
#print(print("\x1b[1;31;10mLoaded {}\x1b[0m\n".format(fileName)))

BASE_EPOCH = 4
NUM_EPOCHS = 2
img_out = 0

loss_train = []
loss_test = []

for epoch in range(BASE_EPOCH, BASE_EPOCH+NUM_EPOCHS):
  print(bcolors.LIGHTRED+"EPOCH {}/{}\n".format(epoch, BASE_EPOCH+NUM_EPOCHS-1)+bcolors.ENDC)
  print(bcolors.LIGHTYELLOW+"TRAINING"+bcolors.ENDC)
  model.train()
  model.training = True
  app_loss_train = []
  
  train_initT = time.time()
  for sequence in params.trainingSeries:
    print(bcolors.LIGHTGREEN+"Sequence: {}".format(sequence)+bcolors.ENDC)
    X, y = DataLoader(params.dir_Dataset, sequence=sequence)
    train_numOfBatch = len(X)
    
    for i in range(train_numOfBatch):
      pm.printProgressBarI(i, train_numOfBatch)
      inputs = X[i]
      labels = y[i]

      model.zero_grad()
      model.reset_hidden_states(sizeHidden=params.BACH_SIZE, zero=True)
        
      outputs = model(inputs)

      loss = criterion(outputs, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      app_loss_train.append(loss.item())
    del X, y, inputs, labels
    gc.collect()
    
    pm.printProgressBarI(train_numOfBatch, train_numOfBatch)
    print("current loss: {}\n\n".format(sum(app_loss_train)/len(app_loss_train)))
  train_elapsedT = time.time() - train_initT
  
  
  
  print(bcolors.LIGHTYELLOW+"TESTING"+bcolors.ENDC)
  model.eval()
  model.training = False
  X, y = DataLoader(params.dir_Dataset, sequence=random.choice(params.testingSeries))
  test_numOfBatch = np.min([5, len(X)])
  app_loss_test = []

  test_initT = time.time()
  outputs = []
  for i in range(test_numOfBatch):
    pm.printProgressBarI(i, test_numOfBatch)
    inputs = X[i]
    labels = y[i]
    
    app_outputs = model(inputs)

    loss = criterion(app_outputs, labels)
    app_loss_test.append(loss.item())
    outputs.append(app_outputs)
  del X, inputs, labels
  gc.collect()
  
  pm.printProgressBarI(test_numOfBatch, test_numOfBatch)
  test_elapsedT = time.time() - test_initT
  
  y_test_det = []
  for i in range(len(y)):
    y_test_det.append(y[i].detach().numpy())
  y_test_det = np.array(y_test_det)
  del y
  gc.collect()

  outputs_det = []
  for i in range(len(outputs)):
    outputs_det.append(outputs[i].detach().numpy())
  outputs_det = np.array(outputs_det)
  del outputs
  gc.collect()

  print(y_test_det.shape)
  print(outputs_det.shape)
  
  pts_yTest = np.array([[0, 0, 0, 0, 0, 0]])
  pts_out = np.array([[0, 0, 0, 0, 0, 0]])
  for i in range(0, len(outputs_det)):
    for j in range(0, params.BACH_SIZE):
      pos = i*params.BACH_SIZE+j
      pts_yTest = np.append(pts_yTest, [pts_yTest[pos] + y_test_det[i, j]], axis=0)
      pts_out = np.append(pts_out, [pts_out[pos] + outputs_det[i, j]], axis=0)

  del y_test_det, outputs_det
  gc.collect()
  print(pts_yTest.shape)
  print(pts_out.shape)
  
  plt.plot(pts_out[:, 0], pts_out[:, 2], color='red')
  plt.plot(pts_yTest[:, 0], pts_yTest[:, 2], color='blue')
  plt.show()
  
  loss_train.append(sum(app_loss_train)/len(app_loss_train))
  loss_test.append(sum(app_loss_test)/len(app_loss_test))

  print("\nepoch %d"%(epoch))
  print("loss_train %.3f, time %.2fs"%(loss_train[-1], train_elapsedT))
  print("loss_test %.3f, time %.2fs"%(loss_test[-1], test_elapsedT))
  
  
  
  #Save the model
  fileName = params.dir_Model+"DeepVO_epoch{}.pt".format(epoch)
  torch.save(model.state_dict(), fileName)
  print(print("\x1b[1;31;10mSaved {}\x1b[0m\n".format(fileName)))








