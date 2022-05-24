
import os
import re
import numpy as np
import matplotlib.pyplot as plt

import params
from utility import PM, bcolors


lossNames = {}
check_re = re.compile("^loss\w+\[\d+\]\.txt$") # \[\d+\-\d+\]
prefix_re = re.compile("^loss\w+")
interval_re = re.compile("(?<=\[)\d+")

for s in os.listdir(params.dir_History):
    matched = check_re.match(s)
    is_match = bool(matched)

    prefix = prefix_re.findall(s)[0]
    if len(interval_re.findall(s)) <= 0:
      break
    interval = interval_re.findall(s)[0]

    if not prefix in lossNames.keys():
        lossNames[prefix] = []
    lossNames[prefix].append(interval)




# for k in lossNames.keys():
#     lossNames[k] = sorted(lossNames[k], key=lambda x: int(x[0]))
# print(lossNames)

for k in lossNames.keys():
    lossNames[k] = sorted(lossNames[k], key=lambda x: int(x))

PM.printD(f"{lossNames}")



totLosses = {}
for name in lossNames.keys():
    PM.printI("Current file: "+bcolors.LIGHTGREEN+f"{name}"+bcolors.ENDC, head="\n")
    totLosses[name] = {'train': {}, 'test': {}}

    for i in lossNames[name]:
        PM.printD(bcolors.LIGHTYELLOW+f"{i}"+bcolors.ENDC)
        currPos = int(i)
        with open(os.path.join(params.dir_History, f"{name}[{i}].txt"), "r") as f:
            totLosses[name]['train'][currPos] = eval(f.readline())['tot']
            totLosses[name]['test'][currPos] = eval(f.readline())['tot']


    # Train
    y_tot = [totLosses[name]['train'][k]['tot'] for k in totLosses[name]['train'].keys()]
    y_pos = [totLosses[name]['train'][k]['pose'] for k in totLosses[name]['train'].keys()]
    y_rot = [totLosses[name]['train'][k]['rot'] for k in totLosses[name]['train'].keys()]

    dimX = len(totLosses[name]['train'])
    x = np.linspace(1, dimX, dimX)

    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(x, y_tot, color='red')
    plt.plot(x, y_pos, color='blue')
    plt.plot(x, y_rot, color='green')
    plt.legend(['total loss', 'position loss', 'rotation loss'])
    plt.show()

    # Test
    y_tot = [totLosses[name]['test'][k]['tot'] for k in totLosses[name]['test'].keys()]
    y_pos = [totLosses[name]['test'][k]['pose'] for k in totLosses[name]['test'].keys()]
    y_rot = [totLosses[name]['test'][k]['rot'] for k in totLosses[name]['test'].keys()]

    dimX = len(y_tot)
    x = np.linspace(1, dimX, dimX)

    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(x, y_tot, color='red')
    plt.plot(x, y_pos, color='blue')
    plt.plot(x, y_rot, color='green')
    plt.legend(['total loss', 'position loss', 'rotation loss'])
    plt.show()













