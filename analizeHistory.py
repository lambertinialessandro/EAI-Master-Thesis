
import os
import re
import numpy as np
import matplotlib.pyplot as plt

from modules.utility import PM, bcolors


def readHistoryFromDir(dir_History):
    check_re_single = re.compile("^loss\w+\[\d+\]\.txt$") # TODO
    interval_re_single = re.compile("(?<=\[)\d+")

    check_re_double = re.compile("^loss\w+\[\d+\-\d+\]\.txt$")
    interval_re_double = re.compile("(?<=\[)\d+\-\d+")

    prefix_re = re.compile("^loss\w+\[\d+\]") # TODO
    lossNames = {}

    for s in os.listdir(dir_History):
        for ss in os.listdir(os.path.join(dir_History, s)):
            if bool(check_re_single.match(ss)):
                interval = interval_re_single.findall(ss)[-1]
            elif bool(check_re_double.match(ss)):
                interval = interval_re_double.findall(ss)[-1].split('-')
            else:
                continue

            prefix = s
            if not prefix in lossNames.keys():
                lossNames[prefix] = []
            lossNames[prefix].append(interval)

    for k in lossNames.keys():
        if isinstance(lossNames[k][0], str):
            lossNames[k] = sorted(lossNames[k], key=lambda x: int(x))
        elif isinstance(lossNames[k][0], list):
            lossNames[k] = sorted(lossNames[k], key=lambda x: int(x[0]))

    return lossNames


def plot_graph(dir_History, name, lossName):
    totLosses = {'train': {}, 'test': {}}

    pb = PM.printProgressBarI(0, len(lossName))
    if isinstance(lossName[0], str):
        for i in lossName:
            currPos = int(i)
            pb.update(currPos)
            with open(os.path.join(dir_History, name, f"loss_{name}[{i}].txt"), "r") as f:
                state = 0
                for line in f:
                    try:
                        app_d = eval(line)
                    except NameError:
                        continue
                    if state == 0:
                        state = 1
                        totLosses['train'][currPos] = app_d['tot']
                    else:
                        state = 0
                        totLosses['test'][currPos] = app_d['tot']
                        currPos = currPos + 1
    elif isinstance(lossName[0], list):
        for minI, maxI in lossName:
            currPos = int(minI)
            with open(os.path.join(dir_History, name, f"loss_{name}[{minI}-{maxI}].txt"), "r") as f:
                state = 0
                for line in f:
                    app_d = eval(line)
                    if state == 0:
                        state = 1
                        totLosses['train'][currPos] = app_d['tot']
                    else:
                        state = 0
                        totLosses['test'][currPos] = app_d['tot']
                        currPos = currPos + 1

    dimX1 = len(totLosses['train'])
    x1 = np.linspace(1, dimX1, dimX1)

    y_tot1 = [totLosses['train'][k]['tot'] for k in totLosses['train'].keys()]
    y_pos1 = [totLosses['train'][k]['pose'] for k in totLosses['train'].keys()]
    y_rot1 = [totLosses['train'][k]['rot'] for k in totLosses['train'].keys()]

    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(x1, y_tot1, color='red')
    plt.plot(x1, y_pos1, color='blue')
    plt.plot(x1, y_rot1, color='green')
    plt.legend(['total loss', 'position loss', 'rotation loss'])
    plt.show()

    y_tot2 = [totLosses['test'][k]['tot'] for k in totLosses['test'].keys()]
    y_pos2 = [totLosses['test'][k]['pose'] for k in totLosses['test'].keys()]
    y_rot2 = [totLosses['test'][k]['rot'] for k in totLosses['test'].keys()]

    dimX2 = len(y_tot2)
    x2 = np.linspace(1, dimX2, dimX2)

    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(x2, y_tot2, color='red')
    plt.plot(x2, y_pos2, color='blue')
    plt.plot(x2, y_rot2, color='green')
    plt.legend(['total loss', 'position loss', 'rotation loss'])
    plt.show()

    dimX = min(dimX1, dimX2)
    x = np.linspace(1, dimX, dimX)

    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(x, y_tot1, color='red')
    plt.plot(x, y_tot2, color='blue')
    plt.legend(['total loss', 'position loss', 'rotation loss'])
    plt.show()


def analizeHistory(dir_History):
    lossNames = readHistoryFromDir(dir_History)

    while True:
        PM.printI(bcolors.LIGHTGREEN+"Select:"+bcolors.ENDC)
        elems = list(lossNames.keys())
        for i in range(len(elems)):
            PM.printI(f"\t{i}: "+bcolors.LIGHTYELLOW+f"{elems[i]}"+bcolors.ENDC)
        PM.printI("\t-1: "+bcolors.LIGHTYELLOW+"Exit"+bcolors.ENDC)

        selc = int(input("Input history: "))

        if selc == -1:
            break

        plot_graph(dir_History, elems[selc], lossNames[elems[selc]])
        input("PAUSE")



if __name__ == "__main__":
    import params
    PM.setFlags(True, True, False)

    dir_History = params.dir_History

    analizeHistory(dir_History)


