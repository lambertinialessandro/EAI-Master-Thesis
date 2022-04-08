# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:30:41 2022

@author: lambe
"""

import matplotlib.pyplot as plt
from IPython.display import HTML, display

from params import FLAG_DEBUG_PRINT, FLAG_INFO_PRINT

class PrintManager:
    def __init__(self, FLAG_DEBUG_PRINT=False, FLAG_INFO_PRINT=False):
        self.FLAG_DEBUG_PRINT = FLAG_DEBUG_PRINT
        self.FLAG_INFO_PRINT = FLAG_INFO_PRINT

    # Debug
    def printD(self, msg, head=""):
      if self.FLAG_DEBUG_PRINT:
        print(head + "### Debug: " + msg)

    def imshowD(self, img, title=""):
      if self.FLAG_DEBUG_PRINT:
        plt.title("### Debug " + title)
        plt.imshow(img)
        plt.show()

    # Print iterations progress
    def printProgressBarI (self, iteration, total, length = 40):
        if self.FLAG_INFO_PRINT:
            percent = ("{0:." + str(2) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = '█' * filledLength + '-' * (length - filledLength)
            print(f'\r### Info Progress:|{bar}| {percent}% Complete', end='\r')
            # Print New Line on Complete
            if iteration == total:
                print()

    def HTMLProgressBarI(self, value, max=100):
        if self.FLAG_INFO_PRINT:
            return HTML("""
              <progress value='{value}' max='{max}', style='width: 50%' >
                  {value}
              </progress>
            """.format(value=value, max=max))

    # Info
    def printI(self, msg, head=""):
      if self.FLAG_INFO_PRINT:
        print(head + "### Info: " + msg)

    def imshowI(self, img, title=""):
      if self.FLAG_INFO_PRINT:
        plt.title("### Info " + title)
        plt.imshow(img)
        plt.show()

class bcolors:
    LIGHTRED = '\x1b[1;31;10m'
    LIGHTGREEN = '\x1b[1;32;10m'
    LIGHTYELLOW = '\x1b[1;33;10m'

    DARKBLUE = '\x1b[1;30;44m'
    DARKYELLOW = '\x1b[1;33;40m'
    DARKGREEN = '\x1b[1;32;40m'

    WARNING = '\x1b[0;30;41m'
    FAIL = '\x1b[0;30;43m'

    ENDC = '\x1b[0m'

PM = PrintManager(FLAG_DEBUG_PRINT, FLAG_INFO_PRINT)


# Testing
def main():
    import time

    FLAG_DEBUG_PRINT = True
    FLAG_INFO_PRINT = True

    pm = PrintManager(FLAG_DEBUG_PRINT, FLAG_INFO_PRINT)

    pm.printD(bcolors.LIGHTGREEN+"ciao"+bcolors.ENDC)
    pm.printI(bcolors.DARKGREEN+"ciao"+bcolors.ENDC)

    for i in range(5):
        pm.printProgressBarI(i, 5)
        time.sleep(0.3)
    pm.printProgressBarI(5, 5)

    # working on colab
    progress = display(pm.HTMLProgressBarI(0, 5-1), display_id=True)
    for i in range(5):
        progress.update(pm.HTMLProgressBarI(i, 5-1))
        time.sleep(0.2)

if __name__ == "__main__":
    main()