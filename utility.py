# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:30:41 2022

@author: lambe
"""

import matplotlib.pyplot as plt
from IPython.display import HTML

class PrintManager:
    def __init__(self, FLAG_DEBUG_PRINT=False, FLAG_INFO_PRINT=False):
        self.FLAG_DEBUG_PRINT = FLAG_DEBUG_PRINT
        self.FLAG_INFO_PRINT = FLAG_INFO_PRINT

    # Debug
    def printD(self, msg):
      if self.FLAG_DEBUG_PRINT:
        print("### Debug: " + msg)

    def imshowD(self, img, title=""):
      if self.FLAG_DEBUG_PRINT:
        plt.title("### Debug "+title)
        plt.imshow(img)

    # Print iterations progress
    def printProgressBar (self, iteration, total, length = 50):
        percent = ("{0:." + str(2) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = '█' * filledLength + '-' * (length - filledLength)
        print(f'\r Progress:|{bar}| {percent}% Complete', end='\r')
        # Print New Line on Complete
        if iteration == total:
            print()

    # Info
    def printI(self, msg):
      if self.FLAG_INFO_PRINT:
        print("### Info: " + msg)

    def imshowI(self, img, title=""):
      if self.FLAG_INFO_PRINT:
        plt.title("### Info "+title)
        plt.imshow(img)

# Testing
def main():
    FLAG_DEBUG_PRINT = True
    FLAG_INFO_PRINT = True

    mm = PrintManager(FLAG_DEBUG_PRINT, FLAG_INFO_PRINT)

    mm.printD("ciao")
    mm.printI("ciao")

if __name__ == "__main__":
    main()