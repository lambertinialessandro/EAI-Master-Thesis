
import matplotlib.pyplot as plt
from IPython.display import HTML, display

from params import FLAG_DEBUG_PRINT, FLAG_INFO_PRINT, FLAG_OUT_HTML


class STDOUT_holder:
    def __init__(self, value, max_v, *args, **kargs):
        super().__init__(*args, **kargs)
        self.max_v = max_v
        self.length = 40

        percent = ("{0:." + str(2) + "f}").format(100 * (value / float(self.max_v)))
        filledLength = int(self.length * value // self.max_v)
        bar = '█' * filledLength + '-' * (self.length - filledLength)
        print(f'\r### Info Progress:|{bar}| {percent}% Complete', end='\r')
        # Print New Line on Complete
        if value == self.max_v:
            print()

    def update(self, value):
        percent = ("{0:." + str(2) + "f}").format(100 * (value / float(self.max_v)))
        filledLength = int(self.length * value // self.max_v)
        bar = '█' * filledLength + '-' * (self.length - filledLength)
        print(f'\r### Info Progress:|{bar}| {percent}% Complete', end='\r')
        # Print New Line on Complete
        if value == self.max_v:
            print()

class HTML_holder:
    def __init__(self, value, max_v, *args, **kargs):
        super().__init__(*args, **kargs)
        self.max_v = max_v
        self.progress = display(
            HTML(f"<progress value='{value}' max='{self.max_v}', style='width: 50%' >\
                  {value}</progress>"), display_id=True)

    def update(self, value):
        self.progress.update(
            HTML(f"<progress value='{value}' max='{self.max_v}', style='width: 50%' >\
                 {value}</progress>"))

class PrintManager:
    def __init__(self, FLAG_DEBUG_PRINT=False, FLAG_INFO_PRINT=False, FLAG_OUT_HTML=False):
        self.FLAG_DEBUG_PRINT = FLAG_DEBUG_PRINT
        self.FLAG_INFO_PRINT = FLAG_INFO_PRINT
        self.FLAG_OUT_HTML = FLAG_OUT_HTML

    # Debug
    def printD(self, msg, head=""):
      if self.FLAG_DEBUG_PRINT:
        print(head + "### Debug: " + msg)

    def imshowD(self, img, title=""):
      if self.FLAG_DEBUG_PRINT:
        plt.title("### Debug " + title)
        plt.imshow(img)
        plt.show()

    def printProgressBarI(self, value, max_v):
        if self.FLAG_INFO_PRINT:
            if self.FLAG_OUT_HTML:
                return HTML_holder(value, max_v)
            else:
                return STDOUT_holder(value, max_v)

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

PM = PrintManager(FLAG_DEBUG_PRINT, FLAG_INFO_PRINT, FLAG_OUT_HTML)


# Testing
def main():
    import time

    FLAG_DEBUG_PRINT = True
    FLAG_INFO_PRINT = True
    FLAG_OUT_HTML = False

    pm = PrintManager(FLAG_DEBUG_PRINT, FLAG_INFO_PRINT, FLAG_OUT_HTML)

    pm.printD(bcolors.LIGHTGREEN+"ciao"+bcolors.ENDC)
    pm.printI(bcolors.DARKGREEN+"ciao"+bcolors.ENDC)

    pb = pm.printProgressBarI(0, 5-1)
    for i in range(5):
        pb.update(i)
        time.sleep(0.3)

if __name__ == "__main__":
    main()