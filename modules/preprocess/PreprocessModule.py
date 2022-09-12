
import sys
sys.path.append('../../')


from enum import Enum

import cv2
import numpy as np

from abc import ABC, abstractmethod

from modules.utility import PM

class PreprocessEnum(Enum):
    RESIZED = "RESIZED"
    SOBEL = "SOBEL"
    CROPPING = "CROPPING"

    QUAT_PURE = "QUAT_PURE"
    QUAT_GRAY = "QUAT_GRAY"
    QUAT_SOBEL = "QUAT_SOBEL"


class AbstractPreprocess(ABC):
    name = "Abstract"
    ch = 0

    def __init__(self, imgSize, imreadFlag=cv2.IMREAD_UNCHANGED, interpolation=cv2.INTER_AREA, step=1):
        self.imgSize = imgSize
        self.imreadFlag = imreadFlag
        self.interpolation = interpolation
        self.step = step

    @abstractmethod
    def processImage(self, imgPath):
        raise NotImplementedError

    @abstractmethod
    def printImage(self, image):
        raise NotImplementedError

    def printName(self):
        print(self.name)

    def suffix(self):
        return f"_{self.name}_{self.imgSize[0]}_{self.imgSize[1]}_{self.step}_loaded.npy"



class ResizedPreprocess(AbstractPreprocess):
    name = PreprocessEnum.RESIZED.value
    ch = 3

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def printImage(self, image):
        PM.imshowD(image)

    def processImage(self, imgPath):
        im = cv2.imread(imgPath, self.imreadFlag)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, self.imgSize, self.interpolation)
        return im / 255.0


class SobelPreprocess(AbstractPreprocess):
    name = PreprocessEnum.SOBEL.value
    ch = 2

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def printImage(self, image):
        PM.imshowD(image)

    def processImage(self, imgPath):
        im = cv2.imread(imgPath, self.imreadFlag)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, self.imgSize, self.interpolation)

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0,
                           borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0,
                           borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        gray = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        gray = np.reshape(gray, gray.shape + (1,))

        return gray / 255.0


class CroppingPreprocess(AbstractPreprocess):
    name = PreprocessEnum.CROPPING.value
    ch = 3

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def printImage(self, image):
        PM.imshowD(image)

    def processImage(self, imgPath):
        im = cv2.imread(imgPath, self.imreadFlag)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w, _ = im.shape
        w2 = round(w/2)

        top = h-self.imgSize[0]
        bottom = h

        iS_w2 = round(self.imgSize[1]/2)
        left = w2-iS_w2
        if self.imgSize[1]%2 == 0:
            right = w2+iS_w2
        else:
            right = w2+iS_w2+1

        im = im[top:bottom, left:right]
        return im / 255.0


class QuatPurePreprocess(AbstractPreprocess):
    name = PreprocessEnum.QUAT_PURE.value
    ch = 4

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def printImage(self, image):
        PM.imshowD(image[:, :, 1:4])
        PM.imshowD(image[:, :, 0])

    def processImage(self, imgPath):
        imRGB = cv2.imread(imgPath, self.imreadFlag)
        imRGB = cv2.cvtColor(imRGB, cv2.COLOR_BGR2RGB)
        h, w, _ = imRGB.shape
        imBlack = np.zeros((h, w, 1), np.uint8)

        quatImg = np.concatenate((imBlack, imRGB), 2)
        quatImg = cv2.resize(quatImg, self.imgSize, interpolation = self.interpolation)
        return quatImg / 255.0


class QuatGrayPreprocess(AbstractPreprocess):
    name = PreprocessEnum.QUAT_GRAY.value
    ch = 4

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def printImage(self, image):
        PM.imshowD(image[:, :, 1:4])
        PM.imshowD(image[:, :, 0])

    def processImage(self, imgPath):
        imRGB = cv2.imread(imgPath, self.imreadFlag)
        imRGB = cv2.cvtColor(imRGB, cv2.COLOR_BGR2RGB)
        h, w, _ = imRGB.shape
        imGray = np.reshape(cv2.cvtColor(imRGB, cv2.COLOR_BGR2GRAY), (h, w, 1))

        quatImg = np.concatenate((imGray, imRGB), 2)
        quatImg = cv2.resize(quatImg, self.imgSize, interpolation = self.interpolation)
        return quatImg / 255.0


class QuadSobelPreprocess(AbstractPreprocess):
    name = PreprocessEnum.QUAT_SOBEL.value
    ch = 4

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def printImage(self, image):
        PM.imshowD(image[:, :, 1:4])
        PM.imshowD(image[:, :, 0])

    def processImage(self, imgPath):
        imRGB = cv2.imread(imgPath, self.imreadFlag)
        imRGB = cv2.cvtColor(imRGB, cv2.COLOR_BGR2RGB)
        h, w, _ = imRGB.shape
        imGray = np.reshape(cv2.cvtColor(imRGB, cv2.COLOR_BGR2GRAY), (h, w, 1))

        quatImg = np.concatenate((imGray, imRGB), 2)
        quatImg = cv2.resize(quatImg, self.imgSize, interpolation = self.interpolation)
        return quatImg / 255.0


class PreprocessFactory():
    PreprocessEnum = PreprocessEnum

    @staticmethod
    def build(type_p: PreprocessEnum,
              imgSize, imreadFlag=cv2.IMREAD_UNCHANGED, interpolation=cv2.INTER_AREA, step=1):
        if type_p == PreprocessEnum.RESIZED:
            preprocess = ResizedPreprocess(imgSize, imreadFlag=imreadFlag,
                                             interpolation=interpolation, step=step)
        elif type_p == PreprocessEnum.SOBEL:
            preprocess = SobelPreprocess(imgSize, imreadFlag=imreadFlag,
                                             interpolation=interpolation, step=step)
        elif type_p == PreprocessEnum.CROPPING:
            preprocess = CroppingPreprocess(imgSize, imreadFlag=imreadFlag,
                                             interpolation=interpolation, step=step)

        elif type_p == PreprocessEnum.QUAT_PURE:
            preprocess = QuatPurePreprocess(imgSize, imreadFlag=imreadFlag,
                                             interpolation=interpolation, step=step)
        elif type_p == PreprocessEnum.QUAT_GRAY:
            preprocess = QuatGrayPreprocess(imgSize, imreadFlag=imreadFlag,
                                             interpolation=interpolation, step=step)
        elif type_p == PreprocessEnum.QUAT_SOBEL:
            preprocess = QuadSobelPreprocess(imgSize, imreadFlag=imreadFlag,
                                             interpolation=interpolation, step=step)

        else:
            raise ValueError

        return preprocess

    @staticmethod
    def listPreproc(imgSize, imreadFlag=cv2.IMREAD_UNCHANGED, interpolation=cv2.INTER_AREA, step=1):
        return [ResizedPreprocess(imgSize, imreadFlag=imreadFlag, interpolation=interpolation, step=step),
                SobelPreprocess(imgSize, imreadFlag=imreadFlag, interpolation=interpolation, step=step),
                CroppingPreprocess(imgSize, imreadFlag=imreadFlag, interpolation=interpolation, step=step),
                ]

    @staticmethod
    def listQuatPreproc(imgSize, imreadFlag=cv2.IMREAD_UNCHANGED, interpolation=cv2.INTER_AREA, step=1):
        return [QuatPurePreprocess(imgSize, imreadFlag=imreadFlag, interpolation=interpolation, step=step),
                QuatGrayPreprocess(imgSize, imreadFlag=imreadFlag, interpolation=interpolation, step=step),
                QuadSobelPreprocess(imgSize, imreadFlag=imreadFlag, interpolation=interpolation, step=step),
                ]

    @staticmethod
    def listAllPreproc(imgSize, imreadFlag=cv2.IMREAD_UNCHANGED, interpolation=cv2.INTER_AREA, step=1):
        return [*PreprocessFactory.listPreproc(imgSize, imreadFlag=imreadFlag, interpolation=interpolation, step=step),
                *PreprocessFactory.listQuatPreproc(imgSize, imreadFlag=imreadFlag, interpolation=interpolation, step=step)]


def main():
    PM.setFlags(True, True, False)

    imgPath = "../../Dataset/sequences/00/image_2/000005.png"
    imgSize = (1280, 384)

    ffs = PreprocessFactory.listAllPreproc(imgSize)

    for ff in ffs:
        im = ff.processImage(imgPath)
        ff.printName()
        print(im.shape)
        ff.printImage(im)


if __name__ == "__main__":
    main()


