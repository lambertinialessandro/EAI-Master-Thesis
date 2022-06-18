
from enum import Enum

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import grey_dilation, grey_erosion
from skimage.util import img_as_ubyte
from skimage.morphology import disk
from skimage.filters.rank import entropy

from abc import ABC, abstractmethod


from modules.utility import PM


class AbstractPreprocess(ABC):
    name = "Abstract"

    def __init__(self, imgSize, imreadFlag=cv2.IMREAD_UNCHANGED, interpolation=cv2.INTER_AREA):
        self.imgSize = imgSize
        self.imreadFlag = imreadFlag
        self.interpolation = interpolation

    @abstractmethod
    def processImage(self, imgPath):
        raise NotImplementedError

    def printImage(self, image):
        raise NotImplementedError

    def printName(self):
        print(self.name)

    def suffix(self):
        return self.name, self.imgSize[0], self.imgSize[1]



class UnchangedPreprocess(AbstractPreprocess):
    name = "UNCHANGED"

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def printImage(self, image):
        PM.imshowD(image)

    def processImage(self, imgPath):
        im = cv2.imread(imgPath, self.imreadFlag)
        im = cv2.resize(im, self.imgSize, self.interpolation)
        return im


class SobelPreprocess(AbstractPreprocess):
    name = "SOBEL"

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def printImage(self, image):
        PM.imshowD(image)

    def processImage(self, imgPath):
        im = cv2.imread(imgPath, self.imreadFlag)
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

        return gray


class CroppingPreprocess(AbstractPreprocess):
    name = "CROPPING"

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def printImage(self, image):
        PM.imshowD(image)

    def processImage(self, imgPath):
        im = cv2.imread(imgPath, self.imreadFlag)
        h, w, _ = im.shape
        w2 = round(w/2)

        top = h-self.imgSize[1]
        bottom = h

        iS_w2 = round(self.imgSize[0]/2)
        left = w2-iS_w2
        if self.imgSize[0]%2 == 0:
            right = w2+iS_w2
        else:
            right = w2+iS_w2+1

        im = im[top:bottom, left:right]
        return im


class QuatPurePreprocess(AbstractPreprocess):
    name = "QUAT_PURE"

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def printImage(self, image):
        PM.imshowD(image[:, :, 1:4])
        PM.imshowD(image[:, :, 0])

    def processImage(self, imgPath):
        imRGB = cv2.imread(imgPath, self.imreadFlag)
        h, w, _ = imRGB.shape
        imBlack = np.zeros((h, w, 1), np.uint8)

        quatImg = np.concatenate((imBlack, imRGB), 2)
        quatImg = cv2.resize(quatImg, self.imgSize, interpolation = self.interpolation)
        return quatImg


class QuatGrayPreprocess(AbstractPreprocess):
    name = "QUAT_GRAY"

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def printImage(self, image):
        PM.imshowD(image[:, :, 1:4])
        PM.imshowD(image[:, :, 0])

    def processImage(self, imgPath):
        imRGB = cv2.imread(imgPath, self.imreadFlag)
        h, w, _ = imRGB.shape
        imGray = np.reshape(cv2.cvtColor(imRGB, cv2.COLOR_BGR2GRAY), (h, w, 1))

        quatImg = np.concatenate((imGray, imRGB), 2)
        quatImg = cv2.resize(quatImg, self.imgSize, interpolation = self.interpolation)
        return quatImg


class QuadCEDPreprocess(AbstractPreprocess):
    name = "QUAT_CED"

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def printImage(self, image):
        PM.imshowD(image[:, :, 1:4])
        PM.imshowD(image[:, :, 0])

    def processImage(self, imgPath):
        imRGB = cv2.imread(imgPath, self.imreadFlag)
        imRGB = cv2.resize(imRGB, (1241, 376), interpolation = self.interpolation)
        imGray = cv2.cvtColor(imRGB, cv2.COLOR_BGR2GRAY)

        canny_img = cv2.Canny(np.array(imRGB), 100, 200)
        canny_img = Image.fromarray(canny_img)

        scaled_entropy = img_as_ubyte(canny_img / np.max(canny_img))
        entropy_image = entropy(scaled_entropy, disk(2))
        scaled_entropy = entropy_image / entropy_image.max()
        mask = scaled_entropy > 0.75
        maskedImg = imGray * mask

        dilated = grey_dilation(maskedImg, footprint=np.ones((3,3)))
        dilated = grey_erosion(dilated, size=(3,3))
        dilated = np.array(Image.fromarray(dilated))

        im_CED = np.reshape(dilated, (376, 1241, 1))
        quatImg = np.concatenate((im_CED, imRGB), 2)
        quatImg = cv2.resize(quatImg, self.imgSize, interpolation = self.interpolation)
        return quatImg


class PreprocessEnum(Enum):
    UNCHANGED = "UNCHANGED"
    SOBEL = "SOBEL"
    CROPPING = "CROPPING"

    QUAD_PURE = "QUAD_PURE"
    QUAD_GRAY = "QUAD_GRAY"
    QUAD_CED = "QUAD_CED" # CANNY_ENTOPY_DILATED


class PreprocessFactory():
    PreprocessEnum = PreprocessEnum

    def build(type_p: PreprocessEnum,
              imgSize, imreadFlag=cv2.IMREAD_UNCHANGED, interpolation=cv2.INTER_AREA):
        if type_p == PreprocessEnum.UNCHANGED:
            preprocess = UnchangedPreprocess(imgSize, imreadFlag=imreadFlag,
                                             interpolation=interpolation)
        elif type_p == PreprocessEnum.SOBEL:
            preprocess = SobelPreprocess(imgSize, imreadFlag=imreadFlag,
                                             interpolation=interpolation)
        elif type_p == PreprocessEnum.CROPPING:
            preprocess = CroppingPreprocess(imgSize, imreadFlag=imreadFlag,
                                             interpolation=interpolation)

        elif type_p == PreprocessEnum.QUAD_PURE:
            preprocess = QuatPurePreprocess(imgSize, imreadFlag=imreadFlag,
                                             interpolation=interpolation)
        elif type_p == PreprocessEnum.QUAD_GRAY:
            preprocess = QuatGrayPreprocess(imgSize, imreadFlag=imreadFlag,
                                             interpolation=interpolation)
        elif type_p == PreprocessEnum.QUAD_CED:
            preprocess = QuadCEDPreprocess(imgSize, imreadFlag=imreadFlag,
                                             interpolation=interpolation)

        else:
            raise ValueError

        return preprocess

    def listPreproc(imgSize, imreadFlag=cv2.IMREAD_UNCHANGED, interpolation=cv2.INTER_AREA):
        return [UnchangedPreprocess(imgSize, imreadFlag=imreadFlag, interpolation=interpolation),
                SobelPreprocess(imgSize, imreadFlag=imreadFlag, interpolation=interpolation),
                CroppingPreprocess(imgSize, imreadFlag=imreadFlag, interpolation=interpolation),
                ]
    def listQuatPreproc(imgSize, imreadFlag=cv2.IMREAD_UNCHANGED, interpolation=cv2.INTER_AREA):
        return [QuatPurePreprocess(imgSize, imreadFlag=imreadFlag, interpolation=interpolation),
                QuatGrayPreprocess(imgSize, imreadFlag=imreadFlag, interpolation=interpolation),
                QuadCEDPreprocess(imgSize, imreadFlag=imreadFlag, interpolation=interpolation),
                ]

    def listAllPreproc(imgSize, imreadFlag=cv2.IMREAD_UNCHANGED, interpolation=cv2.INTER_AREA):
        return [*PreprocessFactory.listPreproc(imgSize, imreadFlag=imreadFlag, interpolation=interpolation),
                *PreprocessFactory.listQuatPreproc(imgSize, imreadFlag=imreadFlag, interpolation=interpolation)]


def main():
    PM.setFlags(True, True, False)

    imgPath = "./Dataset/sequences/00/image_2/000000.png"
    imgSize = (256, 256)

    ffs = PreprocessFactory.listAllPreproc(imgSize)

    for ff in ffs:
        im = ff.processImage(imgPath)
        ff.printName()
        print(im.shape)
        ff.printImage(im)


if __name__ == "__main__":
    main()


