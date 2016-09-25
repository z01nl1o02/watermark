import os,sys,pdb,cPickle
import argparse
import numpy as np
from PIL import Image,ImageFilter,ImageEnhance
import random


def freq_show(fq):
    mag = np.absolute(fq)
    mag = np.log(mag) #enhance view effect
    mag = mag * 255 / mag.max()
    magImg = Image.fromarray(np.uint8(mag))
    magImg.show()
    return 


def gen_random_image(size):
    res = np.zeros(size,dtype=np.uint8)
    for y in range(size[0]):
        for x in range(size[1]):
            res[y,x] = random.randint(10,200) #should be avoid zero 
    return res

def load_hidden_image(path, size):
    img = Image.open(path)
    img = img.convert('L')
    img = img.resize((size[1],size[0]))
    img.show()
    return np.array(img)

def apply_watermark(imgFFT, randImg, hiddenImg,w):
    mag = np.absolute(imgFFT)
    ori = np.angle(imgFFT,False)

    #write to media freq to avoid precision loss during image storage
    lowFq = 0.0
    highFq = 1.0
    randImg = np.float64(randImg)
    radius = (imgFFT.shape[0] + imgFFT.shape[1])/4
    for y in range(imgFFT.shape[0]):
        for x in range(imgFFT.shape[1]):
            dx = x - imgFFT.shape[1] / 2
            dy = y - imgFFT.shape[0] / 2
            rr = (dx**2) + (dy**2)
            if rr < (lowFq * radius) ** 2 or rr > (highFq * radius) ** 2:
                continue
            if hiddenImg[y,x] > 128:
                flag = 1
            else:
                flag = -1

            mag[y,x] +=  w * randImg[y,x] * flag
    imgFFTNew = mag * np.cos(ori) + 1j * mag * np.sin(ori)
    return imgFFTNew


def restore_watermark(img, imgORG,randImg):
    imgFFT = np.fft.fft2(img)
    imgFFT = np.fft.fftshift(imgFFT)
    realImgSRC = np.absolute(imgFFT)

    if 0:
        imgTmp = Image.fromarray(img) 
        imgTmp = imgTmp.filter(ImageFilter.BLUR)
    #    imgTmp.show()
        imgTmp = np.array(imgTmp)
        imgFFT = np.fft.fft2( imgTmp )
        imgFFT = np.fft.fftshift(imgFFT)
        realImgSMOOTH = np.absolute(imgFFT)
    else:
        imgFFT = np.fft.fft2(imgORG)
        imgFFT = np.fft.fftshift(imgFFT)
        realImgSMOOTH = np.absolute(imgFFT)

    diffImg = realImgSRC - realImgSMOOTH
    hiddenImg = diffImg * randImg
     
    pos = hiddenImg > 0
    neg = hiddenImg <= 0
    hiddenImg = np.zeros( img.shape )
    hiddenImg[pos] = 255
    hiddenImg[neg] = 0
    hiddenImg = np.uint8(hiddenImg)    
    return hiddenImg    

def run(imgPath, hiddenPath):
    w = 0.01 #signal weight
    img = Image.open(imgPath)
    img = img.convert('L')
    img.show()
    img = np.array(img)
    imgORG  = img.copy() #used during de-watermark
    hiddenImg = load_hidden_image(hiddenPath,img.shape)

    #generate random singal
    randImg = gen_random_image(img.shape)
   
    #fft of image 
    imgFFT = np.fft.fft2(img)
    imgFFT = np.fft.fftshift(imgFFT)
    freq_show(imgFFT) 
    
    #apply watermark in freq
    imgFFT = apply_watermark(imgFFT, randImg,hiddenImg,w) 
    freq_show(imgFFT)


    #generate image with hidden watermark
    imgFFT = np.fft.fftshift(imgFFT)
    img = np.fft.ifft2(imgFFT)
    img = np.absolute(img)
    img = np.maximum(0,img)
    img = np.minimum(255,img)
    img = np.uint8(img)
    Image.fromarray(img).show()

    #extract watermark from image (non-blind watermark)
    restoredImg = restore_watermark(img,imgORG, randImg)
    Image.fromarray(restoredImg).show()

if __name__=="__main__":
    run('img/img.jpg', 'img/hidden.jpg')
