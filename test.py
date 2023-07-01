"""
Analysis of Stokes vector
"""
import os

import cv2
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
from matplotlib import pyplot as plt

import polanalyser as pa


def convert_u8(image, gamma=1/2.2):
    image = np.clip(image, 0, 255).astype(np.uint8)
    lut = (255.0 * (np.linspace(0, 1, 256) ** gamma)).astype(np.uint8)
    return lut[image]

def main():
    # Read image and demosaicing
    
    Img_0 = cv2.imread('1104/1104_1853_0.JPG', 0)
    Img_45 = cv2.imread('1104/1104_1853_45.JPG', 0)
    Img_90 = cv2.imread('1104/1104_1853_90.JPG', 0)
    Img_135 = cv2.imread('1104/1104_1853_135.JPG', 0)
    Img_polarization = np.array([Img_0, Img_45, Img_90, Img_135], dtype=Img_0.dtype)
    Img_polarization = np.moveaxis(Img_polarization, 0, -1)

    # Calculate the Stokes vector per-pixel
    angles = np.deg2rad([0, 45, 90, 135])#각도 라디안으로 변환
    img_stokes = pa.calcStokes(Img_polarization, angles)#이미지 강도, 각 입력받아 스토크스 벡터 출력

    # Decomposition into 3 components (S0, S1, S2)
    img_S0, img_S1, img_S2 = cv2.split(img_stokes)

    # Convert Stokes vector to meaningful values
    img_intensity = pa.cvtStokesToIntensity(img_stokes)
    img_Imax = pa.cvtStokesToImax(img_stokes)
    img_Imin = pa.cvtStokesToImin(img_stokes)
    img_DoLP = pa.cvtStokesToDoLP(img_stokes) # 0~1
    img_AoLP = pa.cvtStokesToAoLP(img_stokes) # 0~pi
    #img_AoLP2 = pa.cvtStokesToAoLP2(img_stokes,73) # 0~pi
    
    # Normarize and save images
    name="1104_1853"
    
    cv2.imwrite(f"{name}_intensity.png", convert_u8(img_intensity))
    cv2.imwrite(f"{name}_Imax.png",      convert_u8(img_Imax))
    cv2.imwrite(f"{name}_Imin.png",      convert_u8(img_Imin))
    cv2.imwrite(f"{name}_DoLP.png",      convert_u8(img_DoLP*255, gamma=1))#밝을 수록 편광도가 높음
    cv2.imwrite(f"{name}_AoLP.png",      pa.applyColorToAoLP(img_AoLP)) # apply pseudo-color, The color map is based on HSV
    #cv2.imwrite(f"{name}_AoLP2.png",      pa.applyColorToAoLP(img_AoLP2)) 

    img = cv2.imread('/Users/parkchan-yong/Documents/생기부 관련/r&e,대회 등/r&e/polanalyser-master/examples/1019_1625_DoLP.png',0)
    plt.hist(img.ravel(), 256, [0,256]); 
    plt.show()
    """
    화소의 강도(Gray 값 또는 하나의 채널에 대한 값)를 갖는 화소의 개수를 각각 X축과 Y축으로 표시하고 있습니다. 대부분의 경우 화소의 강도는 0-255입니다.
    """
   
    
if __name__=="__main__":
    main()
