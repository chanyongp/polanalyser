# Polanalyser
Polanalyser is polarization image analysis tool. It can be used for demosaicing of bayer images taken with a polarization camera, analysis of Stokes vector and Muller matrix.

## Requirement
* OpenCV
* Numpy
* Numba

## Usage
### import 
```python
import polanalyser as pa
```

### Polarization demosaicing
Demosaic monochrome polarization bayer image taken with the [IMX250MZR](https://www.sony-semicon.co.jp/e/products/IS/polarization/product.html) sensor.
```python
import cv2
import polanalyser as pa

img_bayer = cv2.imread("images/polarizer_IMX250MZR.png", -1)

img_pola = pa.IMX250MZR.demosaicing(img_bayer)

img_pola0   = img_pola[:,:,0]
img_pola45  = img_pola[:,:,1]
img_pola90  = img_pola[:,:,2]
img_pola135 = img_pola[:,:,3]
```

## Estimate the Stokes parameters
```python
import cv2
import numpy as np
import polanalyser as pa

img_bayer = cv2.imread("images/polarizer_IMX250MZR.png", -1)
img_pola = pa.IMX250MZR.demosaicing(img_bayer)

radians = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])
img_stokes = pa.calcStokes(img_pola, radians)

img_S0, img_S1, img_S2 = cv2.split(img_stokes)

img_Imax = pa.cvtStokesToImax(img_stokes)
img_Imin = pa.cvtStokesToImin(img_stokes)
img_DoLP = pa.cvtStokesToDoLP(img_stokes)
img_AoLP = pa.cvtStokesToAoLP(img_stokes)
```

|Example of results| | |
|:-:|:-:|:-:|
|Normal ((Imax+Imin)/2.0)|Imax|Imin|
|![](https://github.com/elerac/PolarizationImaging/blob/manuscript/polarizer_IMX250MZR_intensity.jpg)|![](https://github.com/elerac/PolarizationImaging/blob/manuscript/polarizer_IMX250MZR_max.jpg)|![](https://github.com/elerac/PolarizationImaging/blob/manuscript/polarizer_IMX250MZR_min.jpg)|
|DoLP|AoLP||
|![](https://github.com/elerac/PolarizationImaging/blob/manuscript/polarizer_IMX250MZR_DoLP.jpg)|![](https://github.com/elerac/PolarizationImaging/blob/manuscript/polarizer_IMX250MZR_AoLP.jpg)||
