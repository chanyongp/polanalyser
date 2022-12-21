import warnings
import cv2
import numpy as np
from .mueller import polarizer


def calcStokes(intensities, muellers):
    """Calculate stokes vector from observed intensities and mueller matrix

    Parameters
    ----------
    intensities : np.ndarray
      Measured intensities. (height, width, K)
    muellers : np.ndarray
      Mueller matrix. (3, 3, K) or (4, 4, K)

    Returns
    -------
    stokes : np.ndarray
      Stokes vector. (height, width, 3) or (height, width, 4)
    """
    if not isinstance(intensities, np.ndarray):
        intensities = np.stack(intensities, axis=-1)  # (height, width, K)

    if not isinstance(muellers, np.ndarray):
        muellers = np.stack(muellers, axis=-1)  # (3, 3, K) or (4, 4, K)

    # 1D array case
    if muellers.ndim == 1:    
        thetas = muellers
        return calcLinearStokes(intensities, thetas)

    A = muellers[0].T  # [m11, m12, m13] (K, 3) or [m11, m12, m13, m14] (K, 4)
    A_pinv = np.linalg.pinv(A)  # (3, K) or (K, 4)
    stokes = np.tensordot(A_pinv, intensities, axes=(1, -1))  # (3, height, width) or (4, height, width)
    stokes = np.moveaxis(stokes, 0, -1)  # (height, width, 3) or (height, width, 4)
    return stokes

def calcLinearStokes(intensities, thetas):
    """Calculate only linear polarization stokes vector from observed intensity and linear polarizer angle
    
    Parameters
    ----------
    intensities : np.ndarray
      Intensity of measurements (height, width, n)
    theta : np.ndarray
      Linear polarizer angles (n, )

    Returns
    -------
    S : np.ndarray
      Stokes vector (height, width, 3)
    """
    muellers = [ polarizer(theta)[..., :3, :3] for theta in thetas ]
    return calcStokes(intensities, muellers)

def cvtStokesToImax(img_stokes):
    """
    Convert stokes vector image to Imax image

    Parameters
    ----------
    img_stokes : np.ndarray, (height, width, 3)
        Stokes vector image

    Returns
    -------
    img_Imax : np.ndarray, (height, width)
        Imax image
    """
    S0 = img_stokes[..., 0]
    S1 = img_stokes[..., 1]
    S2 = img_stokes[..., 2]
    return (S0+np.sqrt(S1**2+S2**2))*0.5

def cvtStokesToImin(img_stokes):
    """
    Convert stokes vector image to Imin image

    Parameters
    ----------
    img_stokes : np.ndarray, (height, width, 3)
        Stokes vector image

    Returns
    -------
    img_Imin : np.ndarray, (height, width)
        Imin image
    """
    S0 = img_stokes[..., 0]
    S1 = img_stokes[..., 1]
    S2 = img_stokes[..., 2]
    return (S0-np.sqrt(S1**2+S2**2))*0.5

def cvtStokesToDoLP(img_stokes):
    """
    Convert stokes vector image to DoLP (Degree of Linear Polarization) image

    Parameters
    ----------
    img_stokes : np.ndarray, (height, width, 3)
        Stokes vector image

    Returns
    -------
    img_DoLP : np.ndarray, (height, width)
        DoLP image ∈ [0, 1]
    """
    S0 = img_stokes[..., 0]
    S1 = img_stokes[..., 1]
    S2 = img_stokes[..., 2]
    return np.sqrt(S1**2+S2**2)/S0


def cvtStokesToAoLP(img_stokes):
    """
    Convert stokes vector image to AoLP (Angle of Linear Polarization) image

    Parameters
    ----------
    img_stokes : np.ndarray, (height, width, 3)
        Stokes vector image

    Returns
    -------
    img_AoLP : np.ndarray, (height, width)
        AoLP image ∈ [0, np.pi]
    """
    S1 = img_stokes[..., 1]
    S2 = img_stokes[..., 2]
    return np.mod(0.5*np.arctan2(S2, S1), np.pi)

def cvtStokesToIntensity(img_stokes):
    """
    Convert stokes vector image to intensity image

    Parameters
    ----------
    img_stokes : np.ndarray, (height, width, 3)
        Stokes vector image

    Returns
    -------
    img_intensity : np.ndarray, (height, width)
        Intensity image
    """
    S0 = img_stokes[..., 0]
    return S0*0.5

def cvtStokesToDiffuse(img_stokes):
    """
    Convert stokes vector image to diffuse image

    Parameters
    ----------
    img_stokes : np.ndarray, (height, width, 3)
        Stokes vector image

    Returns
    -------
    img_diffuse : np.ndarray, (height, width)
        Diffuse image
    """
    Imin = cvtStokesToImin(img_stokes)
    return 1.0*Imin

def cvtStokesToSpecular(img_stokes):
    """
    Convert stokes vector image to specular image

    Parameters
    ----------
    img_stokes : np.ndarray, (height, width, 3)
        Stokes vector image

    Returns
    -------
    img_specular : np.ndarray, (height, width)
        Specular image
    """
    S1 = img_stokes[..., 1]
    S2 = img_stokes[..., 2]
    return np.sqrt(S1**2+S2**2) #same as Imax-Imin

def cvtStokesToDoP(img_stokes):
    """
    Convert stokes vector image to DoP (Degree of Polarization) image

    Parameters
    ----------
    img_stokes : np.ndarray, (height, width, 3)
        Stokes vector image

    Returns
    -------
    img_DoP : np.ndarray, (height, width)
        DoP image ∈ [0, 1]
    """
    S0 = img_stokes[..., 0]
    S1 = img_stokes[..., 1]
    S2 = img_stokes[..., 2]
    S3 = img_stokes[..., 3]
    return np.sqrt(S1**2+S2**2+S3**2)/S0

def cvtStokesToEllipticityAngle(img_stokes):
    """
    Convert stokes vector image to ellipticity angle image

    Parameters
    ----------
    img_stokes : np.ndarray, (height, width, 3)
        Stokes vector image

    Returns
    -------
    img_EllipticityAngle : np.ndarray, (height, width)
        ellipticity angle image ∈ [-pi/4, pi/4]
    """
    S1 = img_stokes[..., 1]
    S2 = img_stokes[..., 2]
    S3 = img_stokes[..., 3]
    return 0.5*np.arctan2(S3, np.sqrt(S1**2+S2**2))

def cvtStokesToDoCP(img_stokes):
    """
    Convert stokes vector image to DoCP (Degree of Circular Polarization) image

    Parameters
    ----------
    img_stokes : np.ndarray, (height, width, 3)
        Stokes vector image

    Returns
    -------
    img_DoCP : np.ndarray, (height, width)
        DoCP image ∈ [-1, 1]
    """
    S0 = img_stokes[..., 0]
    S3 = img_stokes[..., 3]
    return S3 / S0



def applyColorToStokes(img_stokes, gain=1.0, positive_color="green", negative_color="red"):
    """Apply color to Stokes image

    By default, positive value is "green" channel and negative value is "red" channel.
    
    Parameters
    ----------
    img_stokes : np.ndarray
        Stokes image.
    gain : float
        Gain parameter to adjust image intensity.
    positive_color : str
        Select from ["blue", "green", "red"]. Default "green".
    negative_color : str
        Select from ["blue", "green", "red"]. Default "red".

    Returns
    -------
    img_stokes_color : np.ndarray
        Colored stokes image. shape is (*img_stokes, 3) and dtype is `uint8`
    """
    img_stokes = gain * img_stokes

    is_overflow = np.any(np.abs(img_stokes) > 255)
    if is_overflow:
        warnings.warn("'img_stokes' absolute values are overflowed (>255). Clip values in the range -255 to 255.", stacklevel=2)
        img_stokes = np.clip(img_stokes, -255.0, 255.0)

    img_stokes_color = np.empty((*img_stokes.shape, 3), dtype=np.uint8)

    for i, color in enumerate(["blue", "green", "red"]):
        if color == positive_color:
            img_stokes_color[..., i] = np.maximum(img_stokes, 0)
        elif color == negative_color:
            img_stokes_color[..., i] = np.maximum(-img_stokes, 0)
        else:
            img_stokes_color[..., i] = 0
    
    return img_stokes_color