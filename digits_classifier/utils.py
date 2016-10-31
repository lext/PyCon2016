from skimage.transform import resize, SimilarityTransform, warp
import numpy as np

def make_mnist(img):
    # Padding
    R, C = img.shape
    pad=450
    tmp = np.zeros((R+2*pad, C+2*pad)).astype(int)
    tmp[pad:pad+R,pad:pad+C] = img
    # Computing the bounding box
    nonzY, nonzX = np.where(tmp)
    ly, lx = nonzY.min(), nonzX.min()
    ry, rx = nonzY.max(), nonzX.max()


    if (rx-lx) < (ry-ly):
        rx = lx+(ry-ly)

    if (rx-lx) > (ry-ly):
        ry = ly+(rx-lx)

    img = resize(tmp[ly:ry,lx:rx].astype(float), (20, 20))
    # Now inserting the 20x20 image
    tmp = np.zeros((28,28))
    tmp[0:20,0:20] = img

    # Calculating translation
    
    Y, X = np.where(tmp)
    R, C = tmp.shape

    tsy, tsx = np.round(R/2-Y.mean()), np.round(C/2-X.mean())
    # Moving the digit
    tf = SimilarityTransform(translation=(-tsx, -tsy))
    tmp = warp(tmp, tf)
    return np.round(tmp).astype(int)
