import cv2 
import numpy as np
import matplotlib.pyplot as plt

# read the computed image
if __name__ == '__main__':

    width = 128-16
    height = 128
    # read raw disparity map bytes
    with open('output', 'rb') as f:
        raw_disp = f.read()
    disp = np.frombuffer(raw_disp, dtype=np.int8)
    # reshape to original shape
    disp = disp.reshape(int(height), int(width))
    print(f"max: {np.max(disp)}, min: {np.min(disp)}, mean: {np.mean(disp)}")
    # generate a distribution of the values
    plt.hist(disp.ravel())
    # save the histogram
    plt.savefig('output/img/c_histogram.png')
    # create a greyscale image from the numpy array
    disparity_normalized = cv2.normalize(disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disparity_normalized = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
    # store the image
    cv2.imwrite('output/img/c_disparity.png', disparity_normalized)