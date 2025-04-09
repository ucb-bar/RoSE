import cv2 
import numpy as np
import matplotlib.pyplot as plt

# read the computed image
if __name__ == '__main__':

    width = 256-32-8
    height = 256-8
    # read raw disparity map bytes
    # Read the raw bytes from the file
    with open('capture.txt', 'rb') as f:
        raw_bytes = f.read()

    # Split the raw bytes into individual byte values
    byte_values = raw_bytes.split()

    # Convert the byte values to integers and store them in a NumPy array of uint8
    data_array = np.array([int(x, 16) for x in byte_values], dtype=np.uint8)
    # print the first 10 values
    print(data_array[:10])
    # reshape to original shape
    curr_height = data_array.shape[0] / width
    disp = data_array.reshape(int(curr_height), width)
    print(f"max: {np.max(disp)}, min: {np.min(disp)}, mean: {np.mean(disp)}")
    # generate a distribution of the values
    plt.hist(disp.ravel())
    # save the histogram
    # plt.savefig('output/img/c_histogram.png')
    # create a greyscale image from the numpy array
    disparity_normalized = cv2.normalize(disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disparity_normalized = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
    # store the image
    cv2.imwrite('disparity.png', disparity_normalized)