import pandas as pd
import numpy as np
import cv2  # Represent images as BGR
import matplotlib.pyplot as plt  # Represent images as RGB

from glob import glob


img_mpl = plt.imread("Snap-24074.tif")
img_cv2 = cv2.imread("Snap-24074.tif", 0)
print(img_cv2.shape)
cv2.imwrite('grey.png', img_cv2)
# cv2.imshow("Greyscale CV2", img_cv2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Image array
'''print(img_mpl.shape, img_cv2.shape)  # 3D arrays (height, width, RGB)
print(img_mpl)
pd.Series(img_mpl.flatten()).plot(kind='hist', bins=50, title='Distribution of pixel values')
plt.show()'''

# Display images
'''fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_mpl)
ax.axis('off')
plt.show()'''

# Display RGB Channels of our image (with mpl)
'''fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 by 3 grid of plots
axs[0].imshow(img_mpl[:, :, 0], cmap='Reds')
axs[1].imshow(img_mpl[:, :, 1], cmap='Greens')
axs[2].imshow(img_mpl[:, :, 2], cmap='Blues')
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
plt.show()'''

# Display RGB Channels of our image (with cv2)
'''fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img_cv2)
axs[1].imshow(img_mpl)
axs[0].axis('off')
axs[1].axis('off')
axs[0].set_title('CV Image')
axs[1].set_title('Matplotlib Image')
plt.show()'''

# Converting from BGR to RGB
'''img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_cv2_rgb)
ax.axis('off')
plt.show()'''


# Image manipulation
# Converting RGB to GRAY
def rgb_to_gray(img):
    img_grey = cv2.cvtColor(img_mpl, cv2.COLOR_RGB2GRAY)
    print(img_grey.shape)  # Converted to 2D array
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_grey, cmap='Greys')
    plt.show()

    return img_grey

# Resizing and scaling
'''img_resized = cv2.resize(img_mpl, None, fx=0.25, fy=0.25)  # Resize by a certain percentage, dsize is a determined
                                                              # new size
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img_resized)
ax.axis('off')
plt.show()'''

# Predetermined size
'''img_resized = cv2.resize(img_mpl, (100, 200))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img_resized)
ax.axis('off')
plt.show()'''

# Up-scaling, ex. machine learning
'''img_upscale = cv2.resize(img_mpl, (5000, 2500), interpolation=cv2.INTER_CUBIC)  # Choose different interpolations
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img_upscale)
ax.axis('off')
plt.show()'''

# CV2 kernels
# Sharpen Image
'''kernel_sharpening = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
img_sharpened = cv2.filter2D(img_mpl, -1, kernel_sharpening)  # Image, depth and filtering kernel
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img_sharpened)
ax.axis('off')
plt.show()'''

# Blurring Image
'''kernel_3x3 = np.ones((3, 3), np.float32) / 9  # 3x3 kernel filled with 0.1111111
img_blurred = cv2.filter2D(img_mpl, -1, kernel_3x3)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img_blurred)
ax.axis('off')
plt.show()'''


# Gaussian blur
def gaussian_blurr(img):
    kernel_gauss = np.array([[1, 2, 1],
                             [2, 4, 2],
                             [1, 2, 1]]) / 16
    img_gauss = cv2.filter2D(img, -1, kernel_gauss)  # Image, depth and filtering kernel
    # img_gauss = cv2.filter2D(img_gauss, -1, kernel_gauss)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_gauss)
    ax.axis('off')
    plt.show()

    return img_gauss


def crop(img, y_start, y_end, x_start, x_end):
    cropped_image = img[y_start:y_end, x_start:x_end]
    return cropped_image


# Saving Image
'''plt.imsave('mlp_rzeszow.png', img_blurred)
cv2.imwrite('cv2_rzeszow.png', img_blurred)'''


# Edge detection - Sobel operator
'''kernel_gx = np.array([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])
kernel_gy = np.array([[1, 2, 1],
                      [0, 0, 0],
                      [-1, 2, -1]])
# img_gray = rgb_to_gray(img_mpl)  # Sharpens colonies in the right layer
img_gauss = gaussian_blurr(img_cv2)  # Returns colored image?
img_sobel = cv2.filter2D(img_gauss, -1, kernel_gx)
img_sobel = cv2.filter2D(img_sobel, -1, kernel_gy)  # Higher layer is sharpened? Need to lower filtering
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img_sobel)
ax.axis('off')
plt.show()'''


# Fun
'''img_cropped = crop(img_cv2, *(500, 1500), *(100, 2200))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img_cropped)
ax.axis('off')
plt.show()'''
img_gauss = cv2.GaussianBlur(img_cv2, (5, 5), sigmaX=0, sigmaY=0)  # Sharper colonies
# cv2.imwrite('Gaussian_blur.png', img_gauss)
# img_bilateral = cv2.bilateralFilter(img_cv2, 9, 75, 75)  # Less background
# cv2.imwrite('Bilateral_blur.png', img_bilateral)
# img_edge_detec = cv2.Sobel(img_gauss, -1, )


# Thresholding
# th, dst = cv2.threshold(img_cv2, 110, 255, cv2.THRESH_BINARY_INV)
# cv2.imwrite('threshold.png', dst)
th, dst = cv2.threshold(img_gauss, 110, 250, cv2.THRESH_BINARY)
cv2.imwrite('threshold_gauss.png', dst)
# th, dst = cv2.threshold(img_bilateral, 110, 255, cv2.THRESH_BINARY_INV)
# cv2.imwrite('threshold_bilateral.png', dst)
# print(dst.shape)


# Blob detection
# Set up the detector with default parameters.
# Setup SimpleBlobDetector parameters.
'''params = cv2.SimpleBlobDetector.Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200

# Filter by Area.
params.filterByArea = True
params.minArea = 100

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.95

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.1

# Create a detector with the parameters
ver = cv2.__version__.split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector()
else:
    detector = cv2.SimpleBlobDetector.create(params)

keypoints = detector.detect(img_cv2)
img_with_keypoints = cv2.drawKeypoints(img_cv2, keypoints, np.array([]), (0, 0, 255),
                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('keypoints.png', img_with_keypoints)'''


# Edge detection - Sobel
'''sobel_x = cv2.Sobel(src=img_gauss, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobel_y = cv2.Sobel(src=img_gauss, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobel_xy = cv2.Sobel(src=img_gauss, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
cv2.imwrite('edge_sobel_x.jpg', sobel_x)
cv2.imwrite('edge_sobel_y.jpg', sobel_y)
cv2.imwrite('edge_sobel_xy.jpg', sobel_xy)'''

# Edge detection - Canny (much better than sobel)
edges = cv2.Canny(image=img_gauss, threshold1=120, threshold2=140)  # Tested best for min 120 max 140
cv2.imwrite('edge_canny.jpg', edges)


# Find contours
contours, hierarchy = cv2.findContours(image=dst, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_NONE)
grey_copy = img_cv2.copy()
cv2.drawContours(image=grey_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
cv2.imwrite('contours_none_grey.jpg', grey_copy)

