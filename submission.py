import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter


################## Functions Definition ##################
# Function to select the patch
def colorPatchSelector(action, x, y, flags, userdata):
    """This function performs a cropping over the region selected with a radius of 15 pixels and stores this patch
    in a variable called patches_list"""
    # Refer global variables
    global patches_list, r
    if action == cv2.EVENT_LBUTTONDOWN:
        center = (x, y)
        patch = selected_frame[
            center[1] - 2 * r : center[1] + 2 * r, center[0] - 2 * r : center[0] + 2 * r
        ]
        print(center)
        cv2.imshow("Patch", patch)
        # Append patches to the list
        patches_list.append(patch)


# Function to calculate the mean of colors selected
def colorPatchSampling(patches_list):
    """To extract the color information from the patch. This can be achieved by sampling the color values
    (e.g., RGB values) of all the pixels within the selected patch."""
    print("Sampling patches...")
    # Transform patches to HSV
    hsv_patches = []
    for patch in patches_list:
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hsv_patches.append(hsv_patch)
    mean_colors = []
    for hsv_patch in hsv_patches:
        mean_color = np.mean(
            hsv_patch, axis=(0, 1)
        )  # Calculate mean BGR color value for the patch
        mean_colors.append(mean_color)
    # Calculate the overall mean color value
    overall_mean_color = np.mean(mean_colors, axis=0).astype(int)
    # print(overall_mean_color)
    return overall_mean_color


# Define a callback function for the trackbar
def on_tolerance_change_HUE(*args):
    """Applies HUE values to a fixed tolerance of +/-1 for upper and lower values respectively"""
    global lower_bound, upper_bound
    tolerance = args[0] * 1.8
    lower_h = overall_mean_color[0] - tolerance % 180
    upper_H = overall_mean_color[0] + tolerance % 180
    lower_bound[0] = lower_h
    upper_bound[0] = upper_H


def on_tolerance_change_SATURIATION(*args):
    """Applies SATURATION values to a fixed tolerance of +/-1 for upper and lower values respectively"""
    global lower_bound, upper_bound
    tolerance = args[0] * 2.55
    lower_s = overall_mean_color[1] - tolerance % 256
    upper_S = overall_mean_color[1] + tolerance % 256
    lower_bound[1] = lower_s
    upper_bound[1] = upper_S


def on_tolerance_change_VALUE(*args):
    """Applies VALUE values to a fixed tolerance of +/-1 for upper and lower values respectively"""
    global lower_bound, upper_bound
    tolerance = args[0] * 2.55
    lower_v = overall_mean_color[2] - tolerance % 256
    upper_V = overall_mean_color[2] + tolerance % 256
    lower_bound[2] = lower_v
    upper_bound[2] = upper_V


def on_tolerance_change_BLUR(softness_value, image):
    ksize = (3, 3)
    sigma_x = softness_value
    sigma_y = softness_value
    blurred_image = cv2.GaussianBlur(image, ksize, sigma_x, sigma_y)
    return blurred_image


def on_tolerance_change_DEBLUR(softness_value, image):
    # Convert the image to grayscale if it's in color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the kernel size and standard deviation based on softness_value
    ksize = (3, 3)
    sigma = softness_value

    # Create the Gaussian kernel
    gaussian_kernel = cv2.getGaussianKernel(ksize[0], sigma)
    gaussian_kernel = gaussian_kernel * gaussian_kernel.T

    # Perform inverse filtering for each channel
    deblurred_channels = []
    for channel in cv2.split(image):
        channel_fft = np.fft.fft2(channel)
        kernel_fft = np.fft.fft2(gaussian_kernel, s=channel.shape)

        # Avoid division by zero
        kernel_fft[kernel_fft == 0] = 1e-6

        deblurred_channel_fft = channel_fft / kernel_fft
        deblurred_channel = np.abs(np.fft.ifft2(deblurred_channel_fft))

        # Convert the deblurred channel back to uint8 format
        # deblurred_channel = np.uint8(deblurred_channel)
        deblurred_channels.append(deblurred_channel)

    # Merge the deblurred channels back into an image
    deblurred_image = cv2.merge(deblurred_channels)

    return deblurred_image


def blur_arguments(*args):
    return args[0]


def apply_gaussian_blur(image, sigma_x=0.0, sigma_y=0.0, ksize=(3, 3)):
    blurred_image = cv2.GaussianBlur(image, ksize, sigma_x, sigma_y)
    return blurred_image


# Read the file
filename = r"..\Chroma Keying\greenscreen-asteroid.mp4"
# filename = "..\Chroma Keying\greenscreen-demo.mp4"
background_path = r"..\Chroma Keying\background.jpg"

# Resize image
new_width = 720
new_height = 380

# Use video capture to read the file and read the backgorund image
background_image = cv2.imread(background_path, cv2.IMREAD_UNCHANGED)
background_image = cv2.resize(background_image, (new_width, new_height))
cap = cv2.VideoCapture(filename)
dummy = cv2.VideoCapture(filename)
ret, frame = cap.read()

# Create a windows section
win_detection_name = "Video"
win_select_patch = "Select Patch"
win_select_patch_mask = "Mask"
win_select_patch_blur = "Softened image"

##################### PATCH SELECTION #####################
# Define a list of patches in which we append the patches
patches_list = []
blurred_images = []

# Define start point and end point for patch selection
r = 15

# Frame number you want to select
target_frame_number = 75  # Change this to the frame number you want

# Loop through the video frames until you reach the target frame
current_frame_number = 0
while current_frame_number < target_frame_number:
    ret, selected_frame = dummy.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    current_frame_number += 1

selected_frame = cv2.resize(selected_frame, (new_width, new_height))
##################### END OF PATCH SELECTION SECTION #####################

##################### PATCH SAMPLING SECTION #####################
# Check if the target frame was found
if current_frame_number == target_frame_number:
    # Make a copy of the selected frame
    selected_frame_copy = selected_frame.copy()
    k = 0
    # Perform patch selection and
    while k != 27:
        # Display the selected frame
        cv2.imshow(win_select_patch, selected_frame)
        k = cv2.waitKey(30) & 0xFF
        if k == 99:
            selected_frame = selected_frame_copy.copy()
        # Perform action patch selection
        cv2.setMouseCallback(win_select_patch, colorPatchSelector)
        """Press ENTER to start the sampling"""
        c = cv2.waitKey(0) & 0xFF
        if c == 13:  # Check for Enter key press (ASCII 13)
            overall_mean_color = colorPatchSampling(patches_list)
            break
    cv2.destroyAllWindows()
cv2.imwrite("selected_frame.jpg", selected_frame)

print(f"Sampling finished. Overall mean color {overall_mean_color}")
##################### END OF PATCH SAMPLING SECTION #####################


##################### MASK CREATION SECTION #####################
print("Performing patch fitting...")
# Create window name
cv2.namedWindow(win_select_patch)
# Define upper and lower bounds
lower_bound = np.zeros(3).astype(int)
upper_bound = np.zeros(3).astype(int)
min_tol = 0
max_tol = 100

# Create trackbars
"""Here the HUE is taken as a cast Slider this control the amounnt of each color present in the image"""
cv2.createTrackbar(
    "H or CAST Tolerance +/-",
    win_select_patch,
    min_tol,
    max_tol,
    on_tolerance_change_HUE,
)
cv2.createTrackbar(
    "S Tolerance +/-",
    win_select_patch,
    min_tol,
    max_tol,
    on_tolerance_change_SATURIATION,
)
cv2.createTrackbar(
    "V Tolerance +/-", win_select_patch, min_tol, max_tol, on_tolerance_change_VALUE
)
cv2.createTrackbar("Softness", win_select_patch, min_tol, max_tol, blur_arguments)

# Init a dictionary to store the values to be used in the future for the video processing
postprocessing_values_dict = {}

# Initialize prev_softness_value
prev_softness_value = 0

# Reset c value
c = 0
selected_frame = cv2.imread("selected_frame.jpg")  # Replace with your image file

while k != 27:
    # Get the softness value from the trackbar
    softness_value = cv2.getTrackbarPos("Softness", win_select_patch)

    if softness_value > prev_softness_value:
        # Apply blurring to the image
        prev_softness_value = softness_value
        blurred_image = on_tolerance_change_BLUR(softness_value, image=selected_frame)
        selected_frame = blurred_image
    elif softness_value < prev_softness_value:
        # Deblur the image a restore it to the previous value
        prev_softness_value = softness_value
        deblurred_image = on_tolerance_change_DEBLUR(softness_value, selected_frame)

    mask = cv2.inRange(selected_frame, lower_bound, upper_bound)

    # Display patch and results
    cv2.imshow(win_select_patch, selected_frame)
    cv2.imshow(win_select_patch_mask, mask)
    k = cv2.waitKey(30) & 0xFF

    if k == 27:
        break

cv2.destroyAllWindows()

# Perform open and dilation
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

# Create an inverted mask
mask_inv = cv2.bitwise_not(mask)
print("Patch fitting finished")
##################### END OF MASK CREATION SECTION #####################

##################### BACKGROUND IMPLEMENTATION SECTION #####################
# We will perform bitwise operations in order to remove the background
# First we perform an and operation between the original image and inverted mask (background in black or 0) to
# eliminate the background and then an operation between the foreground and the mask to eliminate the other objects
img_bg = cv2.bitwise_and(selected_frame_copy, selected_frame, mask=mask_inv)
img_fg = cv2.bitwise_and(background_image, background_image, mask=mask)
# Segmenting of objects
result = cv2.add(img_bg, img_fg)
k = 0
while k != 27:
    cv2.imshow("img_bg", img_bg)
    cv2.imshow("img_fg", img_fg)
    cv2.imshow("mask_inv", mask_inv)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
##################### END OF BACKGROUND IMPLEMENTATION SECTION #####################


##################### MAIN LOOP #####################
# Call and create new variables to postprocess the video
k = 0
# Enter the main loop
while ret:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (new_width, new_height))

    # Apply gaussian blurring with the softness values applied before
    frame = apply_gaussian_blur(
        frame,
        sigma_x=softness_value,
        sigma_y=softness_value,
    )

    # We take the upper and lower values from bprevious steps to create the mask
    mask = cv2.inRange(frame, lower_bound, upper_bound)
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    # print(type(mask), mask.shape[0], mask.shape[1])
    # print(type(background_image), background_image.shape[0], background_image.shape[1])
    # Perform open and dilation
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Create an inverted mask
    mask_inv = cv2.bitwise_not(mask)

    # We will perform bitwise operations in order to remove the background
    # First we perform an and operation between the original image and inverted mask (background in black or 0) to
    # eliminate the background and then an operation between the foreground and the mask to eliminate the other objects
    img_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    img_fg = cv2.bitwise_and(background_image, background_image, mask=mask)

    # Segmentation of the objects
    frame = cv2.add(img_bg, img_fg)

    # Display the results
    cv2.imshow("Results", frame)

    k = cv2.waitKey(30)
    if k == 27 & 0xFF:
        break  # Exit the loop if the 'q' key is pressed

cap.release()
cv2.destroyAllWindows()
##################### END OF MAIN LOOP #####################
