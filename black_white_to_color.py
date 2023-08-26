import numpy as np
import cv2

# Load the pre-trained colorization model
print("Loading models...")
net = cv2.dnn.readNetFromCaffe('colorization_deploy_v2.prototxt', 'colorization_release_v2.caffemodel')
pts = np.load('pts_in_hull.npy')

# Configure model layers and color points
class8_layer_id = net.getLayerId("class8_ab")
conv8_layer_id = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8_layer_id).blobs = [pts.astype("float32")]
net.getLayer(conv8_layer_id).blobs = [np.full([1, 313], 2.606, dtype='float32')]

# Load the input grayscale image
input_image = cv2.imread('rose.jpg')
scaled_image = input_image.astype("float32") / 255.0

# Convert the image to Lab color space
lab_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2LAB)

# Resize and preprocess the Lab image
resized_lab = cv2.resize(lab_image, (224, 224))
L_channel = cv2.split(resized_lab)[0]
L_channel -= 50

# Set the L channel as input to the colorization model
net.setInput(cv2.dnn.blobFromImage(L_channel))

# Run the colorization model to predict 'ab' channels
ab_channels = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab_channels = cv2.resize(ab_channels, (input_image.shape[1], input_image.shape[0]))

# Retrieve the original L channel from the Lab image
original_L_channel = cv2.split(lab_image)[0]

# Combine the L channel and predicted 'ab' channels for colorized Lab image
colorized_lab = np.concatenate((original_L_channel[:, :, np.newaxis], ab_channels), axis=2)

# Convert the colorized Lab image back to RGB color space
colorized_rgb = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)

# Clip pixel values and scale to 8-bit range
colorized_rgb = np.clip(colorized_rgb, 0, 1)
colorized_rgb = (255 * colorized_rgb).astype("uint8")

# Display the original grayscale image
display_height = 540
display_width = int(input_image.shape[1] * (display_height / input_image.shape[0]))
input_image = cv2.resize(input_image, (display_width, display_height))
cv2.imshow("Original Grayscale", input_image)

# Resize and display the colorized image for consistent visualization
display_height = 540
display_width = int(input_image.shape[1] * (display_height / input_image.shape[0]))
colorized_resized = cv2.resize(colorized_rgb, (display_width, display_height))
cv2.imshow("Colorized Image", colorized_resized)

# Wait for a key press and then close the display windows
cv2.waitKey(0)
cv2.destroyAllWindows()
