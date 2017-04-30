**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

###Histogram of Oriented Gradients (HOG)

####1. I extracted HOG features from the training images, the code for this step is contained in the submitted IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt tag](https://github.com/Martijnde/SDC-Project5-CarND-Vehicle-Detection/blob/master/car_and_noncar.png?raw=true)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt tag](https://github.com/Martijnde/SDC-Project5-CarND-Vehicle-Detection/blob/master/hog1.png?raw=true)

![alt tag](https://github.com/Martijnde/SDC-Project5-CarND-Vehicle-Detection/blob/master/hog2.png?raw=true)

The code for this step is contained in the submitted IPython notebook.

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and this way I found the best settings to train the classifier well. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the settings below:

color_space:  RGB
orient:  9
pix_per_cell:  8
cell_per_block:  2
hog_channel:  0
spatial_size:  (32, 32)
hist_bins:  16
spatial_feat:  True
hist_feat:  True
hog_feat:  True
Using spatial binning of: (32, 32) and 16 histogram bins

The code for this step is contained in the submitted IPython notebook.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search the window positions over the image and came up with this ideal settings:

wndws_v1 = slide_window(image, x_start_stop=[0, 1300], y_start_stop=[375, 650],
                             xy_window=(64, 64), xy_overlap=(0.6, 0.6), window_list=None)
wndws_v2 = slide_window(image, x_start_stop=[0, 1280], y_start_stop=[450, 650],
                             xy_window=(120, 120), xy_overlap=(0.6, 0.6),window_list=None)

The window could be found in the image below:

![alt tag](https://github.com/Martijnde/SDC-Project5-CarND-Vehicle-Detection/blob/master/slide_window.png?raw=true)

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to try to minimize false positives and reliably detect cars?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt tag](https://github.com/Martijnde/SDC-Project5-CarND-Vehicle-Detection/blob/master/test_flow_on_images_new.png?raw=true)
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a link to my adjusted output video result on youtube: 

https://youtu.be/-tH9PyQYvJ0


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To be more robust for videos I've added some techniques to detect vehicles in subsequent frames. A heatmap in the functions (get_hot_windows and detect_vehicles respectively) is added to show the location of repeated vehicle detections. The previous results are stored in a custom class Memory, that stores values of the previous frame. This technique was used to reduce the number of false positives, to combine multiple overlapping boundig boxes I've used a function to combine all bounding boxes. the code can be found in the submitted notebook.

### Here is an example of a original image frame, it's corresponding heatmap and the boxes drawed on the output image:

![alt tag](https://github.com/Martijnde/SDC-Project5-CarND-Vehicle-Detection/blob/master/test_flow_on_images_new.png?raw=true)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The implementation of the project was quite hard thus I find the Computer Vision parts of the project kinda hard, I also was a bit confused about the multiple classifiers I used, but they seemed to do not work apart from eachother.

The pipeline might fail when new types of cars are on the road and the way I might improve it if I were going to pursue this project further by using deeplearning to keep training the model for the best and most updated accuracy possible.  

