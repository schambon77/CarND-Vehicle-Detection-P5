# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/17.png "Car image"
[image2]: ./output_images/converted_17.png "Car image converted to YCrCb color space"
[image3]: ./output_images/hog_17.png "HOG features extracted from car image for first channel"
[image4]: ./output_images/extra40.png "Non car image"
[image5]: ./output_images/converted_extra40.png "Non car image converted to YCrCb color space"
[image6]: ./output_images/hog_extra40.png "HOG features extracted from non car image for first channel"
[image7]: ./output_images/spatial_17.png "Spatial features extracted from the car image"
[image8]: ./output_images/color_17.png "Color features extracted from the car image"
[image9]: ./output_images/spatial_extra40.png "Spatial features extracted from the non car image"
[image10]: ./output_images/color_extra40.png "Color features extracted from the non car image"
[image11]: ./test_images/test1.jpg "Test image"
[image12]: ./output_images/boxes_test1.jpg "Sliding window applied on test image"
[image13]: ./output_images/final_test1.jpg "Final car detection on test image"
[image1]: ./examples/car_not_car.png
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

All source code references are made on my Jupyter notebook [VehicleDetection-P5.ipynb](https://github.com/schambon77/CarND-Vehicle-Detection-P5/blob/master/VehicleDetection-P5.ipynb).

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The generic `get_hog_features` function allows the HOG features extraction from an input image according to the orientation, number of pixels per cell and number of cells per block parameters.

The code cell under the title "Extract features" contains the calls to the HOG feature extraction function.

The parameters I used are:
* Image color space: `YCrCb`
* orientation: `9`
* pixels per cell: `8`
* cells per block: `1`

Here is an example of one of each of the `vehicle` and `non-vehicle` classes, with respective conversion to YCrCb color space and extracted HOG features:

![image1]
![image2]
![image3]

![image4]
![image5]
![image6]

#### 2. Explain how you settled on your final choice of HOG parameters.

I chose the `YCrCb` color space as this was giving me the best accuracy results after several experimentations on different color spaces.

I kept the default values for orientation and pixels per cell.

However I reduced cells per block from `2` to `1` in order to reduce the overall feature vector length, and so to minimize possible over-fitting.

Finally, I extracted HOG features for all 3 channels in order to get maximum value of the car (or non car object) shapes in the different channels. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the "Extract features" code cell, I read all vehicles and non-vehicles images. For each I then call the function `single_img_features` which conveniently extracts the HOG features through the function previously mentioned, as well as spatial and color features.

The parameters I used for the spatial features extraction through function `bin_spatial` are:
* spatial_size: (8, 8)

The parameters I used for the color features extraction through function `color_hist` are:
* bins: 16
* range: (0, 256)

The parameters lower than the defaults are chosen in order to keep a reasonnable dimensionality of the feature vector.

Here are visual examples of the 2 feature vectors extracted:

![image1]
![image7]
![image8]

![image4]
![image9]
![image10]

The "Train classifier" code cell contains code to train an SVC classifier, with the default 'rbf' method, but a large C (1e8) and low gamma (4e-4) in order to improve accuracy.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The function `process_image` contains the code for the pipeline that applies a sliding window search and runs the trained classifier to detect cars in an image. This function eventually calls the `slide_window` function, which creates all search windows to apply, and the `search_windows` which extracts the features from each search window.

A full search is applied at start when no prior information is known from the image. I have used a wide number of scales to be sure to detect cars: 32, 64, 96, 128, 160, 192, 224

For the smaller sizes, I restricted the x / y start and stop positions in order to avoid too much processing time, given that small cars are usually detected further away closer to horizon and towards the center of the image. The search window is widened as the image size increases.

False positives are eliminated by computing a heatmap from all the positive results of the classifier, and then applying a threshold (set to 1) on the heatmap in order to eliminate areas where 1 single window was found erroneously as a match. The resulting superimposed matched windows define a minimum box which is drawn on the original image. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here is an example of the detection of cars through sliding window on a test image:

![image11]

![image12]

![image13]

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/detected_vehicles_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

For the video, I introduced some tracking of detected cars. This started by the implementation of the `Car` class, which contains a queue (max length 10) for all final ovelapping windows found, and can return an average of the overlapping windows. This is done to smooth the bounding window drawing around the detected cars.

The function `process_image_zones` was added in order to handle 2 types of search:
* at start, and every 10 frames, a full searchas described before is performed. The resulting bounding windows are added to already detected cars based on the max overlap, or a new Car object is created if no overlap with existing car was found.
* for other frames, a much reduced search around each known Car location is performed within a margin (set to 25 pixels) in each direction. This speeds up the video processing, and reduces the risk to get false positives as the search space is reduced.

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here are a few discussion points I'd like to bring up:
* The main challenge I faced in this problem was the classifier training. This is mainly due to the fact that my feature vector was very long (over 8000 features), and my classifier was over-fitting. Once I reduced the feature vector length, detection of cars was much more accurate.
* I would say however that the classifier can be improved by added training samples from other data sources.
* The full search space can be further optimized. This was compensated by some search optimization on 9 frames out of 10 based on previously detected cars.
