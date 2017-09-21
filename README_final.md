**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

## Load the data set (see below for links to the project data set)
###Loading the data was pretty straight forward. The only important detail was to load, besides the images and the label, the zoom information in order to use it later as data augmentation.
## Explore, summarize and visualize the data set
###Looking at the data set, we can easily see that image count per sign type is very unheaven. In fact, some signs are represented over 10 times more than others. The total number of images is around 35k.
## Design, train and test a model architecture
###In order to feed the model, the first step was to pre-process the images. After the normalization, although I have tried several approaches, I finally decides to simply use a grayscale convertion, in order to focus on finetuning the model parameters. For instance, I realized that the histogram equalization was having a strange effect like removing a lot of information, and having an effect like a dropout, but in a way not as random as the dropout, and that affected the results. For instance a speed limit sign could totally loose the number.
Next step was the data augmentation. I did it in two steps. Since I had the right zoom information from the data set, I simply duplicated the amount of images using that information. Then, in order to level the number of images per sign types, I randomly applied rotation, translation, distortion and brightness adjustment, for each sign type until each one reached 2020 (a little bit over twice the counting of the most represented sign). That created a total number of images of around 175k.
Finally, starting from the LeNet model, I ran several simulations until I managed to find a path to the required test accuracy of 93%. The final model has one more convolution (3x3) than the LeNet, more filters, and two dropouts after fully connected layers (0 and 1).
## Use the model to make predictions on new images
###In order to be as realistic as it could be, I used goolgle earth street view from german roads (around Cologne and in front of the train station at Frankfurt). The signs have all been correctly detected.
## Analyze the softmax probabilities of the new images
###The detected signs have been with a much higher probability (between 60 and 80%) than the remaining options for any of the signs.
## Summarize the results with a written report
###At the end, after a long testing, I managed to reach the 93% accuracy for the test set. All those tests have shown that tiny differences might cause big differences on the output. A mistake that I later realized, is that I tried to go to fast, changing more than a variable at a time (although it did seem logical at that time). I would say that the best approach is to keep it as simple as possible, and then step by step to chase a huge training accuracy. That huge value, once found, with the help of dropouts, can be shared with the validation allowing a suitable value based on the project requirements. With the right balancing of all those parameters, the 93% value on the test evaluation have finally been conquered!
---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
* The size of the validation set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 32x32x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				    |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 3x3x64 	|
| RELU					|												|
| Fully connected		| outputs 576        							|
| Droptout				| 0.7											|
| Fully connected		| outputs 256        							|
| RELU					|												|
| Droptout				| 0.7											|
| Fully connected		| outputs 84        							|
| RELU					|												|
| Fully connected		| outputs 43        							|
|:---------------------:|:---------------------------------------------:| 
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used I made some test to identify an acceptable learning rate, using the batch size of 128 already coming from the LeNet project. I also maintained the Adam Optimizer. I then started adding filters to the existing layers and started seeing some good improvements. Then I added a third convolution layer. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem. 

Since the results were getting quite near that objective, I increased epochs from 20 to 200. And that was it.
Obviously, it is much faster to explain than to figure out what it the right way.
To explain how I managed to reach the result In two words, I would say: more filters.

My final model results were:
* training set accuracy of 98%
* validation set accuracy of 96% 
* test set accuracy of 93%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? LeNet simply because it was the one mentioned by the project.
* What were some problems with the initial architecture? I was very difficult to go over 90% of accuracy, even for the validation set.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? Much more than the 93%, it is the 100% on the signs colected from german roads using the Google Earth street view.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right      			| Right   										| 
| No entry     			| No entry 										|
| Priority 				| Priority 										|
| Front		      		| Front					 						|
| Yield					| Yield 		     							|
| Front	Right			| Front	Right	     							|

The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

