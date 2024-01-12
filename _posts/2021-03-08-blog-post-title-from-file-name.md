## Turorial 1

**Part1: Convolution**

1-Convolution

2-Padding

2-1-Type of Padding

2-1-1-Vaild padding

2-1-2-Same Padding

2-1-3-Full padding

2-2-Advantage of padding

2-3-Realtionship between padding and translation invariance

3-Strise

4-Advantage of Convolution in image processing

5-Convolution Vs cross-Correlation

6-Various method of calculation convolution

6-1-Sliding window

6-2-im2col

6-3-FFT

6-4-Winograd

7-Fully connected Layer

7.1-Batch Normalization For Fully connected Layer

7.2 Different between convolution layer and convolution layer

7.3 Disadvantages o FC

8-Notes

9-Glossary

10-Pytorch Code

11-convolution operation in numpy

----
**convolution**: the process of sliding filter on image (dot product(matrix multiplication)).

a type of matrix operation, consisting of a kernel, a small matrix of weights, that slides over input data performing element-wise multiplication with the part of the input it is on, then summing the results into an output.

**filter**: a grid of a discrete number(Kernels together) The filter has the same depth as input 

**convolution layer**: extract features from image and preserve the relationship between pixels

**feature maps** the output of convolutional layer after the convolution operation has occurred on an image 

**depth**: number of filter used(output channel)

**stride**: the size of convolution layer output .large size of stride meaning no over lap 

**receptive filed:** the part of image the visible to the filter 

**type of features:** Key point  features(specific location in image )	edges

**features detector:** take image and output locations(pixel coordinate)

**features descriptor** take an image and output feature vector

**A kernel** is a matrix of weights which are multiplied with the input to extract features

**A filter** a concatenation of multiple kernels, each kernel assigned to a particular channel of the input. Filters are always one dimension more than the  kernels.

( A filter: A collection of kernels)

**kernel size** is the dimensions of the kernel (H,W) 

**number of filters** refers to the number of output channels created after the kernel is convolved over the input image.

--------------
**1-Convolution**

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/conv1.gif)

---
This kernel “slides” over the 2D input data, performing an elementwise multiplication with the part of the input it is 

currently on, and then summing up the results into a single output pixel.

The kernel repeats this process for every location it slides over, converting a 2D matrix of features into yet another 2D 

matrix of features

Each kernel of the filter "slides" over its respective input channel, producing individual processed versions. 

Some kernels may have greater weight than others to emphasize certain input channels.

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/conv2.gif)

Each core of the filter produces its own processed version which is then summed so that the entire filter produces an overall output channel.

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/conv3.gif)

Finally, there is a term: "bias". Each output filter has a bias term, and the bias is added to the output channel to produce the final output channel.

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/conv4.gif)

Bias in a neural network is required to shift the activation function across the plane either towards the left or the right. 

**Difference between weight and bias in a convolutional neural network (CNN)**

**Weights** refer to the parameters that are learned by our CNN model during training. These weights tell us how much importance each feature has when making a prediction from  data. A higher weight means higher importance, while lower weights mean lesser importance for a specific feature. 

In other words, weights represent the strength of connection between different neurons of the model and helps capture relationships between data points as they pass through 

layers of filters and nodes within the network.

**Bias** 

works alongside weights to make predictions more accurate or add additional information to them beyond what just using weights alone can do. Unlike Weights, Bias 

values remain constant throughout training since they do not change with respect to any given input vector or example image you may input into your model’s system when 

running it through its training algorithm simulations. It essentially introduces an offset or “bias” value which helps shift a model's predicted outcome either up or down 

depending on whether this bias value is positive or negative respectively - aiding an overall better prediction accuracy from your trained models output given its provided 

inputs at test time.

**2-Padding**

refers to edge filling of the input during the convolution operation to control the size and shape of the output feature map. Common padding methods include:

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/padd1.gif)

**2-1.1 Valid padding**

Without padding, the output size is:  (W - F + 1) x (H - F + 1)

Where W is the input width, H is the input height, and F is the convolution kernel size.

**2-1.2. Same padding**

Also known as Zero padding, it will evenly fill the edges of the input image with zeros so that the size of the output feature map is the same as the input.

The output size is: W x H

**2-1.3. Full padding**

Also called **Full convolution**, F-1 circles 0 are filled in the entire input image boundary.

The output size is: (W + F - 1) x (H + F - 1)

**2-2 The main functions of PADDING:**

- Control the size and shape of the output feature map
  
- Maintain image edge information to avoid loss
  
- Increase network depth while keeping the receptive field unchanged
  
Care needs to be taken to avoid excessive padding leading to overfitting. 

Usually Same padding is used the most, which can maintain the output size without being too complicated.

**2-3 what is the relationship between padding and translation invariance?**

Without padding, the output of convolution does not seem to reflect the position very well:




