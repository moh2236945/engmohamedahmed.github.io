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


