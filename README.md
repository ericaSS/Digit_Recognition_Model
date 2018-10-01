# Digit_Recognition_Model
<h4> CCN (3 * 3) --> Relu --> CCN (3 * 3) --> Relu --> MaxPool(2X) (high-dimension) </h3>
<h4> CCN (3 * 3) --> Relu --> CCN (3 * 3) --> Relu --> MaxPool(2X) (low-dimension) </h3>
<h4> Fully Connected layer --> Fully Connected layer </h3>
<h4> Softmax </h3>
</br>
<h3>Some Details</h3>
<body>
Conv (Feature Extraction): to pick up the features from the image.Each convolutional neural network has 16 layers
</br>
Relu (Feature enhancement):  to convert the white area and black area of image, and make the feature remarkable
          Relu(x) = max(0, x)
</br>
MaxPool (dimension degradation): to minimize the size of image
</br>
Q 1: why the structure has 2 times of pooling not once?</br>
Since the FCs are only two, so if we just input original size of image 32 * 32  * 16(CNN)= 1024 - d, FCS cannot handle them very well, thus, I use two times of pooling, the image can be minimize from 32 * 32 -> 16 * 16 -> 8 * 8 = 64 * 16(CNN) - d, which is friendly to FC layer to train the classifier.
</br>
Q 2: why we cannot have more than 2 times pooling</br>
On the contrary, if there are more than 2 times of pooling, before the data input into FC, the feature set is too small, and it is hard for FC to precisely classify the output labels as well.
</br>
Q 3: why the pooling set after each two of CNNs; can we set the first pooling right after the third CNN?</br>
If we do that, it is possible that the first, second and third CNN they may learn similar features during the training that is too inefficient.  
</br>     
FC (classification): to train the classifier to classify the labels
</br>
Softmax: to calculate via loss function to check the accuracy of learning process.

