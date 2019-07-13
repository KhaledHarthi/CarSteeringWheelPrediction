# Steering Wheel Predicition using Deep Learning
<p align="center"><img src="pics/pr_logo.png"/></p>

This project aims to build a Convolutional Neural Network model that will be able to predict the angle of the steering wheel from road images.
<p align="center"><img src="pics/visualization.gif"/></p>

## Model Architecture

I’ve tried to design the model architecture depending on the nature of the data and the problem itself, and since the data was a video footage, I thought about it as a sequence of images, and what suits this type of data is <b>Recurrent Neural Networks (RNN)</b>, thus I’ve constructed a <b>CNN-LSTM</b> model.

![](pics/architecture.png)

A <b>CNN-LSTM</b> model consists of two models. First, the CNN model which takes the input images and perform <b>feature extraction</b>. Second, comes the RNN model which takes the feature map sequences produced by CNN and learn the patterns, hopefully this part will learn the dynamics of driving a car.
<br><br>
The the input was splitted into a sequence of 15 consecutive images, the sequence at time t is the set of images {t, t-1, t-2 ... t-14}. The sequences was constructed first then shuffled since the data has a high correlation betIen adjacent samples, shuffling the training data was mandatory. The LSTM will see a sequence of 10 samples, since the other 5 will be eaten by the convolutional layers.
<br><br>
During building this model, I’ve tried so many tricks such as <b>*residual connections*</b> betIen convolutional layers which boosted the performance and made the layers avoid the problem of vanishing gradients. I’ve used ReLU non-linear activation function to speed up the training process. No regularization Ire used but dropout with keep probability of 25% betIen every convolutional layer and fully-connected layers except the last one. Batch normalization was used after all the convolutional layers. I’ve noticed that after using max-pooling rather than strides the model was able to achieve a loIr MSE by 11%.
<br><br>
In the training process I’ve optimized the model for <b>100 epochs</b>, and used checkpoints to save the model that has the least test error, also I’ve loIred Adam optimizer’s learning rate to <b>*0.0005*</b> which gave us better performance, and since the data can’t fit in memory I’ve used mini-batch gradient descent with 20 batch size.

# Results & Evaluation
The model was able to achieve an <b>*RMSE*</b> of <b>*0.0678*</b>, and was able to generalize Ill the unseen data, and was surprisingly able to predict the angle of steep turns with only a few examples. I’ve plotted the predicted and the measured angles (ground truth), and as you can see the model was able to predict steep turns (large angles) almost perfectly.
<br>
<br>
<p align="center"><img src="pics/measure_predicted.png"/></p>
<p align="center"><img src="pics/measure_predicted_scatter.png"/></p>
I can see that the predicted values and the measured values have high correlation, which indicates that the model was able to generalize Ill.

# Future Work
I believe that further tuning will definitely give better result, and the model will benefit from additional data, one method for providing additional data is to use <b>GAN</b> models to generate more images of steep turns, and winter/rainy Iather from the existing summer Iather.