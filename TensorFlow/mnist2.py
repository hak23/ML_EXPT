# Convolutional Neural Network (based on Deep MNIST Tensorflow.org)
# MNIST digit recognition with Convolitonal Neural Nets
# Also see ./mnist1.py
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()
# Place Holders for input and output
x        = tf.placeholder(tf.float32, shape=[None, 784])
y_actual = tf.placeholder(tf.float32, shape=[None, 10])

# Function for initializing weights
# In     : shape of the variable
# return : a noise initialized tensor with dimension as noted in shape
def weight_init(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# Function for initializing bias
# In     : shape of the varaible
# return : a constant tensor with dimension as specified in the shape
def bias_init(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Function that wraps TF conv2d
# In     : Image and Filter
# return : tf.nn.conv2d output. same as type as Image
def conv2d(x, W):
    return tf.nn.conv2d(x , W, strides=[1,1,1,1], padding='SAME')

# Function that wraps TF max_pool
# In     : Image
# return : max pooled output tensor
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# First Convolutional layer
# We will have 32 filters in the first layer. This will generate 32 outputs which will be input for the next
# convolutional layer

# reshape the image to a 28x28 with 1 color channel
x_image = tf.reshape(x, [-1,28,28,1])

# Initilaize the weights and bias for this layer
w_conv1 = weight_init([5,5,1,32])
b_conv1 = bias_init([32])

# Do the actual convolution, add bias, apply ReLU and then max pool
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional layer
# There are 32 input channels(from previous ouput) and we choose to have 64 output channels for 
# this layer. Weights will basically be 5x5 pathces with 32x64 dimension or in other words 
# a Tensor with shape [5,5,32,64]

w_conv2 = weight_init([5,5,32,64])
b_conv2 = bias_init([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer
# Now we have 64 7x7 patches. We do a fully connected network with 1024 neurons.

# reshape the previous layer output into a vector. [7,7,64] -> [7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])


w_fc1 = weight_init([7*7*64, 1024])
b_fc1 = bias_init([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# Understand drop out and use it here. Is it same as regularization?

# Output layer has 10 neurons corresponding to 10 digits. Add softmax in the end to 
# convert to probabilities
w_fc2 = weight_init([1024, 10])
b_fc2 = bias_init([10])

y_pred = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)

# Use cross entropy as the loss funciton
l_cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual * tf.log(y_pred), reduction_indices=[1]))

# We have built the tensorflow graph above, Now we need to run it

# Using Adam Optimizer as a black box. Look for the details.
train_step = tf.train.AdamOptimizer(1e-4).minimize(l_cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_actual,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    # TODO: Monitor accuracy regularly
    train_step.run(feed_dict = {x:batch[0], y_actual:batch[1]})

# Accuracy on test set
test_accuracy = accuracy.eval(feed_dict={x:mnist.test.images, y_actual:mnist.test.labels})
print("Accuracy on test set: %f", test_accuracy)




























