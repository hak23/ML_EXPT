# Multinomial Logistic Regression(Softmax).cross-entropy loss function
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# some initial setup

# describe a tensor "x" which is place holder. A value is assigned to it when we ask to tensorflow to run some computation. Here, we can input as 
# many images as needed. each of the image is a 784 length vector
x = tf.placeholder(tf.float32, [None, 784])

#setup some variables

w = tf.Variable(tf.zeros([784,10])) #weights iniialized to zero. We will learn them
b = tf.Variable(tf.zeros([10]))

#model. We use softmax as the activation.
y_predict = tf.nn.softmax(tf.matmul(x,w) + b)

# define the loss

# need a place holder for the one-hot vector of the actual values
y_true = tf.placeholder(tf.float32, [None, 10])

# we are using cross entropy loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_predict), reduction_indices=[1]))

# use gradient descent to optimze the loss. Tensor flow already has a gradientdescent based optimizer!!!
# Tensorflow automatically creates a graph of our problem and knows that it has to apply backpropagatioyn for finding the gradient with respect to weights(dc/dw)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess= tf.Session()
sess.run(init)

# run the optimizer 1000 times. Each time pick 100 random images from the set of training images. This is called stochastic training. Here batch size is 100
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_true: batch_ys})

# Evaluating the Model

# argmax basically gives the index of the highest entry in a particular dimension in a matrix
# verify if the highest entry in prediction matches the one in groundtruth
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_true,1))

print correct_prediction

# get the accuracy of the prediction: Run the model on all the test images and collect the value of correct_prediction for each of the image and average the value
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))
