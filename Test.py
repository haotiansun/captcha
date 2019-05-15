import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# index = 100
# plt.imshow(X_train_orig[index])
# print ("y = " + chr(np.squeeze(Y_train_orig[:, index])))
# print(Y_train_orig[:,index])
# plt.show()
# print ("success")



# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
#print(Y_train_orig)
Y_train = convert_to_one_hot(Y_train_orig, 62)
Y_test = convert_to_one_hot(Y_test_orig, 62)

print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

#print (Y_train)


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=[n_x, None])
    Y = tf.placeholder(tf.float32, shape=[n_y, None])
    return X, Y


def initialize_parameters():
    tf.set_random_seed(1)  # so that your "random" numbers match ours
    W1 = tf.get_variable("W1", [80, 1024], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [80, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [60, 80], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [60, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [30, 60], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [30, 1], initializer=tf.zeros_initializer())

    W4 = tf.get_variable("W4", [20, 30], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b4 = tf.get_variable("b4", [20, 1], initializer=tf.zeros_initializer())
    W5 = tf.get_variable("W5", [10, 20], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b5 = tf.get_variable("b5", [10, 1], initializer=tf.zeros_initializer())
    W6 = tf.get_variable("W6", [62, 10], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b6 = tf.get_variable("b6", [62, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  "W6": W6,
                  "b6": b6
                  }
    return parameters

# Forward Propagation in tensorFlow
def forward_propagation(X, parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']
    W6 = parameters['W6']
    b6 = parameters['b6']

    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3

    A3 = tf.nn.sigmoid(Z3)  # A2 = relu(Z2)
    Z4 = tf.add(tf.matmul(W4, A3), b4)  # Z3 = np.dot(W3,Z2) + b3

    A4 = tf.nn.sigmoid(Z4)  # A2 = relu(Z2)
    Z5 = tf.add(tf.matmul(W5, A4), b5)  # Z3 = np.dot(W3,Z2) + b3

    A5 = tf.nn.relu(Z5)
    Z6 = tf.add(tf.matmul(W6, A5), b6) #output layer

    return Z6


def compute_cost(Z6, Y):
   # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z6)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=64, print_cost=True):
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z6 = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z6, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)
        foo = open("result_epoch1500_modified_2211.txt", "w")
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.  # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            #batch_num = 0
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                #print("batch_num: ", batch_num)
                #batch_num = batch_num + 1
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                foo.write(str(epoch_cost) + "\n")
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z6), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        foo.write("train accuracy: " + str(accuracy.eval({X: X_train, Y: Y_train})) + "\n")
        foo.write("test accuracy: " + str(accuracy.eval({X: X_test, Y: Y_test})) + "\n")
        foo.close()
        return parameters
parameters = model(X_train, Y_train, X_test, Y_test)
