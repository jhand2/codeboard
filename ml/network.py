#!/usr/bin/python3
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
    from data_utils import *


class Network:
    def __init__(self, labels):
        # TODO: Regularization: dropout

        self.labels = labels
        self.batch_size = 256
        self.image_size = 45 ** 2
        self.layers = [
            (128, "relu"),
            (256, "relu"),
            (256, "relu"),
            (256, "relu"),
            (128, "relu"),
            (len(labels), None)
        ]

        # set up neural net
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.global_step = tf.Variable(0)
            self.learning_rate = tf.train.exponential_decay(0.0001, self.global_step, self.batch_size, 0.95) 
            self.weights = {}
            self.biases = {}

            self.tf_train_dataset = tf.placeholder(tf.float32,
                                                   shape=(self.batch_size, self.image_size),
                                                   name="tf_train_dataset")
            self.tf_train_labels = tf.placeholder(tf.float32,
                                                  shape=(self.batch_size, len(self.labels)),
                                                  name="tf_train_labels")

            # Initialize weights and biases
            insize = self.image_size
            for i in range(len(self.layers)):
                outsize = self.layers[i][0]
                self.weights[i] = tf.Variable(tf.truncated_normal([insize, outsize], stddev=(2.0/outsize)))
                self.biases[i] = tf.Variable(tf.zeros([outsize]))
                insize = outsize

            self.session = tf.Session(graph=self.graph)

    def forward_prop(self, X):
        inlayer = X
        for i in range(len(self.layers)):
            layer = tf.add(tf.matmul(inlayer, self.weights[i]), self.biases[i])
            fname = self.layers[i][1]
            if fname == "relu":
                layer = tf.nn.relu(layer)
            elif fname == "softmax":
                layer = tf.nn.softmax(layer)
            inlayer = layer

        return inlayer
    
        
    def accuracy(self, predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
              / predictions.shape[0])
        

    def train(self, X, Y, valid_X=None, valid_Y=None, test_X=None, test_Y=None):
        with self.graph.as_default():
            logits = self.forward_prop(self.tf_train_dataset)

            nonreg_loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.tf_train_labels, logits=logits))
            l2_regularization_penalty = 0.01
            l2_loss = 0
            for i in range(len(self.layers)):
                l2_loss += (l2_regularization_penalty * tf.nn.l2_loss(self.weights[i]))

            loss = nonreg_loss + l2_loss

            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
            train_prediction = tf.nn.softmax(logits)

            if valid_X is not None:
                tf_valid_dataset = tf.constant(valid_X)
                valid_prediction = tf.nn.softmax(self.forward_prop(tf_valid_dataset))

            if test_X is not None:
                tf_test_dataset = tf.constant(test_X)
                test_prediction = tf.nn.softmax(self.forward_prop(tf_test_dataset))
            self.session.run(tf.global_variables_initializer())

        num_steps = 8001

        with self.session.as_default():
            for step in range(num_steps):
                # Pick an offset within the training data, which has been randomized.
                # Note: we could use better randomization across epochs.
                offset = (step * self.batch_size) % (Y.shape[0] - self.batch_size)

                # Generate a minibatch.
                batch_data = X[offset:(offset + self.batch_size), :]
                batch_labels = Y[offset:(offset + self.batch_size), :]

                # Prepare a dictionary telling the session where to feed the minibatch.
                # The key of the dictionary is the placeholder node of the graph to be fed,
                # and the value is the numpy array to feed to it.
                feed_dict = {"tf_train_dataset:0": batch_data, "tf_train_labels:0": batch_labels}

                _, l, predictions = self.session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

                if (step % 500 == 0):
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print("Minibatch accuracy: %.1f%%" % self.accuracy(predictions, batch_labels))
                    if valid_X is not None:
                        print("Validation accuracy: %.1f%%" % self.accuracy(
                              valid_prediction.eval(), valid_Y))
            if test_X is not None:
                print("Test accuracy: %.1f%%" % self.accuracy(test_prediction.eval(), test_Y))

    def predict(self, X):
        pass

if __name__ == "__main__":
    data_cache = load_all_data()
    dataset_X = data_cache["dataset_X"]
    dataset_Y = data_cache["dataset_Y"]
    print(dataset_Y.shape)
    labels = data_cache["labels"]

    # Shuffle data and divide it into train, test, and validate
    sets = divide_data(dataset_X, dataset_Y)

    train_X = sets["train_X"].astype(np.float32)
    train_Y = sets["train_Y"].astype(np.float32)

    test_X = sets["test_X"].astype(np.float32)
    test_Y = sets["test_Y"].astype(np.float32)

    valid_X = sets["valid_X"].astype(np.float32)
    valid_Y = sets["valid_Y"].astype(np.float32)

    print("Training set shape:", train_X.shape)
    print("Training labels shape:", train_Y.shape)

    print("Test set shape:", test_X.shape)
    print("Validation set shape:", valid_X.shape)

    nn = Network(labels)
    nn.train(train_X, train_Y, valid_X, valid_Y, test_X, test_Y)
    nn.session.close()

