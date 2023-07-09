import numpy as np
import matplotlib.pyplot as plt
import struct


def load_mnist(labels_path, images_path):
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights1 = np.random.randn(input_size, hidden_size)
        print(f"Number of weight1: {str(np.shape(self.weights1))}")
        self.bias1 = np.zeros(hidden_size)
        print(f"Number of bias1: {np.shape(self.bias1)}")
        self.weights2 = np.random.randn(hidden_size, output_size)
        print(f"Number of weight2: {str(np.shape(self.weights2))}")
        self.bias2 = np.zeros(output_size)
        print(f"Number of bias2: {str(np.shape(self.bias2))}")

    def sigmoid(self, x):
        # Activation function
        # Clip x to avoid overflow in exp
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))


    def sigmoid_derivative(self, x):
        # Derivative of sigmoid function for backpropagation
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward(self, x):
        # Forward propagation
        self.z1 = np.dot(x, self.weights1) + self.bias1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, x, y, output):
        # Backward propagation
        self.dz2 = output - y
        self.dw2 = np.dot(self.a1.T, self.dz2)
        self.db2 = np.sum(self.dz2, axis=0)
        self.dz1 = np.dot(self.dz2, self.weights2.T) * self.sigmoid_derivative(self.z1)
        self.dw1 = np.dot(x.T, self.dz1)
        self.db1 = np.sum(self.dz1, axis=0)

    def update_weights_biases(self, lr):
        # Update weights and biases
        self.weights1 -= lr * self.dw1
        self.bias1 -= lr * self.db1
        self.weights2 -= lr * self.dw2
        self.bias2 -= lr * self.db2

    def compute_loss(self, y, output):
        # Compute the loss
        # Clip output to avoid log(0)
        output_clipped = np.clip(output, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(output_clipped)) / y.shape[0]

    def compute_accuracy(self, X, y):
        # Compute the accuracy
        output = self.forward(X)
        predictions = np.argmax(output, axis=1)
        ground_truth = np.argmax(y, axis=1)
        return np.mean(predictions == ground_truth)

    def train(self, X, y, epochs, learning_rate, batch_size=32):
        # Training loop
        history_accuracy = []  # store accuracy for each epoch
        for i in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for j in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[j:j+batch_size]
                y_batch = y_shuffled[j:j+batch_size]

                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)
                self.update_weights_biases(learning_rate)

            output = self.forward(X)
            loss = self.compute_loss(y, output)
            accuracy = self.compute_accuracy(X, y)
            history_accuracy.append(accuracy)  # save accuracy for this epoch

            if i % 10 == 0:
                print(f"Epoch: {i}, Loss: {loss}, Accuracy: {accuracy}")
        
        # plot the accuracy history
        plt.plot(history_accuracy)
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.savefig('accuracy.png')  # save the accuracy graph

    def predict(self, x):
        # Predict function
        output = self.forward(x)
        predictions = np.argmax(output, axis=1)
        return predictions

if __name__ == "__main__":
    # Load data
    labels_file_path = "t10k-labels.idx1-ubyte"
    images_file_path = "t10k-images.idx3-ubyte"
    X, y = load_mnist(labels_file_path, images_file_path)

    # Normalize the input data
    X = X / 255.0

    # One hot encode the labels
    y = np.eye(10)[y]

    # Initialize the neural network
    nn = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)

    # Split data into train and test sets
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    split_idx = int(0.75 * X.shape[0])  # 75% for training, adjust this to fit your needs

    X_train = X[indices[:split_idx]]
    y_train = y[indices[:split_idx]]

    X_test = X[indices[split_idx:]]
    y_test = y[indices[split_idx:]]

    # Train the model with training set
    nn.train(X_train, y_train, epochs=100, learning_rate=0.001)

    # Test the model with testing set and print the accuracy
    accuracy = nn.compute_accuracy(X_test, y_test)
    print(f"Test Accuracy: {accuracy}")
