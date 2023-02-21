import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os.path


class DigitClassification:

    def __init__(self, lr, epochs):
        self.images, self.labels = self.get_mnist()

        if not os.path.isfile("save/weights_i_h.npy"):
            """
            w = weights, b = bias, i = input, h = hidden, o = output, l = label
            e.g. w_i_h = weights from input layer to hidden layer
            """

            self.w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))  # Generates a Matrix (784 by 20) with random values between .5 and -.5
            self.w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
            self.b_i_h = np.zeros((20, 1))  # Matrix 1 by 20 with zeros
            self.b_h_o = np.zeros((10, 1))

        else:

            self.w_i_h = np.load("save/weights_i_h.npy")
            self.w_h_o = np.load("save/weights_h_o.npy")
            self.b_i_h = np.load("save/biases_i_h.npy")
            self.b_h_o = np.load("save/biases_h_o.npy")

        self.training(int(lr), int(epochs))
        self.run()

    def get_mnist(self):
        with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
            images, labels = f["x_train"], f["y_train"]
        images = images.astype("float32") / 255
        images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
        labels = np.eye(10)[labels]
        return images, labels

    def save_nn(self):
        np.save('save/weights_i_h.npy', self.w_i_h)
        np.save('save/weights_h_o.npy', self.w_h_o)
        np.save('save/biases_i_h.npy', self.b_i_h)
        np.save('save/biases_h_o.npy', self.b_h_o)

    def training(self, lr, epochs):
        learn_rate = lr# 0.01
        nr_correct = 0

        for epoch in range(epochs):
            for img, l in zip(self.images, self.labels):
                img.shape += (1,) #converts Vector to Matrix (784 by 1)
                l.shape += (1,) #converts Vector to Matrix (10, 1)
                # Forward propagation input -> hidden
                h_pre = self.b_i_h + self.w_i_h @ img #Calculate Neuron Value with input weights and biases
                h = 1 / (1 + np.exp(-h_pre)) #Activation function to adjust Neuron value between 0 and 1
                # Forward propagation hidden -> output
                o_pre = self.b_h_o + self.w_h_o @ h #same as before
                o = 1 / (1 + np.exp(-o_pre)) #same as before

                # Cost / Error calculation
                e = 1 / len(o) * np.sum((o - l) ** 2, axis=0) #calculate error of each Neuron in the output layer
                """
                that works as follows
                Neuron Value is between 0 and 1
                label is in binary meaning a 3 is 0 0 0 1 0 0 0 0 0 0
                now lets say the third Neuron had a value of .2 
                we calculate .2 - 0 
                we square
                .04
                calculate the sum of all of this for each neuron in the o layer
                multiply it by 1 / Neurons in o-layer
                we get the error
                """
                nr_correct += int(np.argmax(o) == np.argmax(l))

                # Backpropagation output -> hidden (cost function derivative)
                delta_o = o - l
                self.w_h_o += -learn_rate * delta_o @ np.transpose(h)
                self.b_h_o += -learn_rate * delta_o
                # Backpropagation hidden -> input (activation function derivative)
                delta_h = np.transpose(self.w_h_o) @ delta_o * (h * (1 - h))
                self.w_i_h += -learn_rate * delta_h @ np.transpose(img)
                self.b_i_h += -learn_rate * delta_h

            # Show accuracy for this epoch
            print(f"Acc: {round((nr_correct / self.images.shape[0]) * 100, 2)}%")
            nr_correct = 0

        self.save_nn()

    def run(self):
        # Show results
        while True:
            index = int(input("Enter a number (0 - 59999): "))
            if index < 0 or index > 59999:
                index = 0
            img = self.images[index]
            plt.imshow(img.reshape(28, 28), cmap="Greys")

            print(img.shape, 'Image before')
            img.shape += (1,)
            print(img.shape, 'after')
            # Forward propagation input -> hidden
            h_pre = self.b_i_h + self.w_i_h @ img
            h = 1 / (1 + np.exp(-h_pre))
            # Forward propagation hidden -> output

            o_pre  =self.b_h_o + self.w_h_o @ h
            o = 1 / (1 + np.exp(-o_pre))

            plt.title(f"Ist ne  {o.argmax()}")
            plt.show()


iterations = input('How often should it go over the training data?')
lr = input('In what rate should the values be adjusted during training Phase?')
dC = DigitClassification(lr, iterations)
