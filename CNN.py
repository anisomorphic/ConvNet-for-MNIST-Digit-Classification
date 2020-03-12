# Michael Harris
# CAP 4453 - Robot Vision - Spring 2019
# PA2 - convnet for digit classification

__author__ = "Michael Harris"



# EVALUATION: here there is a single FC layer, mapped to 10 classes.
# sigmoid activation is used after the FC layer.
# Accuracy begins to drop after Epoch #43, indicating that training should stop here to prevent model over-fitting.
class Model_1(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        # ======================================================================
        # One fully connected layer.
        #
        self.layer = nn.Linear(input_dim, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features    #
        #
        sigmoid = torch.nn.Sigmoid()
        x = sigmoid(self.layer(x))
        x = self.output_layer(x)
        return x




# EVALUATION: here convolutional layers are added, outputting 40 kernels at each step.
# max-pooling downsamples, reducing the amount of unnecessary information at each step.
# this is a very useful approach for images due to the kernel based multi-dimensional
# convolutions. here sigmoid activation is still used after each layer. flattening the
# tensors before initiating the FC layers is required, this is where the 4*4*output channel
# trick is used, and where the .view call is located in the forward() method.
# Accuracy begins to drop after Epoch #30, indicating that training should stop here to prevent model over-fitting.
class Model_2(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        # sigmoid activation for this part
        self.activation = nn.Sigmoid()

        # input channel is 1, greyscale. map to 20 kernels, then 40 in next step.
        self.layer1 = nn.Conv2d(1, 40, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # expand to 40 kernels per the question, then pool over 2x2 regions
        self.layer2 = nn.Conv2d(40, 40, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # flatten tensor with 4*4*(output of conv layer) trick
        self.layer3 = nn.Linear(4*4*40, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features
        #
        sigmoid = torch.nn.Sigmoid()
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.activation(x)

        x = self.layer2(x)
        x = self.pool2(x)
        x = self.activation(x)

        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        x = self.activation(x)

        x = self.output_layer(x)
        return x




# EVALUATION: here, the activation function for convolutional layers is switched to ReLU,
# while the convolutional layers continue to use sigmoid activation. this increases accuracy.
# Accuracy begins to drop after Epoch #6, indicating that training should stop here to prevent model over-fitting.
class Model_3(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        # relu for convolutional and sigmoid activation for linear
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()

        # input channel is 1, greyscale. map to 20 kernels, then 40 in next step.
        self.layer1 = nn.Conv2d(1, 40, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # expand to 40 kernels per the question, then pool over 2x2 regions
        self.layer2 = nn.Conv2d(40, 40, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # flatten tensor with 4*4*(output of conv layer) trick
        self.layer3 = nn.Linear(4*4*40, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features
        #
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.activation1(x)

        x = self.layer2(x)
        x = self.pool2(x)
        x = self.activation1(x)

        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        x = self.activation2(x)

        x = self.output_layer(x)
        return x



# EVALUATION: here we insert another FC layer, which gets sigmoid activation.
# Accuracy begins to drop after Epoch #19, indicating that training should stop here to prevent model over-fitting.
class Model_4(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # relu for convolutional and sigmoid activation for linear
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()

        # input channel is 1, greyscale. map to 20 kernels, then 40 in next step.
        self.layer1 = nn.Conv2d(1, 40, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # expand to 40 kernels per the question, then pool over 2x2 regions
        self.layer2 = nn.Conv2d(40, 40, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # flatten tensor with 4*4*(output of conv layer) trick
        self.layer3 = nn.Linear(4*4*40, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features
        #
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.activation1(x)

        x = self.layer2(x)
        x = self.pool2(x)
        x = self.activation1(x)

        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        x = self.activation2(x)


        x = self.layer4(x)
        x = self.activation2(x)

        x = self.output_layer(x)
        return x


# EVALUATION: here we increase the size of the layer inserted in part 4 to 1000 neurons.
# we are to use dropout with a rate of .5, which indicates that it should be placed after
# FC layers. if we were told to use a dropout of .1 or .2, this would indicate that we should
# be placing dropout after the convolutional layer instead. here, i struggled with where to put
# dropout, and i only had success with using dropout on layer4 - that is, the hidden FC layer.
# placing dropout on layer3 alone or in conjunction with layer4 totally ruined my accuracy.
# Accuracy begins to drop after Epoch #15, indicating that training should stop here to prevent model over-fitting.
# this assignment was very fun!
class Model_5(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        # relu for convolutional and sigmoid activation for linear
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()

        self.dropout = nn.Dropout(p=0.5, inplace=False)

        # input channel is 1, greyscale. map to 20 kernels, then 40 in next step.
        self.layer1 = nn.Conv2d(1, 40, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # expand to 40 kernels per the question, then pool over 2x2 regions
        self.layer2 = nn.Conv2d(40, 40, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # flatten tensor with 4*4*(output of conv layer) trick
        self.layer3 = nn.Linear(4*4*40, 1000)
        self.layer4 = nn.Linear(1000, 1000)
        self.output_layer = nn.Linear(1000, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features
        #
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.activation1(x)

        x = self.layer2(x)
        x = self.pool2(x)
        x = self.activation1(x)

        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        x = self.activation2(x)
        # x = self.dropout(x)

        x = self.layer4(x)
        x = self.activation2(x)
        x = self.dropout(x)
        # x = self.dropout(x)#this one being active, and not the one after layer3.activation2, resulted in a 99.45% accuracy

        x = self.output_layer(x)
        return x

class Net(nn.Module):
    def __init__(self, mode, args):
        super().__init__()
        self.mode = mode
        self.hidden_size= args.hidden_size
        # model 1: base line
        if mode == 1:
            in_dim = 28*28 # input image size is 28x28
            self.model = Model_1(in_dim, self.hidden_size)

        # model 2: use two convolutional layer
        if mode == 2:
            self.model = Model_2(self.hidden_size)

        # model 3: replace sigmoid with relu
        if mode == 3:
            self.model = Model_3(self.hidden_size)

        # model 4: add one extra fully connected layer
        if mode == 4:
            self.model = Model_4(self.hidden_size)

        # model 5: utilize dropout
        if mode == 5:
            self.model = Model_5(self.hidden_size)


    def forward(self, x):
        if self.mode == 1:
            x = x.view(-1, 28* 28)
            x = self.model(x)
            # x.cuda()
        if self.mode in [2, 3, 4, 5]:
            x = self.model(x)
            
        logits = F.softmax(x, dim=1)
        # EVALUATION: here I simply use the functional softmax function for classification.

        return logits
