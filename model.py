import numpy as np
import argparse, fasttext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score, classification_report, f1_score

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
print(torch.__version__)  # this should be at least 1.0.0
# torch.set_printoptions(profile="full") # print complete tensors rather than truncated, for debugging


parser = argparse.ArgumentParser()
parser.add_argument('train', help="data training file")
parser.add_argument('dev', help="data dev file")
parser.add_argument('test', help="data test file")
parser.add_argument('--iters', help="epochs (iterations)", type=int, default=10)
args = parser.parse_args()


# ----------------------------------------------------------
# Classes/functions related to PyTorch

class NN(nn.Module):
    """ The neural network that will be used """

    def __init__(self, input_dim, output_dim):
        """
        Args:
            input_dim (int): size of the input features (i.e. vocabulary size)
            output_dim (int): number of classes
        """
        super(NN, self).__init__()
        # self.fc1 = nn.linear(input_dim, output_dim)

        # concat
        self.embeddings = nn.Embedding(input_dim, 100)

        # mean
        # self.embeddings = nn.Embedding(input_dim, 500)

        self.fc1 = nn.Linear(500, 200)
        self.fc2 = nn.Linear(200, output_dim)

    def forward(self, x):
        """The forward pass of the NN

        Args:
            x (torch.Tensor): an input data tensor.
                x.shape should be (batch_size, input_dim)
        Returns:
            the resulting tensor. tensor.shape should be (batch_size, num_classes)
        """
        x = self.embeddings(x)  # every wordid is converted to a 200-dimensional vector
        x = x.view((100, 500))  # concat approach, translates the 5 vectors into one 1000-dimensional vector
        x = self.fc1(x)
        # ------ Comment the part below to get one-layer network
        x = F.leaky_relu(x)
        x = self.fc2(x)
        # ------
        x = F.log_softmax(x, dim=1)
        #        x = F.softmax(x, dim=1) #log_softmax is preferred, it avoids very big (or low) numbers

        return x

    def print_params(self):
        """Print the parameters (theta) of the network. Mainly for debugging purposes"""
        for name, param in model.named_parameters():
            print(name, param.data)


def tensor_desc(x):
    """ Inspects a tensor: prints its type, shape and content"""
    print("Type:   {}".format(x.type()))
    print("Size:   {}".format(x.size()))
    print("Values: {}".format(x))


# ----------------------------------------------------------

## read input data
print("load data..")
X_train, y_train, X_dev, y_dev, X_test, y_test, word2idx, tag2idx = load_data(args.train, args.dev, args.test)

print("#train instances: {}\n#dev instances: {}\n#test instances: {}".format(len(X_train), len(X_dev), len(X_test)))
assert (len(X_train) == len(y_train))
assert (len(X_test) == len(y_test))
assert (len(X_dev) == len(y_dev))

vocabulary_size = len(word2idx.keys())
num_classes = len(tag2idx)
input_size = len(X_train[0])
print(input_size)

print("#build model")
# TODO: Check if embed file exists in gh folder, else download
    #fasttext.util.download_model('en')
    #fasttext.load_model('folder/cc.en.300.bin')
# The network should have as input size the vocabulary size and as output size the number of classes
model = NN(vocabulary_size, num_classes)

print("#Model: {}".format(model))

criterion = nn.NLLLoss()
optimizer = optim.Adam(params=model.parameters())

print("#Training..")
num_epochs = args.iters
size_batch = 100
num_batches = len(X_train) // size_batch
print("#Batch size: {}, num batches: {}".format(size_batch, num_batches))
for epoch in range(num_epochs):
    # Optional. Shuffle the data (X and y) so that it is processed in different order in each epoch
    epoch_loss = 0
    for batch in range(num_batches):
        batch_begin = batch * size_batch
        batch_end = (batch + 1) * (size_batch)
        X_data = X_train[batch_begin:batch_end]
        y_data = y_train[batch_begin:batch_end]
        # Q5. Your code here. Convert X_data and y_data into PyTorch tensors X_tensor and y_tensor.
        X_tensor = torch.LongTensor(X_data)
        y_tensor = torch.LongTensor(y_data)
        # Q5. End of your code
        # print("X_tensor")
        # tensor_desc(X_tensor)
        # print("y_tensor")
        # tensor_desc(y_tensor)

        optimizer.zero_grad()

        y_pred = model(X_tensor)
        #        print("#Y_pred")
        #        tensor_desc(y_pred)
        loss = criterion(y_pred, y_tensor)
        #        print("#Loss: {}".format(loss))

        #        model.print_params()
        loss.backward()
        optimizer.step()
        #        model.print_params()

        epoch_loss += loss.item()

    print("  End epoch {}. Average loss {}".format(epoch, epoch_loss / num_batches))

    print("  Validation")
    epoch_acc = 0
    num_batches_dev = len(X_dev) // size_batch
    print("    #num batches dev: {}".format(num_batches_dev))
    for batch in range(num_batches_dev):
        batch_dev_begin = batch * size_batch
        batch_dev_end = (batch + 1) * (size_batch)
        X_data_dev = X_dev[batch_dev_begin:batch_dev_end]
        y_data_dev = y_dev[batch_dev_begin:batch_dev_end]

        # Q5b. Your code here. Convert X_data_dev and y_data_dev into PyTorch tensors X_tensor_dev and y_tensor_dev
        X_tensor_dev = torch.LongTensor(X_data_dev)
        y_tensor_dev = torch.LongTensor(y_data_dev)
        # Q5b. End of your code

        # Q6. Produce the output of the network: do a forward pass over input X_tensor_dev
        # and place the result in y_pred_dev
        y_pred_dev = model(X_tensor_dev)
        #        print("y_pred_dev")
        #        tensor_desc(torch.exp(y_pred_dev))
        # Q6. End of your code

        # Q7. Evaluate the predictions in y_pred_dev.
        # - Get for each row in y_pred_dev (there are size_batch rows), the best prediction,
        #   i.e. the one with highest (log) probability, and place that in variable output
        # - Compute the accuracy comparing the prediction with the gold label in y_data_dev.
        #   You can use sklearn's accuracy_score. In that case you'll have to convert PyTorch's
        #   tensor into numpy.
        output = np.empty((0, 1), int)
        for i in range(size_batch):
            output = np.append(output, torch.argmax(y_pred_dev[i]))
        #        print(output[1])
        epoch_acc += accuracy_score(output, y_data_dev)
        # Q7. End of your code
    print("    {}".format(epoch_acc / num_batches_dev))