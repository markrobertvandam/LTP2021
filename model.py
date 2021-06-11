import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from utils import create_cross_validator, create_data_loaders, concat_X_y
from operator import itemgetter

from sklearn.metrics import accuracy_score

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
print(torch.__version__)  # this should be at least 1.0.0
# torch.set_printoptions(profile="full") # print complete tensors rather than truncated, for debugging


parser = argparse.ArgumentParser()
parser.add_argument("preprocessed_data", help="Preprocessed data path")
parser.add_argument("save_path", help="Path to save model")
parser.add_argument("--iters", help="epochs (iterations)", type=int, default=10)
args = parser.parse_args()


# ----------------------------------------------------------
# Classes/functions related to PyTorch


class NN(nn.Module):
    """The neural network that will be used"""

    def __init__(self, input_dim, output_dim):
        """
        Args:
            input_dim (int): size of the input features (i.e. vocabulary size)
            output_dim (int): number of classes
        """
        super(NN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, output_dim)

    def forward(self, x):
        """The forward pass of the NN

        Args:
            x (torch.Tensor): an input data tensor.
                x.shape should be (batch_size, input_dim)
        Returns:
            the resulting tensor. tensor.shape should be (batch_size, num_classes)
        """
        x = self.fc1(x)
        x = f.leaky_relu(x)
        x = self.fc2(x)
        # ------
        x = f.log_softmax(x, dim=1)
        #        x = F.softmax(x, dim=1) #log_softmax is preferred, it avoids very big (or low) numbers

        return x

    def print_params(self):
        """Print the parameters (theta) of the network. Mainly for debugging purposes"""
        for name, param in model.named_parameters():
            print(name, param.data)


def tensor_desc(x):
    """Inspects a tensor: prints its type, shape and content"""
    print("Type:   {}".format(x.type()))
    print("Size:   {}".format(x.size()))
    print("Values: {}".format(x))


# ----------------------------------------------------------


def load_data(datafile):
    """load data"""
    data = np.load(datafile)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print()

    return X_train, y_train, X_val, y_val, X_test, y_test


# ----------------------------------------------------------


# read input data
print("load data..")
X_train, y_train, X_dev, y_dev, X_test, y_test = load_data(args.preprocessed_data)

print(
    "#train instances: {}\n#dev instances: {}\n#test instances: {}".format(
        len(y_train), len(y_dev), len(y_test)
    )
)
assert len(X_train) == len(y_train)
assert len(X_dev) == len(y_dev)
assert len(X_test) == len(y_test)

num_classes = 2

# The network should have as input size the vocabulary size and as output size the number of classes
model = NN(300, num_classes)

print("#Model: {}".format(model))

criterion = nn.NLLLoss()
optimizer = optim.Adam(params=model.parameters())

print("#Training..")
num_epochs = args.iters
size_batch = 100
# num_batches = len(X_train) // size_batch
# print("#Batch size: {}, num batches: {}".format(size_batch, num_batches))

# Concat X and y because of the cross validator format
train_data = concat_X_y(X_train, y_train)
# Init cross validation
cross_validator = create_cross_validator()
count_k_fold = 0
for train_idx, val_idx in cross_validator.split(train_data):

    count_k_fold += 1
    train_loader, val_loader = create_data_loaders(
        train_data=itemgetter(*train_idx)(train_data),
        val_data=itemgetter(*val_idx)(train_data),
        batch_size=size_batch,
    )

    # model initialization for each k-fold
    model = NN(300, num_classes)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(params=model.parameters())
    print("#Training k-fold: {}".format(count_k_fold))
    num_epochs = args.iters
    for epoch in range(num_epochs):
        # Optional. Shuffle the data (X and y) so that it is processed in different order in each epoch
        epoch_loss = 0
        # for batch in range(num_batches):
        for X_train_batch, y_train_batch in train_loader:
            # batch_begin = batch * size_batch
            # batch_end = (batch + 1) * size_batch

            # X_data = X_train[batch_begin:batch_end]
            # y_data = y_train[batch_begin:batch_end]

            X_tensor = torch.FloatTensor(X_train_batch)
            y_tensor = torch.LongTensor(y_train_batch)

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

        # print("  End epoch {}. Average loss {}".format(epoch, epoch_loss / num_batches))
        print(
            "  End epoch {}. Average loss {}".format(
                epoch, epoch_loss / len(train_loader)
            )
        )

        print("  Validation")
        epoch_acc = 0
        val_loss = 0
        #num_batches_dev = len(X_dev) // size_batch
        #print("    #num batches dev: {}".format(num_batches_dev))

        for X_val_batch, y_val_batch in val_loader:
            # for batch in range(num_batches_dev):
            # batch_dev_begin = batch * size_batch
            # batch_dev_end = (batch + 1) * size_batch
            # X_data_dev = X_dev[batch_dev_begin:batch_dev_end]
            # y_data_dev = y_dev[batch_dev_begin:batch_dev_end]

            # Convert X_data_dev and y_data_dev into PyTorch tensors X_tensor_dev and y_tensor_dev
            X_tensor_dev = torch.FloatTensor(X_val_batch)
            y_tensor_dev = torch.LongTensor(y_val_batch)

            y_pred_dev = model(X_tensor_dev)

            output = torch.argmax(y_pred_dev, dim=1)
            # print(output, y_data_dev)
            # print(output, y_val_batch)
            # epoch_acc += accuracy_score(output, y_data_dev)
            epoch_acc += accuracy_score(output, y_val_batch)
            val_loss += criterion(output, y_val_batch)

        #print("    {}".format(epoch_acc / num_batches_dev))
        print(" Accuracy: {}  validation-loss: {}".format(epoch_acc / len(val_loader), val_loss/len(val_loader)))

# Train final model on the whole training data

# Concat val and test set because this is the final test set
X_final_test = np.concatenate((X_dev, X_test), axis=0)
y_final_test = np.concatenate((y_dev, y_test), axis=0)

# Concat X and y because of the cross validator format
train_data = concat_X_y(X_train, y_train)
test_data = concat_X_y(X_final_test, y_final_test)

train_loader, test_loader = create_data_loaders(
    train_data=train_data, val_data=test_data, batch_size=size_batch
)

# model initialization for each k-fold
model = NN(300, num_classes)
criterion = nn.NLLLoss()
optimizer = optim.Adam(params=model.parameters())
print("Training final model on 80% train data, 20% test data for 10 epochs...")
num_epochs = args.iters
for epoch in range(num_epochs):
    # Optional. Shuffle the data (X and y) so that it is processed in different order in each epoch
    epoch_loss = 0
    # for batch in range(num_batches):
    for X_train_batch, y_train_batch in train_loader:

        X_tensor = torch.FloatTensor(X_train_batch)
        y_tensor = torch.LongTensor(y_train_batch)

        optimizer.zero_grad()

        y_pred = model(X_tensor)

        loss = criterion(y_pred, y_tensor)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(
        "  End epoch {}. Average loss {}".format(epoch, epoch_loss / len(train_loader))
    )

    print("  Test set")
    epoch_acc = 0
    test_loss = 0

    for X_test_batch, y_test_batch in test_loader:
        # Convert X_data_dev and y_data_dev into PyTorch tensors X_tensor_dev and y_tensor_dev
        X_tensor_test = torch.FloatTensor(X_test_batch)
        y_tensor_test = torch.LongTensor(y_test_batch)

        y_pred_test = model(X_tensor_test)

        output = torch.argmax(y_pred_test, dim=1)

        epoch_acc += accuracy_score(output, y_test_batch)
        test_loss += criterion(output, y_test_batch)

    print(" Accuracy: {}  test-loss: {}".format(epoch_acc / len(test_loader), test_loss/len(test_loader)))

# Save final model
torch.save(model.state_dict(), args.save_path)
