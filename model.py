import numpy as np
import argparse, fasttext, os

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
parser.add_argument("train", help="data training file")
parser.add_argument("dev", help="data dev file")
parser.add_argument("test", help="data test file")
parser.add_argument("--iters", help="epochs (iterations)", type=int, default=10)
args = parser.parse_args()


# ----------------------------------------------------------
# Classes/functions related to PyTorch


class NN(nn.Module):
    """The neural network that will be used"""

    def __init__(self, input_dim, output_dim, embedding):
        """
        Args:
            input_dim (int): size of the input features (i.e. vocabulary size)
            output_dim (int): number of classes
        """
        super(NN, self).__init__()
        # self.fc1 = nn.linear(input_dim, output_dim)

        # concat
        self.embeddings = embedding

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
        x = self.embeddings[x]  # every wordid is converted to a 200-dimensional vector
        x = x.view(
            (100, 500)
        )  # concat approach, translates the 5 vectors into one 1000-dimensional vector
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
    """Inspects a tensor: prints its type, shape and content"""
    print("Type:   {}".format(x.type()))
    print("Size:   {}".format(x.size()))
    print("Values: {}".format(x))


# ----------------------------------------------------------


def get_index(word, word2idx, freeze=False):
    """
    map words to indices
    keep special OOV token (_UNK) at position 0
    """
    if word in word2idx:
        return word2idx[word]
    else:
        if not freeze:
            word2idx[word] = len(word2idx)  # new index
            return word2idx[word]
        else:
            return word2idx["_UNK"]


def load_data(trainfile, devfile, testfile):
    ### load data
    train_titles, train_articles, train_y = open_data(trainfile)
    dev_titles, dev_articles, dev_y = open_data(devfile)
    test_titles, test_articles, test_y = open_data(testfile)

    ### create mapping word to indices
    word2idx = {"_UNK": 0}  # reserve 0 for OOV

    ### convert training etc data to indices
    X_train_titles = [[get_index(w, word2idx) for w in x] for x in train_titles]
    X_train_articles = [[get_index(w, word2idx) for w in x] for x in train_articles]
    freeze = True
    X_dev_titles = [[get_index(w, word2idx, freeze) for w in x] for x in dev_titles]
    X_dev_articles = [[get_index(w, word2idx, freeze) for w in x] for x in dev_articles]

    X_test_titles = [[get_index(w, word2idx, freeze) for w in x] for x in test_titles]
    X_test_articles = [
        [get_index(w, word2idx, freeze) for w in x] for x in test_articles
    ]

    #    print(X_train[0])

    vocab_size = len(word2idx)
    print("#vocabulary size: {}".format(len(word2idx)))

    return (
        X_train_titles,
        X_train_articles,
        train_y,
        X_dev_titles,
        X_dev_articles,
        dev_y,
        X_test_titles,
        X_test_articles,
        test_y,
        word2idx,
    )


def open_data(datafile):
    """ """
    input = [line.strip().split("\t") for line in open(datafile)]
    titles = [title.split() for title, article, label in input]
    articles = [article.split() for title, article, label in input]
    labels = [label for title, article, label in input]
    return titles, articles, labels


# ----------------------------------------------------------


## read input data
print("load data..")
(
    X_train_titles,
    X_train_articles,
    train_y,
    X_dev_titles,
    X_dev_articles,
    dev_y,
    X_test_titles,
    X_test_articles,
    test_y,
    word2idx,
) = load_data(args.train, args.dev, args.test)

print(
    "#train instances: {}\n#dev instances: {}\n#test instances: {}".format(
        len(train_y), len(dev_y), len(test_y)
    )
)
assert len(X_train_titles) == len(train_y)
assert len(X_train_articles) == len(train_y)
assert len(X_dev_titles) == len(dev_y)
assert len(X_dev_articles) == len(dev_y)
assert len(X_test_titles) == len(test_y)
assert len(X_test_articles) == len(test_y)


vocabulary_size = len(word2idx.keys())
num_classes = 2

print("#build model")
if not os.path.exists("data/cc.en.300.bin"):
    fasttext.util.download_model("en")
    # TODO: How to save it in embeddings/?
embedding = fasttext.load_model("embeddings/cc.en.300.bin")

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
