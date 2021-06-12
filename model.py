from pathlib import Path
from early_stopping import EarlyStopping
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from utils import create_cross_validator, create_data_loaders, concat_x_y
from operator import itemgetter
from tap import Tap

from sklearn.metrics import accuracy_score

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
print(torch.__version__)  # this should be at least 1.0.0


class ArgParser(Tap):
    preprocessed_data: Path  # Preprocessed data path
    save_path: Path  # Directory in which the model is saved
    delta_es: float = 0.0  # Delta value for early stopping
    iters: int = 20  # Number of epochs
    batch_size: int = 128  # Batch size

    def configure(self) -> None:
        self.add_argument("preprocessed_data")
        self.add_argument("save_path")


class NN(nn.Module):
    """
    The neural network that will be used
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(NN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the NN
        """
        x = self.fc1(x)
        x = f.leaky_relu(x)
        x = self.fc2(x)
        x = f.log_softmax(x, dim=1)

        return x


def load_data(datafile):
    """
    Load data
    """
    data = np.load(datafile)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print()

    return X_train, y_train, X_val, y_val, X_test, y_test


parser = ArgParser()
args = parser.parse_args()

# read input data
print("Loading data..")
X_train, y_train, X_dev, y_dev, X_test, y_test = load_data(args.preprocessed_data)

print(
    f"#train instances: {len(y_train)}\n"
    f"#dev instances: {len(y_dev)}\n"
    f"#test instances: {len(y_test)}"
)
assert len(X_train) == len(y_train)
assert len(X_dev) == len(y_dev)
assert len(X_test) == len(y_test)

print("#Training..")
num_epochs = args.iters
size_batch = args.batch_size

# Concat X and y because of the cross validator format
train_data = concat_x_y(X_train, y_train)
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

    # Model initialization for each k-fold
    model = NN(300, 2)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(params=model.parameters())
    print(f"#Training K-fold: {count_k_fold}")
    for epoch in range(args.iters):
        epoch_loss = 0
        for X_train_batch, y_train_batch in train_loader:
            X_tensor = torch.FloatTensor(X_train_batch)
            y_tensor = torch.LongTensor(y_train_batch)

            optimizer.zero_grad()

            y_pred = model(X_tensor)
            loss = criterion(y_pred, y_tensor)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"  End epoch {epoch}. Average loss {epoch_loss / len(train_loader)}")

        print("  Validation")
        epoch_acc = 0
        val_loss = 0

        for X_val_batch, y_val_batch in val_loader:
            # Convert X_data_dev and y_data_dev into PyTorch
            # tensors X_tensor_dev and y_tensor_dev
            X_tensor_dev = torch.FloatTensor(X_val_batch)
            y_tensor_dev = torch.LongTensor(y_val_batch)

            y_pred_dev = model(X_tensor_dev)

            output = torch.argmax(y_pred_dev, dim=1)
            epoch_acc += accuracy_score(output, y_val_batch)
            val_loss += criterion(y_pred_dev, y_val_batch)

        print(
            f"  Accuracy: {epoch_acc / len(val_loader)}"
            f"  validation-loss: {val_loss / len(val_loader)}"
        )

# Train final model on the whole training data

# Concat val and test set because this is the final test set
X_final_test = np.concatenate((X_dev, X_test), axis=0)
y_final_test = np.concatenate((y_dev, y_test), axis=0)

# Concat X and y because of the cross validator format
train_data = concat_x_y(X_train, y_train)
test_data = concat_x_y(X_final_test, y_final_test)

train_loader, test_loader = create_data_loaders(
    train_data=train_data, val_data=test_data, batch_size=size_batch
)

# initialize the early_stopping object
early_stopping = EarlyStopping(
    patience=2, verbose=False, delta=float(args.delta_es), path=args.save_path
)

# model initialization for each k-fold
model = NN(300, 2)
criterion = nn.NLLLoss()
optimizer = optim.Adam(params=model.parameters())
print("Training final model on 80% train data, 20% test data for 10 epochs...")
for epoch in range(args.iters):
    epoch_loss = 0
    for X_train_batch, y_train_batch in train_loader:

        X_tensor = torch.FloatTensor(X_train_batch)
        y_tensor = torch.LongTensor(y_train_batch)

        optimizer.zero_grad()

        y_pred = model(X_tensor)

        loss = criterion(y_pred, y_tensor)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"  End epoch {epoch}. Average loss {epoch_loss / len(train_loader)}")

    print("  Test set")
    epoch_acc = 0
    test_loss = 0

    for X_test_batch, y_test_batch in test_loader:
        # Convert X_data_dev and y_data_dev into PyTorch
        # tensors X_tensor_dev and y_tensor_dev
        X_tensor_test = torch.FloatTensor(X_test_batch)
        y_tensor_test = torch.LongTensor(y_test_batch)

        y_pred_test = model(X_tensor_test)

        output = torch.argmax(y_pred_test, dim=1)

        epoch_acc += accuracy_score(output, y_test_batch)
        test_loss += criterion(y_pred_test, y_test_batch)

    print(
        f"  Accuracy: {epoch_acc / len(test_loader)}"
        f"  test-loss: {test_loss / len(test_loader)}"
    )

    early_stopping(test_loss / len(test_loader), model, epoch)
    if early_stopping.early_stop:
        print("Early stopping")
        break
