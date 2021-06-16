from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from utils import create_data_loaders, concat_x_y
from tap import Tap
import torch.nn.functional as f

from sklearn.metrics import accuracy_score

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
print(torch.__version__)  # this should be at least 1.0.0


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


class ArgParser(Tap):
    preprocessed_data: Path  # Preprocessed data path
    model_path: Path  # Directory to the saved model
    data_source: str  # Whether the data that is tested on is harvard or kaggle

    def configure(self) -> None:
        self.add_argument("preprocessed_data")
        self.add_argument("model_path")
        self.add_argument("data_source")


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


def load_model(model_path):
    model = NN(300, 2)
    model.load_state_dict(torch.load(model_path))

    return model.eval()


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

# Concat X and y because of the cross validator format
train_data = concat_x_y(X_train, y_train)

# Train final model on the whole training data

# Concat all data to be one big test_set if loaded_model is harvard model
if args.data_source == "kaggle":
    X_test = np.concatenate((X_train, X_dev, X_test), axis=0)
    y_test = np.concatenate((y_train, y_dev, y_test), axis=0)

test_data = concat_x_y(X_test, y_test)

test_loader, _ = create_data_loaders(train_data=test_data, val_data=None, batch_size=128)

model = load_model(args.model_path)
criterion = nn.NLLLoss()

print("  Test set")
test_acc = 0
test_loss = 0

for X_test_batch, y_test_batch in test_loader:
        
    # Convert X_data_dev and y_data_dev into PyTorch
    # tensors X_tensor_dev and y_tensor_dev
    X_tensor_test = torch.FloatTensor(X_test_batch)
    if args.data_source == "harvard":
       y_test_batch = y_test_batch.to(dtype=torch.int64)

    y_pred_test = model(X_tensor_test)

    output = torch.argmax(y_pred_test, dim=1)

    test_acc += accuracy_score(output, y_test_batch)
    test_loss += criterion(y_pred_test, y_test_batch)

print(
    f"  Accuracy: {test_acc / len(test_loader)}"
    f"  test-loss: {test_loss / len(test_loader)}"
)
