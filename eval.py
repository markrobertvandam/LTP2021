import argparse
import matplotlib.pyplot as plt

from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description="Evaluate the Fake News Classifier")
parser.add_argument("output", type=Path, help="Path to model results text-file")
parser.add_argument("plot_path", type=Path, help="Path to save plots")


def main():
    args = parser.parse_args()
    with open(args.output) as f:
        avg_training_loss = [0] * 20
        training_test = [0] * 20
        avg_val_loss = [0] * 20
        avg_val_accuracy = [0] * 20
        test_loss = [0] * 20
        test_accuracy = [0] * 20
        flag = "training"
        for line in f:
            if "Training final model" in line:
                flag = "testing"
            elif "End epoch" in line:
                epoch = int(line.split("epoch ")[1].split(".")[0])
                if flag == "training":
                    avg_training_loss[epoch] += float(line.split("loss ")[1])
                elif flag == "testing":
                    print(line.split("loss ")[1])
                    training_test[epoch] = float(line.split("loss ")[1])

            elif "validation-loss" in line:
                avg_val_loss[epoch] += float(line.split("loss: ")[1])
                avg_val_accuracy[epoch] += float(
                    line.split("Accuracy: ")[1].split(" validation")[0]
                )
            elif "test-loss" in line:
                test_loss[epoch] = float(line.split("test-loss: ")[1])
                test_accuracy[epoch] += float(
                    line.split("Accuracy: ")[1].split(" test")[0]
                )

        for i in range(20):
            avg_training_loss[i] /= 10
            avg_val_loss[i] /= 10
            avg_val_accuracy[i] /= 10

        plt.figure(1)
        plt.plot(avg_training_loss[1:], "-b", label="training")
        plt.plot(avg_val_loss[1:], "-r", label="validation")
        plt.legend(loc="upper right")
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.title("Avg. Loss during cross-validation")

        plt.figure(2)
        plt.plot(avg_val_accuracy, "-b")
        plt.title("Avg. Validation Accuracy during cross-validation")
        plt.xlabel("epoch")
        plt.ylabel("Accuracy")

        test_accuracy = [i for i in test_accuracy if i != 0]
        training_test = [i for i in training_test if i != 0]
        test_loss = [i for i in test_loss if i != 0]

        plt.figure(3)
        plt.plot(test_accuracy[:], "-b")
        # plt.xticks(np.arange(1, 10))
        # plt.xlim(0.5, 9.5)
        plt.title("Test Accuracy over epochs")
        plt.xlabel("epoch")
        plt.ylabel("Accuracy")

        plt.figure(4)
        plt.plot(training_test[:], "-b", label="training")
        plt.plot(test_loss[:], "-r", label="test")
        # plt.xticks(np.arange(1, 10))
        # plt.xlim(0.5, 9.5)
        plt.legend(loc="upper right")
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.title("Test loss vs Training loss")

        plt.show()


if __name__ == "__main__":
    main()
