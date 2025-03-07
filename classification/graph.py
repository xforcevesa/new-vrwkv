import numpy as np
from matplotlib import pyplot as plt


def process_logfile(filename: str, name: str):
    with open(filename) as file:
        lines = file.readlines()
    lines = [line.replace("\n", " ") for line in lines if "Accuracy" in line and "216" in line]
    accuracy = np.array([
        float(line.strip().split()[-1][:-1])
        for line in lines
    ])
    iterations = np.arange(accuracy.shape[0])
    accuracy_list = list(accuracy.tolist())
    acc_max = max(accuracy_list)
    iter_max = accuracy_list.index(acc_max)
    acc_last = accuracy_list[-1]
    iter_last = len(accuracy_list) - 1
    plt.plot(iterations, accuracy, label=name)
    plt.scatter(iter_last, acc_last, label=f"{name}: {iter_last} - {acc_last}")
    plt.scatter(iter_max, acc_max, label=f"{name}: {iter_max} - {acc_max}")


def main():
    plt.figure(figsize=(10, 8))

    # process_logfile(filename="log.txt.4", name="Original")
    # process_logfile(filename="log.txt.5", name="v7 Uni-Shift")
    # process_logfile(filename="log.txt.6", name="v6")
    process_logfile(filename="log.txt.9", name="v7 Best (Error)")
    process_logfile(filename="log.txt.11", name="v7 Best")
    process_logfile(filename="log.txt.10", name="v6 Best")
    process_logfile(filename="log.txt.7", name="v7 Enhanced")
    process_logfile(filename="log.txt.8", name="v6 Enhanced")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy (%)")
    plt.savefig("images/graph.jpg")

if __name__ == '__main__':
    main()
