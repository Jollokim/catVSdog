import time
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim

from create_data import create_data
from engine import evaluate, train_one_epoch
from model import convnet


def main():
    data = create_data()

    device = None

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on GPU!")
    else:
        device = torch.device("cpu")
        print("Running on CPU!")


    MODELNAME = f"model-catVSdog-{int(time.time())}"
    BATCH_SIZE = 100
    EPOCHS = 6

    model = convnet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    X = torch.Tensor(np.array([i[0] for i in data])).view(-1, 50, 50)
    X = X/255
    y = torch.Tensor(np.array([i[1] for i in data]))

    # validation percentage
    VAL_PCT = 0.1
    val_size = int(len(X)*VAL_PCT)
    # print(val_size)

    train_X = X[:-val_size]
    train_y = y[:-val_size]

    test_X = X[-val_size:]
    test_y = y[-val_size:]

    

    for epoch in range(EPOCHS):
        model.train(True)
        train_one_epoch(model, optimizer, loss_function, train_X, train_y, BATCH_SIZE, device)

        model.train(False)
        with open(f"{MODELNAME}.log", "a") as f:
            f.write(f"{epoch}, {round(evaluate(model, train_X, train_y, device), 3)}, {round(evaluate(model, test_X, test_y, device), 3)}\n")

    print("Accuracy:", evaluate(model, test_X, test_y, device))

    model_state = model.state_dict()

    torch.save(model_state, MODELNAME + ".pt")

    model2 = convnet()
    model2.load_state_dict(torch.load(MODELNAME + ".pt"))
    model2.train(False)

    model2.to(device)
    print("Accuracy:", evaluate(model2, test_X, test_y, device))



if __name__ == '__main__':
    main()