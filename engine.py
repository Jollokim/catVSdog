import torch

from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_function, train_X, train_y, batch_size, device):
    for i in tqdm(range(0, len(train_X), batch_size)):
        # print(i, i+BATCH_SIZE)
        batch_X = train_X[i:i+batch_size].view(-1, 1, 50, 50).to(device)
        batch_y = train_y[i:i+batch_size].to(device)

        model.zero_grad()
        outputs = model(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
        

        # model.zero_grad()
    print("loss:", loss)


def evaluate(net: torch.nn.Module, test_X, test_y, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i])
            net_out = net(test_X[i].view(-1, 1, 50, 50).to(device))[0]

            # print(real_class, net_out)
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1

            total += 1

    # print("Accuracy:", round(correct/total, 3))

    return round(correct/total, 3)