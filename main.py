import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
from dataloader import DataLoader
from model import LSTM_Attention
from model2 import BiLSTM_Attention
from model3 import LSTM_Standard
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


def instantiate_model(vocab_size,model_type):
    vocab_size = vocab_size
    embedding_size = 100
    hidden_dim = 64
    n_layers = 1
    num_class = 3
    if model_type == 1:
        net = LSTM_Attention(vocab_size=vocab_size, embed_dim=embedding_size,
                                  hidden_dim=hidden_dim, n_layers=n_layers, num_class=num_class).to(device)
    if model_type == 2:
        net = BiLSTM_Attention(vocab_size=vocab_size, embedding_dim=embedding_size,
                                        hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    if model_type == 3:
        net = LSTM_Standard(vocab_size=vocab_size, embedding_dim=embedding_size,
                                        hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    return net



def train_model(net, iterator, optimizer, criterion, data_len):
    epoch_loss = 0.0
    epoch_acc = 0.0
    for batch in iterator:
        optimizer.zero_grad()
        preds = net(batch.Title)
        label = batch.Attitude
        loss = criterion(preds, label)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        epoch_acc += ((preds.argmax(axis=1)) == batch.Attitude).sum().item()
    return epoch_loss/(len(iterator)), epoch_acc/data_len


def evaluate_model(net, iterator, criterion, data_len):
    epoch_acc, total_count, epoch_loss = 0.0, 0, 0.0
    with torch.no_grad():
        for batch in iterator:
            preds = net(batch.Title)
            label = batch.Attitude
            loss = criterion(preds, label)
            epoch_loss += loss.item()
            epoch_acc += (preds.argmax(1) == batch.Attitude).sum().item()
            total_count += batch.Attitude.size(0)

    return epoch_loss/(len(iterator)), epoch_acc/data_len


def visualize(plot_1, plot_2, label_1, label_2, save_path):
    plt.figure(figsize=(10, 5), dpi=80)
    plt.plot(plot_1, label=label_1)
    plt.plot(plot_2, color='coral', label=label_2)
    plt.legend(loc=0)
    plt.grid(True, linestyle='--', alpha=1)
    plt.xlabel('epoch', fontsize=15)
    plt.savefig(save_path)
    plt.show()



def main():

    if os.path.exists("train.csv") == False:
        dataloader = DataLoader(path='dataset.csv', batch_size=32)
        # dataloader.get_attitude_value()
        dataloader.divide_dataset(0.05)
    dataloader = DataLoader(path='train.csv', batch_size=32)
    train_dataset = dataloader.preprocess()[0]
    iterator = dataloader.preprocess()[1]
    vocab = dataloader.preprocess()[2]

    '''
    instantiate_model args: vocab_size, model_type
    model_type  1: LSTM+self-attention 2: BiLSTM+softattention 3. LSTM only'''
    model_type = 2
    net = instantiate_model(vocab.shape[0],model_type)

    # set optimizer,criterion,embedding

    net.embedding.weight.data.copy_(vocab)


    optimizer = optim.Adam(net.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.3)

    # testing
    dataloader_test = DataLoader(path='test.csv', batch_size=16)
    test_dataset = dataloader_test.preprocess()[0]
    test_iterator = dataloader_test.preprocess()[1]

    # begin training ,testing and visualization
    test_acc,test_loss,acc,loss,best_valid_acc= 0,0,0,0,0
    acc_list, loss_list, test_acc_list, test_lost_list = [], [], [], []
    for epoch in range(50):

        if epoch != 0:
            scheduler.step()
        train_loss, train_acc = train_model(
            net, iterator, optimizer, criterion, len(train_dataset))
        print(
            f"Epoch: {epoch+1}/{50}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}")
        result = evaluate_model(
            net, test_iterator, criterion, len(test_dataset))
        print("test loss {:8.3f},test accuracy{:8.3f}".format(
            result[0], result[1]))
        
        if result[1] > best_valid_acc:          #save the best
            best_valid_acc = result[1]
            torch.save(net.state_dict(), './save model/model{0}.pt'.format(model_type))

        acc_list.append(train_acc)
        loss_list.append(train_loss)
        test_acc_list.append(result[1])
        test_lost_list.append(result[0])

        if epoch > 10:
            acc += train_acc
            loss += train_loss
            test_acc += result[1]
            test_loss += result[0]
    acc = acc/40
    loss  = loss/40
    test_acc = test_acc/40
    test_loss = test_loss/40
    print("average_acc {:8.3f},average_loss{:8.3f}".format(acc, loss))
    print("test_acc {:8.3f},test_loss{:8.3f}".format(test_acc, test_loss))
    
    data_to_csv = pd.DataFrame({'train_acc':acc_list,'train_loss':loss_list,'test_acc':test_acc_list,'test_loss':test_lost_list})
    data_to_csv.to_csv('outcome{0}.csv'.format(model_type))
    visualize(acc_list, test_acc_list, "train_acc",
              "test_acc_list", "./png/acc{0}.png".format(model_type))
    visualize(loss_list, test_lost_list, "train_loss",
              "test_lost_list", "./png/loss{0}.png".format(model_type))

    # save or load model
    # torch.save(net, "./save model/")
    # net = torch.load("./save model/")
    # net.eval()


if __name__ == "__main__":

    main()
