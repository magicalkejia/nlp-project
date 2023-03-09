Tips:

pip install torchtext== 0.6.0

pip install torch

pip install matplotlib

pip install pandas

train_set : test_set = 9:1

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    criterion = CrossEntropyLoss().to(device)
