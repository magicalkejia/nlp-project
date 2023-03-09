import pandas as pd
import torch
from torchtext.data import Field, Iterator, BucketIterator
from torchtext.data import TabularDataset
from torchtext.data.utils import get_tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataLoader:
    def __init__(self, path,batch_size):
        self.tokenizer = get_tokenizer("basic_english")
        self.path = path
        self.batch_size = batch_size

    def divide_dataset(self,split_ratio):
        train_data = pd.read_csv(self.path)
        train_data = train_data.sample(frac=1.0)
        cut_idx = int(round(split_ratio * train_data.shape[0]))
        df_test, df_train = train_data.iloc[:cut_idx], train_data.iloc[cut_idx:]
        df_test.to_csv('test.csv', index=None)
        df_train.to_csv('train.csv', index=None)


    def get_labels_vector(self):
        data = pd.read_csv(self.path)
        label_list = data["Decisions"]
        text = data["Title"]
        vectors = []
        for i in range(0, len(label_list)):
            dict = eval(label_list[i])
            line = self.tokenizer(text[i])
            keys = dict.keys()
            vector = [0]*len(line)
            for key in keys:
                list_ = self.tokenizer(key)
                for item in list_:
                    if item in line:

                        loc = line.index(item)
                        if dict[key] == "neutral":
                            vector[loc] = 1
                        elif dict[key] == "positive":
                            vector[loc] = 2
                        elif dict[key] == "negative":
                            vector[loc] = -1
                        else:
                            print("error", line[i])

            # print(vector)
            vectors.append(vector)
        data.insert(4, "Vector", vectors)
        data.to_csv(self.path, index=None)

    def get_attitude_value(self):
        data = pd.read_csv(self.path)
        attitude_list = []
        label_list = data["Decisions"]
        for i in range(0, len(label_list)):
            dict = eval(label_list[i])
            keys = dict.keys()
            value = 0
            for key in keys:
                if dict[key] == "neutral":
                    value += 0 
                elif dict[key] == "positive":
                    value += 1
                elif dict[key] == "negative":
                    value += 2
                else:
                    print("error")

            average = round(value/len(keys))
            attitude_list.append(average)
        data.insert(5, "Attitude", attitude_list)
        data.to_csv(self.path, index=None)

    def preprocess(self):
        # tokenization
        TEXT = Field(sequential=True, lower=True, tokenize=self.tokenizer,dtype=torch.long)

        LABEL = Field(sequential=False, use_vocab=False)

        Fields = [("S No.", None), ('Title', TEXT),
                ('Decisions', None), ("Words", None), ("Vector",None),("Attitude", LABEL)]

        data_set = TabularDataset(
            path=self.path, format='csv', skip_header=True, fields=Fields)
        
        # build vocab
        TEXT.build_vocab(data_set)
        TEXT.vocab.load_vectors('glove.6B.100d', unk_init=torch.Tensor.normal_)
        vocab = TEXT.vocab.vectors


        iter = BucketIterator(
            dataset=data_set,
            batch_size=self.batch_size,
            device=device,
            sort_within_batch=False, repeat=False)

        # for batch in iter:
        #     print (batch.Title.shape)
        #batch.Attitude.shape torch.Size([14])
        #batch.Title.shape torch.Size([16, 14])

        return data_set,iter,vocab
