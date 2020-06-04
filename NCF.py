# Python imports
import argparse
import pickle
from collections import defaultdict
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

# Workspace imports
from .evaluate import evaluate_model

from .Dataset import MovieLensDataset
from time import time
import matplotlib.pyplot as plt
import pandas as pd


torch.manual_seed(0)
feature_vector_size = 64
item_vectors = defaultdict(list)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Run NCF.")
    parser.add_argument("--path", nargs="?", default="Data/", help="Input data path.")
    parser.add_argument(
        "--dataset", nargs="?", default="vlad_u", help="Choose a dataset."
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument(
        "--layers",
        nargs="?",
        default="[16,32,16,8]",
        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings."
             "So layers[0]/2 is the embedding size.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.00001,
        help="Regularization for each layer",
    )
    parser.add_argument(
        "--num_neg_train",
        type=int,
        default=4,
        help="Number of negative instances to pair with a positive instance while training",
    )
    parser.add_argument(
        "--num_neg_test",
        type=int,
        default=100,
        help="Number of negative instances to pair with a positive instance while testing",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0,
        help="Add dropout layer after each dense layer, with p = dropout_prob",
    )
    parser.add_argument(
        "--learner",
        nargs="?",
        default="adam",
        help="Specify an optimizer: adagrad, adam, rmsprop, sgd",
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="Show performance per X iterations"
    )
    parser.add_argument(
        "--out", type=int, default=1, help="Whether to save the trained model."
    )
    return parser.parse_args()


class NCF(nn.Module):
    def __init__(self, n_users, n_items, layers=[16, 8], dropout=False):
        """
        Simple Feedforward network with Embeddings for users and items
        """
        super().__init__()
        assert layers[0] % 2 == 0, "layers[0] must be an even number"
        self.__alias__ = "NCF {}".format(layers)
        self.__dropout__ = dropout

        # user and item embedding layers
        embedding_dim = int(layers[0] / 2)
        self.user_embedding = torch.nn.Embedding(n_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(n_items, embedding_dim)

        # list of weight matrices
        self.fc_layers = torch.nn.ModuleList()
        # hidden dense layers
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        # final prediction layer
        self.output_layer = torch.nn.Linear(layers[-1], 1)

    def forward(self, feed_dict):
        users = feed_dict["user_id"]
        items = feed_dict["item_id"]
        item_emb = np.empty([0, feature_vector_size])
        for item in items:
            if item_vectors.get(str(item.data.numpy())) is not None:
                new_list = item_vectors.get(str(item.data.numpy()))[0]
                new_list = [float(i) for i in new_list]
                item_emb = np.append(item_emb, [new_list], axis=0)
            else:
                embedding = np.zeros(feature_vector_size)
                item_emb = np.append(item_emb, [embedding], axis=0)

        user_embedding = self.user_embedding(users).float()
        item_embedding = self.item_embedding(items)
        # concatenate user and item embeddings to form input
        x = torch.cat([user_embedding, item_embedding], 1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            x = self.fc_layers[idx](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.__dropout__, training=self.training)
        logit = self.output_layer(x)
        rating = torch.sigmoid(logit)
        return rating

    def predict(self, feed_dict):
        # return the score, inputs and outputs are numpy arrays
        for key in feed_dict:
            if type(feed_dict[key]) != type(None):
                feed_dict[key] = torch.from_numpy(feed_dict[key]).to(
                    dtype=torch.long, device=device
                )
        output_scores = self.forward(feed_dict)
        return output_scores.cpu().detach().numpy()

    def get_alias(self):
        return self.__alias__


def train_one_epoch(
    model, data_loader, loss_fn, optimizer, epoch_no, device, verbose=1
):
    "trains the model for one epoch and returns the loss"
    print("Epoch = {}".format(epoch_no))
    # Training
    # get user, item and rating data
    t1 = time()
    epoch_loss = []
    # put the model in train mode before training
    model.train()
    # transfer the data to GPU
    for feed_dict in data_loader:
        for key in feed_dict:
            if type(feed_dict[key]) != type(None):
                feed_dict[key] = feed_dict[key].to(dtype=torch.long, device=device)
        # get the predictions
        prediction = model(feed_dict)
        # get the actual targets
        rating = feed_dict["rating"]

        # convert to float and change dim from [batch_size] to [batch_size,1]
        rating = rating.float().view(prediction.size())
        loss = loss_fn(prediction, rating)
        # clear the gradients
        optimizer.zero_grad()
        # backpropagate
        loss.backward()
        # update weights
        optimizer.step()
        # accumulate the loss for monitoring
        epoch_loss.append(loss.item())
    epoch_loss = np.mean(epoch_loss)
    if verbose:
        print("Epoch completed {:.1f} s".format(time() - t1))
        print("Train Loss: {}".format(epoch_loss))
    return epoch_loss


def test(model, full_dataset: MovieLensDataset, topK):
    "Test the HR and NDCG for the model @topK"
    # put the model in eval mode before testing
    if hasattr(model, "eval"):
        # print("Putting the model in eval mode")
        model.eval()
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, full_dataset, topK)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print("Eval: HR = %.4f, NDCG = %.4f [%.1f s]" % (hr, ndcg, time() - t1))
    return hr, ndcg


def main():

    item_feature_file = open(
        "/Users/saleh/Documents/auto_encoder_vlad_64.csv", mode="r"
    )
    item_lines = item_feature_file.readlines()
    user_item_file = open("Data/vlad_u.train.rating", mode="r")

    users = []
    items = []

    for line in item_lines:
        line = line.strip("\n")
        item_vectors[line.split("-")[0]].append(
            line.rsplit(" ", feature_vector_size)[1 : feature_vector_size + 1]
        )

    u_i_lines = user_item_file.readlines()
    for line in u_i_lines:
        users.append(int(line.split("\t")[0]))
        items.append(int(line.split("\t")[1]))

    args = parse_args()
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)
    weight_decay = args.weight_decay
    num_negatives_train = args.num_neg_train
    num_negatives_test = args.num_neg_test
    dropout = args.dropout
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose

    topK = 10
    print("NCF arguments: %s " % (args))
    # model_out_file = 'Pretrain/%s_NCF_%s_%d.h5' %(args.dataset, args.layers, time())

    # Load data

    t1 = time()
    full_dataset = MovieLensDataset(
        path + dataset,
        num_negatives_train=num_negatives_train,
        num_negatives_test=num_negatives_test,
    )
    train, testRatings, testNegatives = (
        full_dataset.trainMatrix,
        full_dataset.testRatings,
        full_dataset.testNegatives,
    )
    num_users, num_items = train.shape
    print(
        "Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
        % (time() - t1, num_users, num_items, train.nnz, len(testRatings))
    )

    training_data_generator = DataLoader(
        full_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # Build model
    model = NCF(num_users, num_items, layers=layers, dropout=dropout)
    # Transfer the model to GPU, if one is available
    model.to(device)
    if verbose:
        print(model)

    loss_fn = torch.nn.BCELoss()
    # Use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)

    # Record performance
    hr_list = []
    ndcg_list = []
    BCE_loss_list = []

    # Check Init performance
    hr, ndcg = test(model, full_dataset, topK)
    hr_list.append(hr)
    ndcg_list.append(ndcg)
    # do the epochs now

    for epoch in range(epochs):
        epoch_loss = train_one_epoch(
            model, training_data_generator, loss_fn, optimizer, epoch, device
        )

        if epoch % verbose == 0:
            hr, ndcg = test(model, full_dataset, topK)
            hr_list.append(hr)
            ndcg_list.append(ndcg)
            BCE_loss_list.append(epoch_loss)

    print("hr for epochs: ", hr_list)
    print("ndcg for epochs: ", ndcg_list)
    print("loss for epochs: ", BCE_loss_list)

    best_iter = np.argmax(np.array(hr_list))
    best_hr = hr_list[best_iter]
    best_ndcg = ndcg_list[best_iter]
    print(
        "End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. "
        % (best_iter, best_hr, best_ndcg)
    )


if __name__ == "__main__":
    print("Device available: {}".format(device))
    main()
