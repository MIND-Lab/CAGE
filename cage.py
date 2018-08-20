'''
Created on Nov, 2016

@author: hugo

'''

from time import time
import numpy as np
import networkx as nx
import pickle as pickle
import csv
from tabulate import tabulate
import argparse
import os
import sys
import itertools

from autoencoder.datasets.dblp import construct_train_test_corpus
from autoencoder.utils.io_utils import dump_json

from static_graph_embedding import StaticGraphEmbedding

from autoencoder.core.ae import fit_quadruple_hyperas
from autoencoder.preprocessing.preprocessing import load_corpus, doc2vec
from autoencoder.utils.op_utils import vecnorm
from utils import graph_util

from evaluation import evaluation_classification_original as ec

from evaluation import compute_tasks as ct

path_source = "..//"


class CAGE(StaticGraphEmbedding):

    def __init__(self, *hyper_dict, **kwargs):
        ''' Initialize the LaplacianEigenmaps class

        Args:
            d: dimension of the embedding
        '''
        hyper_params = {
            'method_name': 'lap_eigmap_svd'
        }
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])
        for dictionary in hyper_dict:
            for key in dictionary:
                self.__setattr__('_%s' % key, dictionary[key])

    def get_node_representation(self, G, X_data, dict_id):

        def edge_to_vector_a(edge):
            (a, b) = edge
            #        return X_data[X_idx.index(dict_nodes[a])]
            if (dict_id is None):
                X_batch_v_i = X_data[(a)]
            else:
                X_batch_v_i = X_data[(dict_id[a])]

            return X_batch_v_i

        def edge_to_vector_b(edge):
            (a, b) = edge
            #        return X_data[X_idx.index(dict_nodes[b])]
            if (dict_id is None):
                X_batch_v_j = X_data[(b)]
            else:
                X_batch_v_j = X_data[(dict_id[b])]

            return X_batch_v_j

        graph_edges_tuple = G.edges()
        graph_edges_vector_a = np.array(list(map(edge_to_vector_a, graph_edges_tuple)), dtype=np.float32)
        graph_edges_vector_b = np.array(list(map(edge_to_vector_b, graph_edges_tuple)), dtype=np.float32)
        graph_edges_vector = [graph_edges_vector_a, graph_edges_vector_b]

        return graph_edges_vector

    def learn_embedding(self, graph=None, edge_f=None, is_weighted=False, no_python=False, path_output="", dataset=""):

        n_dim = self._d
        method = "sdne"
        input = path_output + '/train.corpus'
        path_graph_embedding = path_source + "embedding/" + dataset + "/embedding_gem_sdne_" + dataset + "_" + str(
            n_dim) + ".txt"
        path_graph_embedding_id = path_source + "embedding/" + dataset + "/id_gem_" + method + "_" + dataset + "_" + str(
            n_dim) + ".txt"

        save_model = 'model'
        optimizer = "adadelta"

        val_split = 0.0214

        batch_size = self._batch_size
        comp_topk = self._comp_topk
        optimizer = self._optimizer
        lr = self._lr
        alpha = self._alpha
        kfactor = self._kfactor
        gamma = self._gamma
        select_diff = self._select_diff
        select_loss = self._select_loss
        select_graph_np_diff = self._select_graph_np_diff

        contractive = None
        ctype = "kcomp"
        n_dim = 128
        nb_epoch = 1000
        save_model = 'model'

        if not graph and not edge_f:
            raise Exception('graph/edge_f needed')
        if not graph:
            graph = graph_util.loadGraphFromEdgeListTxt(edge_f)
        num_nodes = graph.number_of_nodes()
        graph3 = nx.DiGraph()
        graph3.add_nodes_from(range(0, num_nodes))
        f1 = csv.reader(open(edge_f, "r"), delimiter=' ')
        for x, y in f1:
            # print(x,y)
            graph3.add_edge(int(x), int(y))
        S = nx.to_scipy_sparse_matrix(graph, nodelist=sorted(graph.nodes()))
        t1 = time()
        S = (S + S.T) / 2
        node_num = graph.number_of_nodes()
        edges_num = graph.number_of_edges()
        dict_nodes = {k: v for v, k in enumerate(sorted(graph.nodes()))}

        ## Load Graph Embeddings
        if (path_graph_embedding.endswith(".txt")):
            print("Loading SDNE embeddings")
            graph_embeddings = np.loadtxt(path_graph_embedding, delimiter=',')
            with open(path_graph_embedding_id) as temp_file:
                graph_embedding_id = [line.rstrip('\n') for line in temp_file]
            dict_graph = {k: v for v, k in enumerate(graph_embedding_id)}

        else:
            raise Exception('sdne embeddings do not exist')
            graph_embeddings = pickle.load(open(path_graph_embedding, "rb"))

        ## Load text data
        print("Loading textual corpus")
        corpus = load_corpus(input)
        n_vocab = len(corpus['vocab'])
        docs = corpus['docs']
        corpus.clear()  # save memory
        doc_keys = np.array(list(docs))
        dict_doc = {int(k): v for v, k in enumerate((doc_keys))}

        X_docs = []
        for k in list(docs):
            X_docs.append(vecnorm(doc2vec(docs[k], n_vocab), 'logmax1', 0))
            del docs[k]
        X_docs = np.r_[X_docs]
        # dump_json(dict(zip(doc_keys.tolist(), X_docs.tolist())), path_source+'embedding\\'+dataset+'\\bow.txt')

        text_vector = self.get_node_representation(graph, X_docs, dict_doc)
        graph_vector = self.get_node_representation(graph, graph_embeddings, dict_nodes)

        # return S,node_num,edges_num,graph_embeddings, X_docs,n_vocab, doc_keys, text_vector, graph_vector

        train_data = [text_vector, text_vector, graph_vector]

        result, _Y, model = fit_quadruple_hyperas(n_vocab, n_dim, comp_topk=comp_topk, ctype=ctype,
                                                  save_model=save_model,
                                                  kfactor=kfactor, alpha=alpha, gamma=gamma, num_nodes=node_num,
                                                  num_edges=edges_num,
                                                  train_data=train_data, test_data=X_docs, val_split=val_split,
                                                  nb_epoch=nb_epoch, \
                                                  batch_size=batch_size, contractive=contractive, optimizer=optimizer,
                                                  lr=lr,
                                                  select_diff=select_diff, select_loss=select_loss,
                                                  select_graph_np_diff=select_graph_np_diff)

        dump_json(dict(zip(doc_keys.tolist(), _Y.tolist())),
                  path_source + 'embedding\\' + dataset + '\\predicted_cage_embedding.txt')
        print('Saved doc codes file')

        self._Y = _Y
        self._node_num = node_num
        self._X = X_docs
        _Y_id = doc_keys.tolist()

        return _Y, _Y_id, len(result.history["loss"]), t1

    def get_embedding(self, filesuffix=None):
        return self._Y if filesuffix is None else np.loadtxt(
            'embedding_' + filesuffix + '.txt'
        )

    def get_edge_weight(self, i, j):
        return np.exp(
            -np.power(np.linalg.norm(self._Y[i] - self._Y[j]), 2)
        )

    def get_reconstructed_adj(self, X=None, node_l=None):
        if X is not None:
            node_num = X.shape[0]
            self._X = X
        else:
            node_num = self._node_num
        adj_mtx_r = np.zeros((node_num, node_num))
        for v_i in range(node_num):
            for v_j in range(node_num):
                if v_i == v_j:
                    continue
                adj_mtx_r[v_i, v_j] = self.get_edge_weight(v_i, v_j)
        return adj_mtx_r


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--edgelist', type=str, help='path to the edgelist file')
    parser.add_argument('-t', '--text', type=str, help='path to the label text file')
    parser.add_argument('-d', type=int, help='embedding dimension')
    args = parser.parse_args(arguments)
    main_method(args.edgelist, args.text, args.d)


def main_method_arg(edge_f, path_text, d,
                batch_size, comp_topk, optimizer, lr, alpha, kfactor, gamma,
                select_diff, select_loss, select_graph_np_diff):
    method = "cage"
    path_source = "../"
    path_output = "../"
    test_ratio_arr = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    rounds = 5
    dataset = edge_f[(edge_f.rfind("/") + 1):(edge_f.find(".edgelist"))]
    path_output = path_text[:path_text.rfind("/") + 1]

    if not (os.path.exists(path_output + "train.corpus")):
        construct_train_test_corpus(path_text, 0.0, path_output, threshold=10, topn=2000)

    G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=False)
    t1 = time()

    info = "btc_" + str(batch_size) + "_tk_" + str(comp_topk) + "_lr_" + str(lr) + "_opt_" + optimizer + "_a_" + str(
        alpha) + "_k_" + str(kfactor) + "_g_" + str(gamma) + \
           "_s1_" + str(select_diff) + "_s2_" + str(select_loss) + "_s3_" + str(select_graph_np_diff)
    print("--info--- ", info)

    path_embedding = path_source + "embedding/" + dataset + "/" + method + "/embedding_gem_" + method + "_" + dataset + "_" + str(
        d) + "_" + info + ".txt"
    path_embedding_id = path_source + "embedding/" + dataset + "/" + method + "/id_gem_" + method + "_" + dataset + "_" + str(
        d) + "_" + info + ".txt"
    if (os.path.exists(path_embedding)):
        print("exist")
        embedding_vector = np.loadtxt(path_embedding, delimiter=",")
        with open(path_embedding_id) as temp_file:
            embedding_id = [line.rstrip('\n') for  line in temp_file]
        embedding = CAGE(d=d, X=embedding_vector)
    else:
        embedding = CAGE(d=d, batch_size=batch_size, comp_topk=comp_topk, optimizer=optimizer, lr=lr, alpha=alpha,
                         kfactor=kfactor, gamma=gamma,
                         select_diff=select_diff, select_loss=select_loss, select_graph_np_diff=select_graph_np_diff)
        embedding_vector, embedding_id, n_epochs_done, t = embedding.learn_embedding(graph=None, edge_f=edge_f,
                                                                                     is_weighted=True, no_python=True,
                                                                                     path_output=path_output,
                                                                                     dataset=dataset)
        ec.saveEmbeddings(embedding_vector, embedding_id, path_source, dataset, method, d, info, n_epochs_done)

        print('CAGE:\n\tTraining time: %f' % (time() - t1))

        Y = ec.loadLabels(dataset, embedding_id, path_input=path_source)
        path_result = path_source + "results/" + dataset + "/" + method

        path_train_test = '../data/' + dataset + '/' + dataset + '_train_test.txt'
        ec.expNC(embedding_vector, test_ratio_arr, "lr",
                 rounds, path_result, method + info, dataset, embedding_id, path_source)

    ct.computeTasks(embedding=embedding_vector, embedding_model=embedding, dataset=dataset, method=method, d=d, G=G,
                    path_output=path_output, path_source=path_source)

def main_method(edge_f,path_text,d,dataset, verbose = 0):

    #dataset = "dcc"
    # dataset = "mcc"
    #d = 128
    #edge_f = path_source + 'data/' + dataset + '/' + dataset + '.edgelist'
    #path_text = path_source + 'data/' + dataset + '/' + dataset + '_text.txt'


    if (dataset == "dcc"):
        kfactors = [3]
        select_diffs = [0]
        select_losss = [5]
        select_graph_np_diffs = [0]

        batches = [50]
        topks = [128]
        optimizers = ["adadelta"]
        lrs = [6.]
        alphas = [1e-4]
        gammas = [1e-4]

    if (dataset == "mcc"):
        kfactors = [1]
        select_diffs = [3]
        select_losss = [0]
        select_graph_np_diffs = [0]

        batches = [50]
        topks = [128]
        optimizers = ["adam"]
        lrs = [0.1]
        alphas = [1e-4]
        gammas = [1]
    for s1, s2, s3, a, k, g, b, t, o, l in list(itertools.product(
            select_diffs, select_losss, select_graph_np_diffs,
            alphas, kfactors, gammas, batches, topks, optimizers, lrs,
    )):
        main_method_arg(edge_f, path_text, d, b, t, o, l, a, k, g, s1, s2, s3)


