import networkx as nx
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.metrics import jaccard_score
import os
import math
import pickle

def create_data():
    # 5 points
    df_1 = pd.read_csv('hand_built_anomaly.csv')
    #195 points
    df_2 = pd.read_csv('anomaly_syhthetic_1.csv')
    print(len(df_2))
    print(len(df_1))

    df = df_1.append(df_2, ignore_index=True)
    cols = list(df.columns)
    cols_dict = {e[1]:e[0] for e in enumerate(cols,0)}

    def baptize(
            row,
            col,
            cols_dict
    ):
        _i = cols_dict[col]

        return str(_i) + '_'+ str(row[col])

    for col in cols :
        print('--',col)
        df[col] = df.apply(
            baptize,
            axis=1,
            args=( col, cols_dict,)
        )

    df['id'] = list(df.index)
    _anomalies = list(range(195,200))
    return df, _anomalies

'''
A_ft: F x T
'''
def get_A_ft(graph, F, T, F_id_dict ):
    _F_id_dict = {
        v:k for k,v in F_id_dict.items()
    }
    a = np.zeros([len(F),len(T)])
    for e in graph.edges():

        i = None
        j = None

        if ((e[0] in F) and (e[1] in T)):
            j = _F_id_dict[e[0]]
            i = e[1]

        elif ((e[0] in T) and (e[1] in F))  :
            i = _F_id_dict[e[1]]
            j = e[0]

        if i is not None and j is not None:
            a[i][j] = 1

    return a

# ------------------------ #
# D_f is the matrix that captures both feature interaction & feature importance(rarity)
# ------------------------ #
def get_D_f(
        df,
        feature_entities,
        feature_entities_id
):
    feature_cols = list(df.columns)
    feature_cols.remove('id')
    _F_id_dict = {
        v: k for k, v in feature_entities_id.items()
    }
    num_features = len(feature_entities)
    # idf is 1/p
    idf = np.zeros([num_features])

    for col in feature_cols:
        if col =='id':
            continue
        _prob = Counter(list(df[col]))
        print(_prob)
        for k,v in _prob.items():
            idx = _F_id_dict[k]
            p =(v+1)/len(df)
            idf[idx] = 1/math.log(p,math.e)

    idf = np.reshape(normalize(np.reshape(idf, [1, -1]), norm='max'), [-1])
    # calculate co-occurrence matrix
    coocc_matrix = np.zeros([num_features,num_features],dtype=np.float32)

    for uv in itertools.combinations(feature_cols,2):
        print('---',uv)
        tmp_df = df[list(uv)]
        tmp_df_1 = pd.DataFrame(tmp_df.groupby(uv).size().reset_index(
            name='counts')
        )
        for i,row in tmp_df_1.iterrows():
            u = _F_id_dict[row[0]]
            v = _F_id_dict[row[1]]
            coocc_matrix[u][v] = int(row['counts'])


    a = np.matmul(np.transpose(idf),idf)
    b = coocc_matrix

    # element wise multiplication
    c = a * b

    # set the diagonal elements to be idf^2
    for i in range(c.shape[0]):
        c[i][i] = idf[i]*idf[i]

    return c

# ---------------------------------- #
def show_heatmap(arr):
    if len(arr.shape) == 1:
        arr = np.reshape(arr,[-1,1])

    plt.imshow(arr, cmap='hot', interpolation='nearest')
    plt.tight_layout()
    plt.show()
    plt.close()

def get_A_t(
        df,
        t_node_ids
):
    num_t_nodes = len(t_node_ids)
    A_t = np.zeros(
        [num_t_nodes,num_t_nodes]
    )
    # simple jaccard simailarity

    for t1 in  t_node_ids:
        tmp_df = df.loc[df['id'] == t1]
        del tmp_df['id']
        s1_idx = tmp_df.index.to_list()[0]
        s1 = tmp_df.loc[[s1_idx]].values[0]

        for t2 in t_node_ids:
            if t1 == t2 :
                continue

            tmp_df1 = df.loc[df['id'] == t2]
            del tmp_df1['id']
            s2_idx = tmp_df1.index.to_list()[0]
            s2 = tmp_df1.loc[[s2_idx]].values[0]

            sim = len(set(s1).intersection(set(s2)))/ len(set(s1).union(set(s2)))
            A_t[t1][t2] = sim
    print('A_t', A_t)
    return A_t

def func_1(refresh = True ):
    if os.path.exists('datapkl_1.pkl') and refresh==False :
        with open('datapkl_1.pkl', 'rb') as fh:
            result = pickle.load(fh)
            df = result[0]
            _anomalies = result[1]
            A_ft = result[2]
            D_f = result[3]
            W_f = result[4]
            W_t = result[5]
            A_t = result[6]
            g = result[7]
            return df, _anomalies, A_ft, D_f, W_f, W_t, A_t

    df, _anomalies = create_data()
    g = nx.Graph()

    # T nodes
    t_node_ids = list(df['id'])
    print(t_node_ids)
    g.add_nodes_from(t_node_ids)

    # feature nodes
    feature_entities = []
    for col in list(df.columns):
        if col =='id' : continue
        feature_entities.extend(list(df[col]))

    feature_entities = list(set(feature_entities))

    # id : feature_name
    feature_entities_id = {
        e[0]:e[1]
        for e in enumerate(feature_entities,0)
    }

    g.add_nodes_from(feature_entities)
    print(g.number_of_nodes())
    for i,row in df.iterrows():
        _nodes = list(row)
        for u_v in itertools.combinations(_nodes, 2):
            g.add_edge(u_v[0],u_v[1], type=1)
    print(g.number_of_edges())
    print(g.number_of_nodes())

    A_ft = get_A_ft(g,feature_entities,t_node_ids,feature_entities_id)

    # -------------- #
    # set the weights of the graph nodes
    # -------------- #

    W_t = np.zeros([len(t_node_ids)])
    for i in _anomalies:
        W_t[i] = 1
    show_heatmap(W_t)
    W_f = np.zeros([len(feature_entities)])
    D_f =  get_D_f(df,feature_entities,feature_entities_id)
    A_t = get_A_t(df, t_node_ids)
    result = [df,_anomalies, A_ft, D_f, W_f, W_t,A_t,g]
    with open('datapkl_1.pkl','wb') as fh:
         pickle.dump(result,fh,pickle.HIGHEST_PROTOCOL)

    return df,_anomalies ,A_ft, D_f, W_f, W_t,A_t, g


def func_2():

    df, _anomalies, A_ft, D_f, W_f, W_t,A_t = func_1(False)
    iter = 0
    epsilon = 0.0001
    _beta = 2
    # iteration of dqe
    W_t_orig = W_t
    E = np.matmul(np.transpose(A_ft), np.matmul(D_f, A_ft)) + _beta * (A_t)
    E = normalize(
        E,
        axis=0,
        norm='l1'
    )


    while True:
        W_t_new = np.matmul(
            E,
            W_t
        )

        #update:
        for i in range(len(W_t)):
            if W_t_orig[i] == 1:
                W_t_new[i] = 1
            else:
                W_t_new[i] = min(W_t_new[i],1)

        delta1 = (np.max(W_t_new - W_t))
        W_t = W_t_new

        if delta1 < epsilon :
            break
        iter += 1
        if iter%200 == 0 :
            show_heatmap(W_t)
            show_heatmap(W_f)
        print(iter)

        if iter>1000:
            break

    show_heatmap(W_t)
    show_heatmap(W_f)

    print(W_f)
    print(W_t)


def func_3():
    df,_anomalies, A_ft, D_f, W_f, W_t, A_t = func_1(True)

    iter = 0
    epsilon = 0.0001
    _beta = 2
    # iteration of dqe
    W_t_orig = W_t
    E = np.matmul(np.transpose(A_ft), np.matmul(D_f, A_ft)) + _beta * (A_t)
    E = normalize(
        E,
        axis=0,
        norm='l1'
    )


    set_labelled = list(_anomalies)
    while True:
        W_t_new = np.matmul(
            E,
            W_t
        )

        min_l1 = 10
        min_l1_idx = -1
        for _j in set_labelled:
            if W_t_new[_j] < min_l1:
                min_l1 = W_t_new[_j]
                min_l1_idx = _j

        _l2 = []
        max_l2 = -1
        max_l2_idx = None
        for _j in range(W_t.shape[0]):
            if _j not in set_labelled:
                if W_t_new[_j] > max_l2:
                    max_l2_idx = _j
                    max_l2 = W_t_new[_j]


        print(min_l1, max_l2)
        set_labelled.remove(min_l1_idx)
        set_labelled.append(max_l2_idx)

        delta1 = (W_t_new[min_l1_idx] - W_t_new[max_l2_idx])
        W_t = W_t_new

        if delta1 < epsilon:
            break

        iter += 1
        print(iter)

        if iter > 1000:
            break

    show_heatmap(W_t)

    print(W_f)
    print(W_t)


def func_4():

    df, _anomalies, A_ft, D_f, W_f, W_t, A_t,g = func_1(True)



    return


func_4()