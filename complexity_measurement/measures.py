import typing as t
import numpy as np
from sklearn.preprocessing import StandardScaler


def precompute_pca_tx(X: np.ndarray) -> t.Dict[str, t.Any]:
    prepcomp_vals = {}

    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    pca = PCA(n_components=0.95)
    pca.fit(X)

    m_ = pca.explained_variance_ratio_.shape[0]
    m = X.shape[1]
    n = X.shape[0]

    prepcomp_vals["m_"] = m_
    prepcomp_vals["m"] = m
    prepcomp_vals["n"] = n

    return prepcomp_vals


from complexity_measurement.overlapping import *
from complexity_measurement.neighborhood import *
from complexity_measurement.dimensionality import *
from complexity_measurement.balance import *


def compute_F1(X, Y):
    precompute = precompute_fx(X, Y)
    cls_index = precompute['cls_index']
    cls_n_ex = precompute['cls_n_ex']
    result = []
    res_sum = 0
    for j in range(len(cls_index)):
        # print("F1 score: ", ft_F1_paper(X, cls_index, cls_n_ex, j))
        numpy_res = cls_index[j] + 0
        if np.sum(numpy_res) == 0:
            result.append(0.99)
            # a=list(np.unique(np.array(0), return_counts=True)[0].min())
            # cls_n_ex.insert(j, np.unique(np.array(0), return_counts=True)[0].min())
            # cls_n_ex[0]=0
            # list(np.unique(np.array(0), return_counts=True)[0])
            # np.array(0)
            continue
        result.append(ft_F1_paper(X, cls_index, cls_n_ex, j))
    return result
