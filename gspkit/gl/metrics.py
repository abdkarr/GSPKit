import numpy as np
import numpy.typing as npt

from sklearn.metrics import (f1_score, average_precision_score, accuracy_score,
                             normalized_mutual_info_score, precision_score, 
                             recall_score)

def _to_mv(data):
    # TODO Testing
    if not isinstance(data, list):
        data = [data]

    return data

def _one(func, data):
    # TODO Testing
    data = _to_mv(data)

    n_views = len(data)
    result = np.zeros(n_views)

    for v in range(n_views):
        result[v] = func(data[v])

    return result if n_views>1 else result[0]

def _one_to_one(func, data1, data2):
    # TODO Testing
    data1 = _to_mv(data1)
    data2 = _to_mv(data2)

    n_views = len(data1)
    result = np.zeros(n_views)

    for v in range(n_views):
        result[v] = func(data1[v], data2[v])

    return result if n_views>1 else result[0]

def _one_to_all(func, data1, data2):
    # TODO Testing
    data1 = _to_mv(data1)
    data2 = _to_mv(data2)

    n_views1 = len(data1)
    n_views2 = len(data2)
    result = np.zeros((n_views1, n_views2))
    for v1 in range(n_views1):
        for v2 in range(n_views2):
            result[v1, v2] = func(data1[v1], data2[v2])

    if n_views1 == 1 and n_views2 == 1:
        return result[0, 0]
    else:
        return result.squeeze()

def density(w: list[npt.NDArray] | npt.NDArray) -> float | npt.NDArray:
    r"""Calculate density of a given set of graphs.

    Parameters
    ----------
    w : npt.NDArray or list of npt.NDArray
        Upper triangular part of the adjacency matrices of the graphs. If not 
        a list, it is assumed that a single graph is given.

    Returns
    -------
    float or npt.NDArray
        Graph densities. If a single graph is given, a single value is returned.
    """

    # TODO Testing

    def _density(d):
        return np.count_nonzero(d)/len(d)

    return _one(_density, w)

def correlation(
        w1: list[npt.NDArray] | npt.NDArray, 
        w2: list[npt.NDArray] | npt.NDArray
    ) -> float | npt.NDArray:
    r"""Calculate correlation between two sets of graphs.

    Given two set of graphs, :math:`\mathcal{G} = \{G^i\}_{i=1}^N`, and
    :math:`\mathcal{H} = \{H^i\}_{i=1}^M`, this function calculates correlation
    between pair of graphs, :math:`(G^i, H^j)` for all :math:`i` and \math:`j`.

    Parameters
    ----------
    w_1 : npt.NDArray or list of npt.NDArray
        Upper triangular part of the adjacency matrices of the graphs in first
        set. If np.array, it is assumed that :math:`N=1`.
    w_2 : npt.NDArray or list of npt.NDArray
        Upper triangular part of the adjacency matrices of the graphs in second
        set. If np.array, it is assumed that :math:`M=1`.

    Returns
    -------
    float or npt.NDArray
        (N, M) dimensional array of correlation. If :math:`N=1`, it is (M,)
        dimensional. If :math:`M=1`, it is (N,) dimensional. If :math:`N=1` and 
        :math:`M=1`, it is a single value.  
    """

    # TODO Testing

    def _correlation(d1, d2):
        return np.corrcoef(np.squeeze(d1), np.squeeze(d2))[0,1]

    return _one_to_all(_correlation, w1, w2)

def f1(
        w_gt: list[npt.NDArray] | npt.NDArray, 
        w_hat: list[npt.NDArray] | npt.NDArray
    ) -> float | npt.NDArray:
    r"""Calculate F1-score between ground truth and learned graphs. 

    Given :math:`N` ground truth graphs, :math:`\mathcal{G} = \{G^i\}_{i=1}^N`, 
    and learned graphs :math:`\widehat{\mathcal{G}} = \{\widehat{G}^i\}_{i=1}^N`, 
    this function calculates :math:`F1(G^i, \widehat{G}^i)` for each :math:`i`. 

    Parameters
    ----------
    w_gt : npt.NDArray or list of npt.NDArray
        Upper triangular part of the adjacency matrices of ground truth graphs.
        If not a list, it is assumed that :math:`N=1`.
    w_hat : npt.NDArray or list of npt.NDArray
        Upper triangular part of the adjacency matrices of learned graphs.
        If not a list, it is assumed that :math:`N=1`.
    
    Returns
    -------
    float or npt.NDArray
        Calculated F1 scores. If :math:`N=1`, it is a single F1-score.
    """

    # TODO Testing

    def _f1(w1, w2):
        return f1_score((w1 > 0).astype(int).squeeze(), 
                        (w2 > 0).astype(int).squeeze())

    return _one_to_one(_f1, w_gt, w_hat)

def auprc(
        w_gt: list[npt.NDArray] | npt.NDArray, 
        w_hat: list[npt.NDArray] | npt.NDArray
    ) -> float | npt.NDArray:
    r"""Calculate AUPRC score between ground truth and learned graphs. 

    Given :math:`N` ground truth graphs, :math:`\mathcal{G} = \{G^i\}_{i=1}^N`, 
    and learned graphs :math:`\widehat{\mathcal{G}} = \{\widehat{G}^i\}_{i=1}^N`, 
    this function calculates :math:`AUPRC(G^i, \widehat{G}^i)` for each :math:`i`. 

    Parameters
    ----------
    w_gt : npt.NDArray or list of npt.NDArray
        Upper triangular part of the adjacency matrices of ground truth graphs.
        If not a list, it is assumed that :math:`N=1`.
    w_hat : npt.NDArray or list of npt.NDArray
        Upper triangular part of the adjacency matrices of learned graphs.
        If not a list, it is assumed that :math:`N=1`.
    
    Returns
    -------
    float or npt.NDArray
        Calculated AUPRC scores. If :math:`N=1`, it is a single AUPRC score.
    """

    # TODO Testing

    def _auprc(w1, w2):
        return average_precision_score(
            (w1 > 0).astype(int).squeeze(), w2.squeeze()
        )

    return _one_to_one(_auprc, w_gt, w_hat)

def accuracy(
        w_gt: list[npt.NDArray] | npt.NDArray, 
        w_hat: list[npt.NDArray] | npt.NDArray
    ) -> float | npt.NDArray:
    r"""Calculate accuracy score between ground truth and learned graphs. 

    Given :math:`N` ground truth graphs, :math:`\mathcal{G} = \{G^i\}_{i=1}^N`, 
    and learned graphs :math:`\widehat{\mathcal{G}} = \{\widehat{G}^i\}_{i=1}^N`, 
    this function calculates :math:`Accuracy(G^i, \widehat{G}^i)` for each :math:`i`. 

    Parameters
    ----------
    w_gt : npt.NDArray or list of npt.NDArray
        Upper triangular part of the adjacency matrices of ground truth graphs.
        If not a list, it is assumed that :math:`N=1`.
    w_hat : npt.NDArray or list of npt.NDArray
        Upper triangular part of the adjacency matrices of learned graphs.
        If not a list, it is assumed that :math:`N=1`.
    
    Returns
    -------
    float or npt.NDArray
        Calculated accuracy scores. If :math:`N=1`, it is a single score.
    """

    # TODO Testing

    def _accuracy(w1, w2):
        return accuracy_score(
            (w1 > 0).astype(int).squeeze(), (w2 > 0).astype(int).squeeze()
        )

    return _one_to_one(_accuracy, w_gt, w_hat)

def nmi(
        w_gt: list[npt.NDArray] | npt.NDArray, 
        w_hat: list[npt.NDArray] | npt.NDArray
    ) -> float | npt.NDArray:
    r"""Calculate NMI between ground truth and learned graphs. 

    Given :math:`N` ground truth graphs, :math:`\mathcal{G} = \{G^i\}_{i=1}^N`, 
    and learned graphs :math:`\widehat{\mathcal{G}} = \{\widehat{G}^i\}_{i=1}^N`, 
    this function calculates :math:`NMI(G^i, \widehat{G}^i)` for each :math:`i`. 

    Parameters
    ----------
    w_gt : npt.NDArray or list of npt.NDArray
        Upper triangular part of the adjacency matrices of ground truth graphs.
        If not a list, it is assumed that :math:`N=1`.
    w_hat : npt.NDArray or list of npt.NDArray
        Upper triangular part of the adjacency matrices of learned graphs.
        If not a list, it is assumed that :math:`N=1`.
    
    Returns
    -------
    float or npt.NDArray
        Calculated NMI values. If :math:`N=1`, it is a single value.
    """

    # TODO Testing

    def _nmi(w1, w2):
        return normalized_mutual_info_score(
            (w1 > 0).astype(int).squeeze(), (w2 > 0).astype(int).squeeze()
        )

    return _one_to_one(_nmi, w_gt, w_hat)

def precision(
        w_gt: list[npt.NDArray] | npt.NDArray, 
        w_hat: list[npt.NDArray] | npt.NDArray
    ) -> float | npt.NDArray:
    r"""Calculate precision score between ground truth and learned graphs. 

    Given :math:`N` ground truth graphs, :math:`\mathcal{G} = \{G^i\}_{i=1}^N`, 
    and learned graphs :math:`\widehat{\mathcal{G}} = \{\widehat{G}^i\}_{i=1}^N`, 
    this function calculates :math:`Precision(G^i, \widehat{G}^i)` for each :math:`i`. 

    Parameters
    ----------
    w_gt : npt.NDArray or list of npt.NDArray
        Upper triangular part of the adjacency matrices of ground truth graphs.
        If not a list, it is assumed that :math:`N=1`.
    w_hat : npt.NDArray or list of npt.NDArray
        Upper triangular part of the adjacency matrices of learned graphs.
        If not a list, it is assumed that :math:`N=1`.
    
    Returns
    -------
    float or npt.NDArray
        Calculated precision scores. If :math:`N=1`, it is a single score.
    """

    # TODO Testing

    def _precision(w1, w2):
        return precision_score(
            (w1 > 0).astype(int).squeeze(), (w2 > 0).astype(int).squeeze()
        )

    return _one_to_one(_precision, w_gt, w_hat)

def recall(
        w_gt: list[npt.NDArray] | npt.NDArray, 
        w_hat: list[npt.NDArray] | npt.NDArray
    ) -> float | npt.NDArray:
    r"""Calculate recall score between ground truth and learned graphs. 

    Given :math:`N` ground truth graphs, :math:`\mathcal{G} = \{G^i\}_{i=1}^N`, 
    and learned graphs :math:`\widehat{\mathcal{G}} = \{\widehat{G}^i\}_{i=1}^N`, 
    this function calculates :math:`Recall(G^i, \widehat{G}^i)` for each :math:`i`. 

    Parameters
    ----------
    w_gt : npt.NDArray or list of npt.NDArray
        Upper triangular part of the adjacency matrices of ground truth graphs.
        If not a list, it is assumed that :math:`N=1`.
    w_hat : npt.NDArray or list of npt.NDArray
        Upper triangular part of the adjacency matrices of learned graphs.
        If not a list, it is assumed that :math:`N=1`.
    
    Returns
    -------
    float or npt.NDArray
        Calculated recall scores. If :math:`N=1`, it is a single score.
    """

    # TODO Testing

    def _recall(w1, w2):
        return recall_score(
            (w1 > 0).astype(int).squeeze(), (w2 > 0).astype(int).squeeze()
        )

    return _one_to_one(_recall, w_gt, w_hat)

def recovery_error(
        w_gt: list[npt.NDArray] | npt.NDArray, 
        w_hat: list[npt.NDArray] | npt.NDArray, 
        norm: str="l2",
    ) -> float | npt.NDArray:
    r"""Calculate recovery error between ground truth and learned graphs. 

    Given :math:`N` ground truth graphs, :math:`\mathcal{G} = \{G^i\}_{i=1}^N`,
    and learned graphs :math:`\widehat{\mathcal{G}} =
    \{\widehat{G}^i\}_{i=1}^N`, this function calculates recovery error between
    :math:`G^i` and :math:`\widehat{G^i} for each :math:`i` as follows:

    .. math::

        RE(\mathbf{w}_0, \mathbf{w}) = \frac{||\cdot \mathbf{w}_0 -
        \mathbf{w}||_p}{||\mathbf{w}_0||_p}, 

    where :math:`\mathbf{w}_0` and :math:`\mathbf{w}` are strictly upper
    triangular part of ground truth and learned adjacency matrix, respectively.
    :math:`||\cdot||_p` is the norm to use. 

    Parameters
    ----------
    w_gt : npt.NDArray or list of npt.NDArray
        Upper triangular part of the adjacency matrices of ground truth graphs.
        If not a list, it is assumed that :math:`N=1`.
    w_hat : np.NDArray or list of np.NDArray
        Upper triangular part of the adjacency matrices of learned graphs. If
        not a list, it is assumed that :math:`N=1`.
    norm: str, optional
        Norm to use when calculating the recovery error. Either "l2" or "l1".
        By defualt "l2".
    
    Returns
    -------
    float or npt.NDArray
        Calculated recovery errors. If :math:`N=1`, it is a single score.
    """

    # TODO Testing

    def _recovery_error(w1, w2):
        ord = 2 if norm == "l2" else 1
        return np.linalg.norm(
            (w1 > 0).astype(int).squeeze() - (w2 > 0).astype(int).squeeze(), ord
        )/np.linalg.norm((w1 > 0).astype(int).squeeze(), ord)

    return _one_to_one(_recovery_error, w_gt, w_hat)