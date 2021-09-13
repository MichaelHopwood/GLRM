from numpy import ones, round, zeros, expand_dims, Inf, tile, arange, repeat, array
from functools import wraps
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from numpy.ma import masked_where
from numpy import maximum, minimum
import cvxpy as cp
import numpy as np

def pplot(As, titles):
    # setup
    try: vmin = min([A.min() for A, t in zip(As[:-1], titles) if "missing" not in t]) # for pixel color reference
    except: vmin = As[0].min()
    try: vmax = max([A.max() for A, t in zip(As[:-1], titles) if "missing" not in t])
    except: vmax = As[0].max()
    my_dpi = 96
    plt.figure(figsize=(1.4*(250*len(As))/my_dpi, 250/my_dpi), dpi = my_dpi)
    for i, (A, title) in enumerate(zip(As, titles)):
        plt.subplot(1, len(As), i+1)
        if i == len(As)-1: vmin, vmax = A.min(), A.max()
        if "missing" in title:
            missing = A
            masked_data = ones(As[i-1].shape)
            for j,k in missing:  masked_data[j,k] = 0
            masked_data = masked_where(masked_data > 0.5, masked_data)
            plt.imshow(As[i-1], interpolation = 'nearest', vmin = vmin, vmax = vmax)
            plt.colorbar()
            plt.imshow(masked_data, cmap = cm.binary, interpolation = "nearest")
        else:
            plt.imshow(A, interpolation = 'nearest', vmin = vmin, vmax = vmax)
            plt.colorbar()
        plt.title(title)
        plt.axis("off")
   
    plt.show()
# 
# def unroll_missing(missing, ns):
#     missing_unrolled = []
#     for i, (MM, n) in enumerate(zip(missing, ns)):
#         for m in MM:
#             n2 = m[1] + sum([ns[j] for j in range(i)])
#             missing_unrolled.append((m[0], n2))
#     return missing_unrolled
# 
def shrinkage(a, kappa):
    """ soft threshold with parameter kappa). """
    try: return maximum(a - kappa(ones(a.shape), 0)) - maximum(-a - kappa*ones(a.shape), 0)
    except: return max(a - kappa, 0) - max(-1 - kappa, 0)

def missing2mask(A_shape,missing_idx):
    """
    generates a mask from missing_idx
    
    input
    -----
    A_shape : shape
        Shape of the data matrix
    missing_idx : array
        shape (n_missing, 2) with first column the row and 2nd column the column of the missing index
        
    returns
    =------
    
    a mask with False where values are missing
    """
    
    mask = np.ones(A_shape,dtype=np.bool)
    mask[missing_idx[:,0],missing_idx[:,1]]=False
    return mask

def gen_random_missing_mask(A,n_missing,return_indices = False):
    """
    generates a mask + (optional) indices of the missing data
    """
    missing_idx =np.random.choice(np.arange(A.shape[0]),size=(n_missing,1),replace=False)
    missing_idx =np.hstack([missing_idx,np.random.randint(0,A.shape[1],size=(n_missing,1))])
    
    mask = np.ones_like(A,dtype=np.bool)
    mask[missing_idx[:,0],missing_idx[:,1]]=False
    if return_indices:
        return mask, missing_idx
    else:
        return mask
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def oneHot(A,columns,max_index = None,):
    """
    does one hot encoding of the categorical values
    
    Assumes that the categoricals are already integers starting from 0
    A : data matrix
    columns : iterable of the columns that contain categoricals
    max_index: max value of classes , if None will be inferred. If not None
        then needs to be the same shape as oclumns
        
    could probably also call the sklearn onehotencoder
    """
    columns = np.asarray(columns)
    A_col = A[:,columns].astype(np.int)
    if max_index is None:
        max_index = A_col.max(axis=0)
#         print(max_index)
    elif len(max_index) != len(columns):
        raise ValueError('Yikes len(max_index) != len(columns)')
    else:
        max_index = np.asarray(max_index).astype(np.int)
#     print(max_index)
    m = A_col.shape[0]
    one_hot = []
    for i in range(len(columns)):
        one_hot.append(np.zeros(m,max_index[i]+1))
        mat_size = max_index[i]+1
        arr = -np.ones((mat_size,mat_size))
        arr[np.diag_indices_from(arr)]=1
#         print(arr)
        one_hot[-1] = arr[A_col[:,i]]
    return one_hot

def oneHotTransform(A,columns=None,max_index = None):
    """
    does one hot encoding of the categorical values and transforms the array
    """
    if columns is None:
        columns = np.arange(A.shape[1])
    else:
        columns = np.asarray(columns)
    one_hot = oneHot(A,columns,max_index)
    #stack the onehot stuff 
    return np.hstack([A[:,i][:,None] if not i in columns else one_hot[np.abs(columns-i).argmax()] for i in range(A.shape[1])])