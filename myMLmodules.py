"""
Created on 03 mag 2017

@author: Alessandro
"""
import numpy as np
import time


def split_data(data, labels, data_rates=1,
               val_rate=0.2, test_rate=0.1, split_index=None,
               shuffling=True, train_balancing=True, val_balancing=True, test_balancing=True):
    """ Loading dataset from extracted features

    dataSet    :   #samples-by-#features numpy array data
    labels     :   #samples-by-label numpy array data
    data_rates  :   data reduction rate
    val_rate    :   validation/train rate
    test_rate   :   test/train rate]
    split_index :   dictionary contains indexes for train/validation/test
    balancing  :   balancing data classes-wise

    OUTPUT: T (training set), V (validation set), E (test set)
    """
    
    print('\n#   #   #   Data Splitting   #   #   #')
    print('   Splitting rates:')
    print('        Test:'+str(test_rate))
    print('        Validation:'+str(val_rate))
    
    # possible data random shuffling       
    if shuffling:
        print('\n#   #   #   Shuffling Data   #   #   #')
        p = np.random.permutation(data.shape[0])
        data = data[p]
        labels = labels[p]
        
    # possible data reduction
    if data_rates < 1:
        print('\n#   #   #   Reducing Data   #   #   #')
        print('         Data rate: ' + str(data_rates))
        nd = int(data.shape[0] * data_rates)
        x = data[:nd]
        y = labels[:nd]
    else:
        x = data
        y = labels
    
    v_s = int(x.shape[0] * val_rate)
    e_s = int(x.shape[0] * test_rate)
    t_s = x.shape[0] - v_s - e_s
    
    # training splitting
    try:
        tx = x[split_index['train']]
        ty = y[split_index['train']]
    except:        
        tx = x[:t_s]
        ty = y[:t_s]
    # validation splitting
    try:
        vx = x[split_index['validation']]
        vy = y[split_index['validation']]
    except:
        if val_rate > 0:
            vx = x[t_s:t_s + v_s]
            vy = y[t_s:t_s + v_s]
        else:
            vx = None
            vy = None
    # indexing
    try:
        ex = x[split_index['test']]
        ey = y[split_index['test']]
    except:
        if test_rate > 0:
            ex = x[-e_s:]
            ey = y[-e_s:]
        else:
            ex = None
            ey = None
        
    print('\n Obtained data:')
    print('    Train size: ' + str(tx.shape[0]))
    try:
        print('    Validation size: ' + str(vx.shape[0]))
    except:
        pass
    try:
        print('    Test size: ' + str(ex.shape[0]))
    except:
        pass
    time.sleep(2)
    # balancing training samples per classes
    if train_balancing:
        print('\n#   #   #   Balancing Training Data   #   #   #')
        tx, ty = balance_set(tx, ty)
        if shuffling:
            print('re-shuffling training data after balancing')
            p = np.random.permutation(tx.shape[0])
            tx = tx[p]
            ty = ty[p]
    if val_balancing:
        try:
            print('\n#   #   #   Balancing Validation Data   #   #   #')
            vx, vy = balance_set(vx, vy)
        except:
            pass
    if test_balancing:
        try:
            print('\n#   #   #   Balancing Test Data   #   #   #')
            ex, ey = balance_set(ex, ey)
        except:
            pass

    return tx, ty, vx, vy, ex, ey

        
def balance_set(x, y):
    """ Reduces the samples of the given set balancing label-wise

    x  :  #samples-by-#features numpy array data
    y  :  #samples-by-label numpy array data
    """
    n_classes = np.unique(y)
    tmp_set = dict()
    tmp_target = dict()
    print('\n Balancing data over classes: ' + str(n_classes))
    print(' Samples distribution:')
    #  counting samples label-wise
    for i in range(n_classes.shape[0]):
        tmp_set[i] = x[y == i]
        tmp_target[i] = y[y == i]

        print('   Class ' + str(i)+': ' + str(tmp_set[i].shape[0]))
        try:
            min_s = np.min((min_s, tmp_set[i].shape[0]))
        except:
            min_s = tmp_set[i].shape[0]
    #  generating new data set
    for i in range(n_classes.shape[0]):
        try:
            tmp_x = np.concatenate((tmp_x, tmp_set[i][:min_s]), axis=0)
            tmp_y = np.concatenate((tmp_y, tmp_target[i][:min_s]), axis=0)
        except:
            tmp_x = tmp_set[i][:min_s]
            tmp_y = tmp_target[i][:min_s]

    print(' Final Data size: ' + str(tmp_x.shape[0]))
    
    return tmp_x, tmp_y

    
def normalize_data(x, method='statistical', p1=None, p2=None, data_name='Given'):
    """ Implementation of different data normalization procedures

    x : #samples-by-#features numpy array data

    method : statistical  -- p1=features mean, p2=features std
             minmax -- p1=features min, p2=features max
    """
    print('\n#   #   #   Normalizing '+data_name+' Data   #   #   #')
    print('Chosen Method: '+method)
    #  computing parameters
    try:
        #  normalization
        if method == 'statistical':
            x = (x - p1)/p2
        elif method == 'minmax':
            d = (p2 - p1).astype(float)
            d[d == 0] = 1.0
            x = (x - p1)/d
    except:
        if method == 'statistical':
            p1 = np.mean(x, axis=0)
            print('  computing mean . . . ')
            p2 = np.std(x, axis=0)
            print('  computing std . . . ')
        elif method == 'minmax':
            p1 = np.min(x, axis=0)
            print('  computing min . . . ')
            p2 = np.max(x, axis=0)
            print('  computing max . . . ')
        # normalization
        if method == 'statistical':
            x = (x - p1)/p2
        elif method == 'minmax':
            d = (p2 - p1).astype(float)
            d[d == 0] = 1.0
            x = (x - p1)/d
    
    print('Final Statistics: ')
    print('    min: ' + str(np.min(x)))
    print('    max: ' + str(np.max(x)))
    print('    mean: ' + str(np.mean(x)))
    print('    std: ' + str(np.std(x)))
    time.sleep(1)
    return x, p1, p2


def ind2vec(labels, classes=None):
    """ generate 1-hot encoding from a 1-D vector of labels

    labels :  #samples 1-D numpy array containing the class labels (from 0 to #classes-1)
    classes : number of classes (default=max(labels+1))
    """
    #  setting number of classes
    if classes is not None:
        classes = np.max(labels)+1
    #  output vector 0-initialization
    y = np.zeros((labels.shape[0], classes))
    #  setting the labeled positions to 1
    y[np.arange(labels.shape[0]), labels.astype(int)] = 1
    
    return y


def data2seq(data, labels, seq_length, shuffling=False, labeling='max'):
    """ generate a set of sequence from input data

    data :  #samples-by-(#features) numpy array data
    labels :  #samples 1-D numpy array containing class labels
    seq_length :  desired length of sequences
    shuffling :  final random shuffling os sequences order
    labeling :  'middle' select the label in the half of the sequence
                'last'  select the last label of the sequence
                'max' select the max label (suitable for anomalies or 0-1 classes)
    """
    data_dim = [data.shape[0] - seq_length + 1]
    data_dim = data_dim + [seq_length]
    for i in range(1, data.ndim):
        data_dim = data_dim + [data.shape[i]]
        
    print('\n Generating ' + str(data_dim[0]) + ' sequences of length ' +
          str(seq_length) + ' from ' + str(data.shape[0]) + ' samples')
    xtmp = np.zeros(data_dim)
    if labels is not None:
        ytmp = np.zeros(data_dim[0])
    else:
        ytmp = None
    
    for i in range(data_dim[0]):
        xtmp[i] = data[i:i + seq_length]
        if labels is not None:
            if labeling == 'middle':
                ytmp[i] = labels[i + seq_length/2]
            elif labeling == 'last':
                ytmp[i] = labels[i + seq_length - 1]
            elif labeling == 'max':
                ytmp[i] = np.max(labels[i:i + seq_length])
            else:
                raise NameError('Unknown type of labeling')

    if shuffling:
        p = np.random.permutation(xtmp.shape[0])
        xtmp = xtmp[p]
        if labels is not None:
            ytmp = ytmp[p]
    
    return xtmp, ytmp


def class_thresholding(x, y, c):
    """ Compute threshold to cut of a class
    x : (samples, timesteps, inputsize) data matrix
    y : labels (categorical, 1-D)
    c : class to cut of (int, < max(y))
    """
    xtmp = x
    s = xtmp.ndim
    while s > 1:
        xtmp = np.sum(xtmp, axis=s-1)
        s = xtmp.ndim

    return np.max(xtmp[y == c])
