"""Read only explicitly permitted MAT variables during discovery."""
import numpy as np
from scipy.io import loadmat


def labels(y, ideal, bit_labels):
    symbol = y[:, 0].astype(float)+1j*y[:, 1]
    indices = np.argmin(abs(symbol[:, None]-ideal[None, :]), axis=1)
    if abs(symbol-ideal[indices]).max() > 1e-6:
        raise ValueError('Targets do not use the declared MATLAB constellation')
    return bit_labels[indices]


def load(path, phase='discovery'):
    splits = ['train', 'validation'] + (['test'] if phase=='confirmation' else [])
    metadata=loadmat(path,variable_names=['split_counts'])
    packed='split_counts' in metadata
    if packed:
        counts=metadata['split_counts'].ravel().astype(int)
        if len(counts)!=len(splits):
            raise ValueError('Dataset phase does not match requested phase; refusing to read other splits')
    variables = ['X','Y','split_counts']+[v+'_'+s for s in splits for v in ['X','Y']]+['ideal_symbols','ideal_bits','condition_json','configuration_json']
    data = loadmat(path, variable_names=variables)
    if packed:
        edges=np.r_[0,np.cumsum(counts)]
        if edges[-1]!=len(data['X']) or len(data['Y'])!=len(data['X']):
            raise ValueError('Split counts do not match X/Y rows')
        for k,split in enumerate(splits):
            data['X_'+split]=data['X'][edges[k]:edges[k+1]]
            data['Y_'+split]=data['Y'][edges[k]:edges[k+1]]
    ideal = data['ideal_symbols'].ravel()
    bits = data['ideal_bits'].astype(np.int64)
    for split in splits:
        x, y = data['X_'+split], data['Y_'+split]
        if x.ndim!=2 or x.shape[1]%2 or y.shape!=(len(x),2) or not np.isfinite(x).all():
            raise ValueError('Invalid dataset')
        data['bits_'+split] = labels(y, ideal, bits)
    return data
