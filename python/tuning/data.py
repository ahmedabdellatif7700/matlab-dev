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
    variables = [v+'_'+s for s in splits for v in ['X','Y']]+['ideal_symbols','ideal_bits','condition_json','configuration_json']
    data = loadmat(path, variable_names=variables)
    ideal = data['ideal_symbols'].ravel()
    bits = data['ideal_bits'].astype(np.int64)
    for split in splits:
        x, y = data['X_'+split], data['Y_'+split]
        if x.ndim!=2 or x.shape[1]%2 or y.shape!=(len(x),2) or not np.isfinite(x).all():
            raise ValueError('Invalid dataset')
        data['bits_'+split] = labels(y, ideal, bits)
    return data
