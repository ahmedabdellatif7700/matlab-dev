"""Validation-tuned classical receivers, including bit-objective controls."""
from __future__ import annotations
import itertools
import math
import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.spatial.distance import cdist


def bce(logits, target):
    return float(np.mean(np.logaddexp(0, logits)-target*logits))


def nearest_bits(value, ideal, labels):
    # Cartesian slicing avoids charging an unnecessarily expensive M-way
    # distance search to conventional QAM receivers.
    re, im = np.unique(ideal.real), np.unique(ideal.imag)
    lookup = np.zeros((len(re), len(im)), dtype=int)
    for k, symbol in enumerate(ideal):
        lookup[np.searchsorted(re, symbol.real), np.searchsorted(im, symbol.imag)] = k
    ri = np.searchsorted((re[:-1]+re[1:])/2, value[:, 0])
    qi = np.searchsorted((im[:-1]+im[1:])/2, value[:, 1])
    return labels[lookup[ri, qi]]


def make_basis(x, degree):
    mean = x.mean(axis=0, dtype=np.float64)
    scale = np.maximum(x.std(axis=0, dtype=np.float64), 1e-8)
    terms = [t for p in range(1, degree+1) for t in itertools.combinations_with_replacement(range(x.shape[1]), p)]
    model = dict(transform='polynomial', mean=mean, scale=scale, terms=terms)
    phi = raw_features(model, x)
    model['feature_mean'] = phi.mean(axis=0)
    model['feature_scale'] = np.maximum(phi.std(axis=0), 1e-8)
    return model


def raw_features(model, x):
    z = (np.asarray(x, dtype=float)-model['mean'])/model['scale']
    if model['transform']=='polynomial':
        return np.column_stack([np.prod(z[:, term], axis=1) for term in model['terms']])
    return np.exp(-model['gamma']*cdist(z, model['centers'], metric='sqeuclidean'))


def features(model, x):
    phi = (raw_features(model, x)-model['feature_mean'])/model['feature_scale']
    return np.column_stack((phi, np.ones(len(phi))))


def predict(model, x, ideal, labels):
    if model['kind']=='complex_ffe':
        cx = x[:, 0::2]+1j*x[:, 1::2]
        value = cx@model['weights']
        return nearest_bits(np.column_stack((value.real,value.imag)), ideal, labels)
    output = features(model, x)@model['weights']
    return output >= 0 if model['kind']=='bits' else nearest_bits(output, ideal, labels)


def logistic(phi, bits, ridge, maxiter=300):
    n, d = phi.shape; outputs = bits.shape[1]
    def objective(flat):
        w = flat.reshape(d, outputs)
        z = phi@w
        loss = bce(z, bits)+ridge*np.sum(w[:-1]**2)/2
        grad = phi.T@(expit(z)-bits)/(n*outputs)
        grad[:-1] += ridge*w[:-1]
        return loss, grad.ravel()
    fit = minimize(objective, np.zeros(d*outputs), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-6))
    return fit.x.reshape(d, outputs), dict(iterations=int(fit.nit), converged=bool(fit.success))


def fit_all(x, y, bits, vx, vy, vbits, ideal, labels, *, degrees=(1,2,3,5,7),
            bit_degrees=(1,2,3,5), kernel_sizes=(16,64,256), seed=42, quick=False):
    """No test arrays enter this function. Select each model on validation BER.

    Nonlinear demappers optimize the same bitwise loss as the neural heads.
    Higher-order models remain in the unrestricted comparison even when they
    exceed the NN's observed inference cost.
    """
    models=[]
    def retain(model, name, params, mul, nonlin, selection):
        decision=predict(model, vx, ideal, labels)
        model['info']=dict(name=name, parameters=int(params), multiplications=int(mul),
                           nonlinear_evaluations=int(nonlin), comparisons=bits.shape[1],
                           validation_ber=float(np.mean(decision!=vbits)), **selection)
        models.append(model)

    cx=x[:,0::2].astype(float)+1j*x[:,1::2]
    cy=y[:,0]+1j*y[:,1]
    w=np.linalg.lstsq(cx,cy,rcond=None)[0]
    retain(dict(kind='complex_ffe',weights=w), 'Complex FFE',2*w.size,4*w.size,0,{})
    for degree in degrees:
        basis=make_basis(x, degree); phi=features(basis,x); val=features(basis,vx)
        u,s,vt=svd(phi[:,:-1],full_matrices=False,check_finite=False)
        target_mean=y.mean(axis=0); projected=u.T@(y-target_mean)
        best=None; best_score=(math.inf,math.inf)
        for ridge in (0,1e-7,1e-5,1e-3,1e-1):
            factors=np.zeros_like(s)
            if ridge:
                factors=s/(s*s+len(x)*ridge)
            else:
                keep=s>np.finfo(float).eps*max(phi.shape)*s[0]
                factors[keep]=1/s[keep]
            weights=np.vstack((vt.T@(factors[:,None]*projected),target_mean))
            output=val@weights
            score=(float(np.mean(nearest_bits(output,ideal,labels)!=vbits)),float(np.mean((output-vy)**2)))
            if score<best_score:
                best_score=score; best=(weights,ridge)
        basis['weights']=best[0]; basis['kind']='regression'
        count=len(basis['terms']); cost=count-x.shape[1]  # Cache each lower-order monomial once.
        retain(basis, f'Volterra regression {degree}',2*(count+1),cost+2*count,0,dict(order=degree,ridge=best[1]))
        if degree in bit_degrees:
            best_score=(math.inf,math.inf); best=None
            for ridge in ((1e-4,) if quick else (1e-6,1e-4,1e-2)):
                weights, fit=logistic(phi,bits,ridge,maxiter=150 if quick else 300)
                logits=val@weights
                score=(float(np.mean((logits>=0)!=vbits)),bce(logits,vbits))
                if score<best_score:
                    best_score=score; best=(weights,ridge,fit)
            bit_model=dict(basis, kind='bits', weights=best[0])
            retain(bit_model,f'Volterra bit classifier {degree}',(count+1)*bits.shape[1],
                   cost+count*bits.shape[1],0,dict(order=degree,ridge=best[1],fit=best[2]))
    rng=np.random.default_rng(seed)
    mean=x.mean(axis=0,dtype=np.float64); scale=np.maximum(x.std(axis=0,dtype=np.float64),1e-8)
    z=(x-mean)/scale
    for count in kernel_sizes:
        centers=z[rng.choice(len(z),min(count,len(z)),replace=False)]
        median=max(float(np.median(cdist(centers,centers,'sqeuclidean'))),1e-6)
        best=None; best_score=(math.inf,math.inf)
        for bandwidth in ((1.,) if quick else (.25,1.,4.)):
            basis=dict(transform='rbf',mean=mean,scale=scale,centers=centers,gamma=bandwidth/median)
            phi=raw_features(basis,x)
            basis['feature_mean']=phi.mean(axis=0); basis['feature_scale']=np.maximum(phi.std(axis=0),1e-8)
            phi=features(basis,x); val=features(basis,vx)
            for ridge in ((1e-4,) if quick else (1e-6,1e-4,1e-2)):
                weights,fit=logistic(phi,bits,ridge,maxiter=150 if quick else 300)
                logits=val@weights
                score=(float(np.mean((logits>=0)!=vbits)),bce(logits,vbits))
                if score<best_score:
                    best_score=score; best=(dict(basis,kind='bits',weights=weights),ridge,fit,bandwidth)
        params=centers.size+best[0]['weights'].size
        retain(best[0],f'RBF kernel bit classifier {count}',params,
               count*(x.shape[1]+bits.shape[1]+3),count,
               dict(centers=count,bandwidth=best[3],ridge=best[1],fit=best[2]))
    return models
