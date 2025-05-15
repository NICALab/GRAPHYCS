
"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved     Heintzmann Lab, Friedrich-Schiller-University Jena, Germany

@author: Rainer Heintzmann, Sheng Liu, Jonas Hellgoth
"""

import torch
import torch.fft
import torch.nn.functional as F
import numpy as np
import scipy as sp
from math import factorial 
import numbers
from scipy import ndimage
import scipy.fft as fft
defaultTFDataType="float32"
defaultTFCpxDataType="complex64"


##### The following code for zernike polynomial generation / noll and ANSI indexing is copied from the uiPSF code
##### https://github.com/ries-lab/uiPSF


def nl2noll(n,l):
    mm = abs(l)
    j = n * (n + 1) / 2 + 1 + max(0, mm - 1)
    if ((l > 0) & (np.mod(n, 4) >= 2)) | ((l < 0) & (np.mod(n, 4) <= 1)):
       j = j + 1
    
    return np.int32(j)

def noll2nl(j):
    n = np.ceil((-3 + np.sqrt(1 + 8*j)) / 2)
    l = j - n * (n + 1) / 2 - 1
    if np.mod(n, 2) != np.mod(l, 2):
       l = l + 1
    
    if np.mod(j, 2) == 1:
       l= -l
    
    return np.int32(n),np.int32(l)

def nl2ansi(n,l):
    j = (n*(n+2)+l)/2
    return j

def noll2ansi(i):
    n, l = noll2nl(i)
    return int(nl2ansi(n, l))

def radialpoly(n,m,rho):
    if m==0:
        g = np.sqrt(n+1)
    else:
        g = np.sqrt(2*n+2)
    r = np.zeros(rho.shape)
    for k in range(0,(n-m)//2+1):
        coeff = g*((-1)**k)*factorial(n-k)/factorial(k)/factorial((n+m)//2-k)/factorial((n-m)//2-k)
        p = rho**(n-2*k)
        r += coeff*p

    return r

def genZern1(n_max,xsz):
    Nk = (n_max+1)*(n_max+2)//2
    Z = np.ones((Nk,xsz,xsz))
    pkx = 2/xsz
    xrange = np.linspace(-xsz/2+0.5,xsz/2-0.5,xsz)
    [xx,yy] = np.meshgrid(xrange,xrange)
    rho = np.lib.scimath.sqrt((xx*pkx)**2+(yy*pkx)**2)
    phi = np.arctan2(yy,xx)

    for j in range(0,Nk):
        [n,l] = noll2nl(j+1)
        m = np.abs(l)
        r = radialpoly(n,m,rho)
        if l<0:
            Z[j] = r*np.sin(phi*m)
        else:
            Z[j] = r*np.cos(phi*m)
    return Z



def genZernAnsi(order_max, n_max_ansi,xsz):
    zernikesNoll = genZern1(order_max,xsz)
    zernikesAnsi = np.zeros((len(zernikesNoll),xsz,xsz))
    n_max_noll = 21 ## assuming a maximum order of 5
    if order_max == 4:
        n_max_noll = 15
    elif order_max == 5:
        n_max_noll = 21
    elif order_max == 6:
        n_max_noll = 28
    noll_indices = np.arange(1,n_max_noll+1,1).tolist()
    ansi_indices = [noll2ansi(i) for i in noll_indices]

    print(ansi_indices)
    for i in range(len(zernikesNoll)):
        zernikesAnsi[ansi_indices[i]] = zernikesNoll[i]
    
    zernikesAnsi = zernikesAnsi[:n_max_ansi+1]

    return zernikesAnsi


## for polynomial order greater than 14, use the rest of the Zernike polynomials for fitting (i.e. greater than 15)
def zernike_pd_generation_higher_order(order_max, n_max_ansi, M, pixelSize, wavelength, NA):
    ratio = pupilRadius(M, pixelSize, wavelength, NA)
    R = int( (1/ratio) * M)

    if n_max_ansi > 14:
        zernike_non_norm = genZernAnsi(6, 14, R)
        zernikes_higher_order = genZern1(order_max, R)
        zernikes_higher_order = zernikes_higher_order[15:n_max_ansi+1]
        zernike_non_norm = np.concatenate((zernike_non_norm, zernikes_higher_order), axis=0)
    else:
        print("n_max_ansi:", n_max_ansi)
        print("order_max:", order_max)
        zernike_non_norm = genZernAnsi(order_max, n_max_ansi, R)
    
    xx, yy = torch.meshgrid(torch.linspace(-1, 1, R), torch.linspace(-1, 1, R))
    dm_crop = torch.zeros((R, R))
    dm_crop[ xx**2 + yy**2 < 1] = 1

    zernike = torch.zeros(((n_max_ansi + 1), R, R))
    

    for z in range(len(zernike_non_norm)):
        zernike_term = dm_crop * torch.from_numpy(zernike_non_norm[z]).float()
        zernike[z] = zernike_term / zernike_term.max()
    padding = (M - R) // 2
    extra_padding = (M - R) % 2
    zernikeR = F.pad(zernike, (padding, padding + extra_padding, padding, padding + extra_padding))
    dm_crop = F.pad(dm_crop, (padding, padding + extra_padding, padding, padding + extra_padding))
    return dm_crop, zernikeR

def prechirpz1(kpixelsize, pixelsize_x, pixelsize_y, N, M, device):
    krange = torch.linspace(-N/2+0.5, N/2-0.5, N, dtype=torch.float32).to(device)
    xxK, yyK = torch.meshgrid(krange, krange, indexing='ij')
    xxK, yyK = xxK.to(device), yyK.to(device)
    
    xrange = torch.linspace(-M/2+0.5, M/2-0.5, M, dtype=torch.float32).to(device)
    xxR, yyR = torch.meshgrid(xrange, xrange, indexing='ij')
    xxR, yyR = xxR.to(device), yyR.to(device)
    
    a = 1j * torch.pi * kpixelsize
    A = torch.exp(a * (pixelsize_x * xxK**2 + pixelsize_y * yyK**2))
    C = torch.exp(a * (pixelsize_x * xxR**2 + pixelsize_y * yyR**2))
    
    brange = torch.linspace(-(N+M)/2+1, (N+M)/2-1, N+M-1, dtype=torch.float32).to(device)
    xxB, yyB = torch.meshgrid(brange, brange, indexing='ij')
    xxB, yyB = xxB.to(device), yyB.to(device)
    B = torch.exp(-a * (pixelsize_x * xxB**2 + pixelsize_y * yyB**2))
    
    Bh = torch.fft.fft2(B)
    
    return A, Bh, C


def cztfunc1(datain, param, device):
    A, Bh, C = param
    N = A.shape[0]
    L = Bh.shape[0]
    M = C.shape[0]

    
    Apad = torch.cat((A * datain / N, torch.zeros(datain.shape[:-1] + (L-N,), dtype=torch.complex64).to(device)), dim=-1)
    Apad = torch.cat((Apad, torch.zeros(Apad.shape[:-2] + (L-N, Apad.shape[-1]), dtype=torch.complex64).to(device)), dim=-2)
    
    Ah = torch.fft.fft2(Apad)
    cztout = torch.fft.ifft2(Ah * Bh / L)
    
    dataout = C * cztout[..., -M:, -M:]
    
    return dataout


def pupilRadius(M, pixelSize, wavelength, NA):
    k_max = NA / wavelength
    sampling = 1 / (M * pixelSize)
    return ( sampling / k_max) * (M / 2)
