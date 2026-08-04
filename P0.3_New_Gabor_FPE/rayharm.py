import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import shift as ndshift

def gabor(theta,lam,sn,st,ks):
    h=ks//2; y,x=np.mgrid[-h:h+1,-h:h+1].astype(float)
    xt=x*np.cos(theta)+y*np.sin(theta); xn=-x*np.sin(theta)+y*np.cos(theta)
    g=np.exp(-(xt**2/(2*st**2)+xn**2/(2*sn**2)))*np.exp(1j*2*np.pi*xn/lam)
    return g-g.mean()

def estack(img,TH,lam=10.,sn=5.,st=9.,ks=35):
    return np.abs(np.stack([fftconvolve(img,gabor(t,lam,sn,st,ks),mode='same') for t in TH]))

def ray_harmonics(E,TH,d=16.0,K=144,nmax=4):
    """Dense ray-harmonic maps.
    R(y,x,phi) = E[theta=phi mod pi](y + d sin phi, x + d cos phi)
    c_n(y,x)   = (1/K) sum_phi R(y,x,phi) exp(-i n phi)
    Implemented as K rigid shifts of orientation channels + a linear combination
    -> the whole thing is a LINEAR filter over the lifted field (no thresholds).
    """
    NT=len(TH); H,W=E.shape[1:]
    phis=np.linspace(0,2*np.pi,K,endpoint=False)
    C=np.zeros((nmax+1,H,W),complex)
    for ph in phis:
        ti=int(round((ph%np.pi)/np.pi*NT))%NT
        sh=ndshift(E[ti],(-d*np.sin(ph),-d*np.cos(ph)),order=1,mode='constant')
        for n in range(nmax+1):
            C[n]+=sh*np.exp(-1j*n*ph)
    return C/K

def ray_profile(E,TH,y,x,d=16.0,K=144):
    from scipy.ndimage import map_coordinates
    NT=len(TH); phis=np.linspace(0,2*np.pi,K,endpoint=False); out=np.zeros(K)
    for i,ph in enumerate(phis):
        ti=int(round((ph%np.pi)/np.pi*NT))%NT
        out[i]=map_coordinates(E[ti],[[y+d*np.sin(ph)],[x+d*np.cos(ph)]],order=1,mode='constant')[0]
    return phis,out
