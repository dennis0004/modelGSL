





# FV ADI - HV scheme

import numpy as np
#import matplotlib.pyplot as plt
from scipy.stats import beta

import os
import pickle
import datetime


def genMesh1d01(x0, m):
    X = np.zeros(m+1)
    xMax = m*x0
    c = x0/5
    dxi = ( np.arcsinh((xMax-x0)/c)-np.arcsinh(-x0/c) )/m
    for i in range(1, m):
        xi_i = np.arcsinh(-x0/c) + i*dxi
        X[i] = x0 + c*np.sinh(xi_i)
    X[m] = m*x0
    return X

def genMesh2d01(x10, x20, m):
    X1 = genMesh1d01(x10, m)
    X2 = genMesh1d01(x20, m)
    mX1, mX2 = np.meshgrid(X1, X2)
    return X1, X2, mX1, mX2

def genMesh2d02(Xmin, Xmax, dx):
    X1 = np.arange(Xmin, Xmax+dx/10, dx)
    X2 = np.arange(Xmin, Xmax+dx/10, dx)
    mX1, mX2 = np.meshgrid(X1, X2)
    return X1, X2, mX1, mX2

def int1d(fX, XmPoint5, XpPoint5):
    return np.sum( fX*(XpPoint5-XmPoint5) )

def int2d(fX, mX1mPoint5, mX1pPoint5, mX2mPoint5, mX2pPoint5):
    return np.sum( fX*(mX1pPoint5-mX1mPoint5)*(mX2pPoint5-mX2mPoint5) )

def initPDF1d(x0, X, XmPoint5, XpPoint5):
    pX = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        if x0-XmPoint5[i] >= 0 and XpPoint5[i]-x0 > 0:
            pX[i] = 1/(XpPoint5[i]-XmPoint5[i])
    return pX

def initPDF2d(x10, x20, X1, X2, X1mPoint5, X1pPoint5, X2mPoint5, X2pPoint5):
    pX = np.zeros((X1.shape[0], X2.shape[0]))
    for j in range(X1.shape[0]):
        for i in range(X2.shape[0]):
            if x10-X1mPoint5[j]>=0 and X1pPoint5[j]-x10>0 and x20-X2mPoint5[i]>=0 and X2pPoint5[i]-x20>0:
                pX[i,j] = 1/(X1pPoint5[j]-X1mPoint5[j])/(X2pPoint5[i]-X2mPoint5[i])
    return pX

def initPDF2d_pmf(pmfX, X1, X2, X1mPoint5, X1pPoint5, X2mPoint5, X2pPoint5):
    pX = np.zeros((X1.shape[0], X2.shape[0]))
    for j in range(X1.shape[0]):
        for i in range(X2.shape[0]):
            for (x10,x20),w0 in pmfX.items():
                if x10-X1mPoint5[j]>=0 and X1pPoint5[j]-x10>0 and x20-X2mPoint5[i]>=0 and X2pPoint5[i]-x20>0:
                    pX[i,j] = w0/(X1pPoint5[j]-X1mPoint5[j])/(X2pPoint5[i]-X2mPoint5[i])
    return pX

def initPDF2d_beta(betaInitPara, Xmax, Xmin, X1):
    X1Trans = (X1-Xmin)/(Xmax-Xmin)
    rv = beta(betaInitPara[0],betaInitPara[1])
    pX1 = rv.pdf(X1Trans)/(Xmax-Xmin)
    rv = beta(betaInitPara[2],betaInitPara[3])
    pX2 = rv.pdf(X1Trans)/(Xmax-Xmin)

    pX = pX2[:,np.newaxis]*pX1[np.newaxis,:]
    return pX

def TDMA(a,b,c,f):
    n = len(f)
    v = np.zeros(n)
    y = np.zeros(n)

    w = a[0]
    y[0] = f[0]/w
    for i in range(1, n):
        v[i-1] = c[i-1]/w
        w = a[i] - b[i]*v[i-1]
        y[i] = ( f[i]-b[i]*y[i-1] ) / w
    for j in range(n-2, -1, -1):
        y[j] = y[j] - v[j]*y[j+1]
    return y

def cal_dt_P(vecpX, m, A0i__j__, A0im1j__, A0ip1j__, A0i__jm1, A0i__jp1, A0im1jm1, A0im1jp1, A0ip1jm1, A0ip1jp1, A1i__j__, A1i__jm1, A1i__jp1, A2i__j__, A2im1j__, A2ip1j__):
    A_P = np.zeros(( (m+1)*(m+1) ))
    A0P = np.zeros(( (m+1)*(m+1) ))
    A1P = np.zeros(( (m+1)*(m+1) ))
    A2P = np.zeros(( (m+1)*(m+1) ))

    for i in range(0,m+1):
        for j in range(0,m+1):
            A0P[i*(m+1)+j] += A0i__j__[i*(m+1)+j]*vecpX[i*(m+1)+j]
            if i==0:
                A0P[i*(m+1)+j] += A0im1j__[i*(m+1)+j]*vecpX[(i-0)*(m+1)+j]
            else:
                A0P[i*(m+1)+j] += A0im1j__[i*(m+1)+j]*vecpX[(i-1)*(m+1)+j]
            if i==m:
                A0P[i*(m+1)+j] += A0ip1j__[i*(m+1)+j]*vecpX[(i+0)*(m+1)+j]
            else:
                A0P[i*(m+1)+j] += A0ip1j__[i*(m+1)+j]*vecpX[(i+1)*(m+1)+j]
            if j==0:
                A0P[i*(m+1)+j] += A0i__jm1[i*(m+1)+j]*vecpX[i*(m+1)+(j-0)]
            else:
                A0P[i*(m+1)+j] += A0i__jm1[i*(m+1)+j]*vecpX[i*(m+1)+(j-1)]
            if j==m:
                A0P[i*(m+1)+j] += A0i__jp1[i*(m+1)+j]*vecpX[i*(m+1)+(j+0)]
            else:
                A0P[i*(m+1)+j] += A0i__jp1[i*(m+1)+j]*vecpX[i*(m+1)+(j+1)]
            if i==0 and j==0:
                A0P[i*(m+1)+j] += 0
            elif i==0:
                A0P[i*(m+1)+j] += A0im1jm1[i*(m+1)+j]*vecpX[(i-0)*(m+1)+(j-1)]
            elif j==0:
                A0P[i*(m+1)+j] += A0im1jm1[i*(m+1)+j]*vecpX[(i-1)*(m+1)+(j-0)]
            else:
                A0P[i*(m+1)+j] += A0im1jm1[i*(m+1)+j]*vecpX[(i-1)*(m+1)+(j-1)]
            if i==0 and j==m:
                A0P[i*(m+1)+j] += 0
            elif i==0:
                A0P[i*(m+1)+j] += A0im1jp1[i*(m+1)+j]*vecpX[(i-0)*(m+1)+(j+1)]
            elif j==m:
                A0P[i*(m+1)+j] += A0im1jp1[i*(m+1)+j]*vecpX[(i-1)*(m+1)+(j+0)]
            else:
                A0P[i*(m+1)+j] += A0im1jp1[i*(m+1)+j]*vecpX[(i-1)*(m+1)+(j+1)]
            if i==m and j==0:
                A0P[i*(m+1)+j] += 0
            elif i==m:
                A0P[i*(m+1)+j] += A0ip1jm1[i*(m+1)+j]*vecpX[(i+0)*(m+1)+(j-1)]
            elif j==0:
                A0P[i*(m+1)+j] += A0ip1jm1[i*(m+1)+j]*vecpX[(i+1)*(m+1)+(j-0)]
            else:
                A0P[i*(m+1)+j] += A0ip1jm1[i*(m+1)+j]*vecpX[(i+1)*(m+1)+(j-1)]
            if i==m and j==m:
                A0P[i*(m+1)+j] += 0
            elif i==m:
                A0P[i*(m+1)+j] += A0ip1jp1[i*(m+1)+j]*vecpX[(i+0)*(m+1)+(j+1)]
            elif j==m:
                A0P[i*(m+1)+j] += A0ip1jp1[i*(m+1)+j]*vecpX[(i+1)*(m+1)+(j+0)]
            else:
                A0P[i*(m+1)+j] += A0ip1jp1[i*(m+1)+j]*vecpX[(i+1)*(m+1)+(j+1)]
            # X1
            A1P[i*(m+1)+j] += A1i__j__[i*(m+1)+j]*vecpX[i*(m+1)+j]
            if j==0:
                A1P[i*(m+1)+j] += 0
            else:
                A1P[i*(m+1)+j] += A1i__jm1[i*(m+1)+j]*vecpX[i*(m+1)+(j-1)]
            if j==m:
                A1P[i*(m+1)+j] += 0
            else:
                A1P[i*(m+1)+j] += A1i__jp1[i*(m+1)+j]*vecpX[i*(m+1)+(j+1)]
            # X2
            A2P[i*(m+1)+j] += A2i__j__[i*(m+1)+j]*vecpX[i*(m+1)+j]
            if i==0:
                A2P[i*(m+1)+j] += 0
            else:
                A2P[i*(m+1)+j] += A2im1j__[i*(m+1)+j]*vecpX[(i-1)*(m+1)+j]
            if i==m:
                A2P[i*(m+1)+j] += 0
            else:
                A2P[i*(m+1)+j] += A2ip1j__[i*(m+1)+j]*vecpX[(i+1)*(m+1)+j]

            A_P[i*(m+1)+j] += A0P[i*(m+1)+j]
            A_P[i*(m+1)+j] += A1P[i*(m+1)+j]
            A_P[i*(m+1)+j] += A2P[i*(m+1)+j]
    return A_P, A1P, A2P

def FV_ADI_2d(gameName, A, Qmin, Qmax, d, alpha, tau, T, dt, nQ_point1, initDist, initPara, tStart, SLm):
    suffix = ''
    if SLm == 1:
        resultDir = 'model_Q-boltzmann_%s_lr%.2f_tau%.1f_Q0-%s%s_%s_nQ0point1-%d_dt%.3f_Qmin%.1f_Qmax%.1f_SL%s'%(gameName, alpha, tau, initDist, str(initPara).replace(':','-'), 'FV-ADI', nQ_point1, dt, Qmin, Qmax, suffix)
    else:
        resultDir = 'model_Q-boltzmann_%s_lr%.2f_tau%.1f_Q0-%s%s_%s_nQ0point1-%d_dt%.3f_Qmin%.1f_Qmax%.1f_SL%d%s'%(gameName, alpha, tau, initDist, str(initPara).replace(':','-'), 'FV-ADI', nQ_point1, dt, Qmin, Qmax, SLm, suffix)
    print(resultDir)
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)

    dq = 0.1/nQ_point1
    nQ_ = (Qmax-Qmin)*nQ_point1*10+1
    m = int(nQ_-1)
    nCount1 = int(1/dt)
    nCount = int(T/dt)

    t = tStart
    count = t*nCount1

    e1 = np.array([[1],[0]])
    e2 = np.array([[0],[1]])

    X1, X2, mX1, mX2 = genMesh2d02(Qmin, Qmax, dq)

    print('m:', m)
    print('X1 shape:', X1.shape)
    print('X1:', X1[:5], '...', X1[-5:])

    dX1 = np.hstack( (0,X1[1:]-X1[:-1]) )
    dX1p1 = np.hstack( (X1[1:]-X1[:-1], 0) )
    dX2 = np.hstack( (0,X2[1:]-X2[:-1]) )
    dX2p1 = np.hstack( (X2[1:]-X2[:-1], 0) )
#    print(dX1)

    X1mPoint5 = X1 - dX1/2
    X1pPoint5 = X1 + dX1p1/2
    X2mPoint5 = X2 - dX2/2
    X2pPoint5 = X2 + dX2p1/2
    mX1mPoint5, mX2mPoint5 = np.meshgrid(X1mPoint5, X2mPoint5)
    mX1pPoint5, mX2pPoint5 = np.meshgrid(X1pPoint5, X2pPoint5)
#    print(mX1mPoint5)

    constant1 = Qmax/2

    eTauX1 = np.exp(tau*(mX1-constant1))
    eTauX2 = np.exp(tau*(mX2-constant1))
    sumETauX = eTauX1+eTauX2
    softmaxX1 = eTauX1/sumETauX
    softmaxX2 = eTauX2/sumETauX

    eTauX1mPoint5 = np.exp(tau*(mX1mPoint5-constant1))
    eTauX1pPoint5 = np.exp(tau*(mX1pPoint5-constant1))
    eTauX2mPoint5 = np.exp(tau*(mX2mPoint5-constant1))
    eTauX2pPoint5 = np.exp(tau*(mX2pPoint5-constant1))
    softmaxX1mPoint5 = eTauX1mPoint5/(eTauX1mPoint5+eTauX2)
    softmaxX1pPoint5 = eTauX1pPoint5/(eTauX1pPoint5+eTauX2)
    softmaxX2mPoint5 = eTauX2mPoint5/(eTauX1+eTauX2mPoint5)
    softmaxX2pPoint5 = eTauX2pPoint5/(eTauX1+eTauX2pPoint5)

    # Boltzmann
    y1 = softmaxX1
    y1mPoint5 = softmaxX1mPoint5
    y1pPoint5 = softmaxX1pPoint5
    y2 = softmaxX2
    y2mPoint5 = softmaxX2mPoint5
    y2pPoint5 = softmaxX2pPoint5

    if t==0:
        if initDist=='beta':
            pX = initPDF2d_beta(initPara, np.amax(A), np.amin(A), X1)
        if initDist=='all':
            pX = initPDF2d(initPara[0], initPara[0], X1, X2, X1mPoint5, X1pPoint5, X2mPoint5, X2pPoint5)
        if initDist=='pmf':
            pX = initPDF2d_pmf(initPara, X1, X2, X1mPoint5, X1pPoint5, X2mPoint5, X2pPoint5)

#        print('Area:', int2d(pX, mX1mPoint5, mX1pPoint5, mX2mPoint5, mX2pPoint5))
        filePath1 = '%s/pQ%03d.pickle'%(resultDir, t)
        f01 = open(filePath1, 'wb')
        pickle.dump(pX, f01)
        f01.close()
    else:
        filePath1 = '%s/pQ%03d.pickle'%(resultDir, t)
        f01 = open(filePath1, 'rb')
        pX = pickle.load(f01)
        f01.close()

    vecpX = np.ravel(pX)

    ExT = np.zeros((T+1,d))
    EQT = np.zeros((T+1,d))
    yt = np.zeros((d,1))
    EQt = np.zeros((d,1))

    EQt[0] = int2d(mX1*pX, mX1mPoint5, mX1pPoint5, mX2mPoint5, mX2pPoint5)
    EQt[1] = int2d(mX2*pX, mX1mPoint5, mX1pPoint5, mX2mPoint5, mX2pPoint5)
    yt[0] = int2d(y1*pX, mX1mPoint5, mX1pPoint5, mX2mPoint5, mX2pPoint5)
    yt[1] = int2d(y2*pX, mX1mPoint5, mX1pPoint5, mX2mPoint5, mX2pPoint5)
    if t==0:
        print('time %d'%(count//nCount1))
        ExT[t,:] = yt.T
        EQT[t,:] = EQt.T
    else:
        filePath1 = '%s/ExT.pickle'%(resultDir)
        f01 = open(filePath1, 'rb')
        ExTinit = pickle.load(f01)
        f01.close()
        ExT[:t+1,:] = ExTinit
        filePath1 = '%s/EQT.pickle'%(resultDir)
        f01 = open(filePath1, 'rb')
        EQTinit = pickle.load(f01)
        f01.close()
        EQT[:t+1,:] = EQTinit


    while True:
        # MU and Sigma

#        MU1ij = alpha*softmaxX1*(e1.T@A@yt - mX1)
#        MU2ij = alpha*softmaxX2*(e2.T@A@yt - mX2)

        MU1ijmPoint5 = alpha*softmaxX1mPoint5*(e1.T@A@yt - mX1mPoint5)
        MU1ijpPoint5 = alpha*softmaxX1pPoint5*(e1.T@A@yt - mX1pPoint5)
        MU2imPoint5j = alpha*softmaxX2mPoint5*(e2.T@A@yt - mX2mPoint5)
        MU2ipPoint5j = alpha*softmaxX2pPoint5*(e2.T@A@yt - mX2pPoint5)

        if SLm==-1:
            Sigma11ij = alpha**2*(e1.T@A@yt-mX1)**2*y1*(1-y1)
            Sigma22ij = alpha**2*(e2.T@A@yt-mX2)**2*y2*(1-y2)

            if gameName=='SH':
                Sigma11ij += 2e-5
                Sigma22ij += 0
        else:
            Sigma11ij = alpha**2*(e1.T@A@yt-mX1)**2*y1*(1-y1) + alpha**2*(e1.T@A**2@yt-(e1.T@A@yt)**2)*y1/SLm
            Sigma22ij = alpha**2*(e2.T@A@yt-mX2)**2*y2*(1-y2) + alpha**2*(e2.T@A**2@yt-(e2.T@A@yt)**2)*y2/SLm

#        Sigma12ij = -1 * alpha**2 * (e1.T@A@yt - mX1) * (e2.T@A@yt - mX2) * y1 * y2
        Sigma12imPoint5jmPoint5 = -1*alpha**2 * (e1.T@A@yt-mX1mPoint5)*(e2.T@A@yt-mX2mPoint5) * y1mPoint5*y2mPoint5
        Sigma12imPoint5jpPoint5 = -1*alpha**2 * (e1.T@A@yt-mX1pPoint5)*(e2.T@A@yt-mX2mPoint5) * y1pPoint5*y2mPoint5
        Sigma12ipPoint5jmPoint5 = -1*alpha**2 * (e1.T@A@yt-mX1mPoint5)*(e2.T@A@yt-mX2pPoint5) * y1mPoint5*y2pPoint5
        Sigma12ipPoint5jpPoint5 = -1*alpha**2 * (e1.T@A@yt-mX1pPoint5)*(e2.T@A@yt-mX2pPoint5) * y1pPoint5*y2pPoint5

        # FV start
        A0i__j__ = np.zeros(( (m+1)*(m+1) ))
        A0im1j__ = np.zeros(( (m+1)*(m+1) ))
        A0ip1j__ = np.zeros(( (m+1)*(m+1) ))
        A0i__jm1 = np.zeros(( (m+1)*(m+1) ))
        A0i__jp1 = np.zeros(( (m+1)*(m+1) ))
        A0im1jm1 = np.zeros(( (m+1)*(m+1) ))
        A0im1jp1 = np.zeros(( (m+1)*(m+1) ))
        A0ip1jm1 = np.zeros(( (m+1)*(m+1) ))
        A0ip1jp1 = np.zeros(( (m+1)*(m+1) ))
        # X1
        A1i__j__ = np.zeros(( (m+1)*(m+1) ))
        A1i__jm1 = np.zeros(( (m+1)*(m+1) ))
        A1i__jp1 = np.zeros(( (m+1)*(m+1) ))
        # X2
        A2i__j__ = np.zeros(( (m+1)*(m+1) ))
        A2im1j__ = np.zeros(( (m+1)*(m+1) ))
        A2ip1j__ = np.zeros(( (m+1)*(m+1) ))

        A_P = np.zeros(( (m+1)*(m+1) ))
        A0P = np.zeros(( (m+1)*(m+1) ))
        A1P = np.zeros(( (m+1)*(m+1) ))
        A2P = np.zeros(( (m+1)*(m+1) ))

        for i in range(0,m+1):
            for j in range(0,m+1):
                A0i__j__[i*(m+1)+j] = Sigma12imPoint5jmPoint5[i,j]/(dX1[j]+dX1p1[j])/(dX2[i]+dX2p1[i]) \
                    + -Sigma12imPoint5jpPoint5[i,j]/(dX1[j]+dX1p1[j])/(dX2[i]+dX2p1[i]) \
                    + -Sigma12ipPoint5jmPoint5[i,j]/(dX1[j]+dX1p1[j])/(dX2[i]+dX2p1[i]) \
                    + Sigma12ipPoint5jpPoint5[i,j]/(dX1[j]+dX1p1[j])/(dX2[i]+dX2p1[i])
                A0im1j__[i*(m+1)+j] = Sigma12imPoint5jmPoint5[i,j]/(dX1[j]+dX1p1[j])/(dX2[i]+dX2p1[i]) \
                    + -Sigma12imPoint5jpPoint5[i,j]/(dX1[j]+dX1p1[j])/(dX2[i]+dX2p1[i])
                A0ip1j__[i*(m+1)+j] = -Sigma12ipPoint5jmPoint5[i,j]/(dX1[j]+dX1p1[j])/(dX2[i]+dX2p1[i]) \
                    + Sigma12ipPoint5jpPoint5[i,j]/(dX1[j]+dX1p1[j])/(dX2[i]+dX2p1[i])
                A0i__jm1[i*(m+1)+j] = Sigma12imPoint5jmPoint5[i,j]/(dX1[j]+dX1p1[j])/(dX2[i]+dX2p1[i]) \
                    + -Sigma12ipPoint5jmPoint5[i,j]/(dX1[j]+dX1p1[j])/(dX2[i]+dX2p1[i])
                A0i__jp1[i*(m+1)+j] = -Sigma12imPoint5jpPoint5[i,j]/(dX1[j]+dX1p1[j])/(dX2[i]+dX2p1[i]) \
                    + Sigma12ipPoint5jpPoint5[i,j]/(dX1[j]+dX1p1[j])/(dX2[i]+dX2p1[i])
                A0im1jm1[i*(m+1)+j] = Sigma12imPoint5jmPoint5[i,j]/(dX1[j]+dX1p1[j])/(dX2[i]+dX2p1[i])
                A0im1jp1[i*(m+1)+j] = -Sigma12imPoint5jpPoint5[i,j]/(dX1[j]+dX1p1[j])/(dX2[i]+dX2p1[i])
                A0ip1jm1[i*(m+1)+j] = -Sigma12ipPoint5jmPoint5[i,j]/(dX1[j]+dX1p1[j])/(dX2[i]+dX2p1[i])
                A0ip1jp1[i*(m+1)+j] = Sigma12ipPoint5jpPoint5[i,j]/(dX1[j]+dX1p1[j])/(dX2[i]+dX2p1[i])
                # X1
                if j==0:
                    A1i__j__[i*(m+1)+j] = -MU1ijpPoint5[i,j]/(dX1[j]+dX1p1[j])-Sigma11ij[i,j]/dX1p1[j]/(dX1[j]+dX1p1[j])
                elif j==m:
                    A1i__j__[i*(m+1)+j] = MU1ijmPoint5[i,j]/(dX1[j]+dX1p1[j])-Sigma11ij[i,j]/dX1[j]/(dX1[j]+dX1p1[j])
                else:
                    A1i__j__[i*(m+1)+j] = MU1ijmPoint5[i,j]/(dX1[j]+dX1p1[j])-Sigma11ij[i,j]/dX1[j]/(dX1[j]+dX1p1[j]) \
                        + -MU1ijpPoint5[i,j]/(dX1[j]+dX1p1[j])-Sigma11ij[i,j]/dX1p1[j]/(dX1[j]+dX1p1[j])
                if j==0:
                    A1i__jm1[i*(m+1)+j] = 0
                else:
                    A1i__jm1[i*(m+1)+j] = MU1ijmPoint5[i,j]/(dX1[j]+dX1p1[j])+Sigma11ij[i,j-1]/dX1[j]/(dX1[j]+dX1p1[j])
                if j==m:
                    A1i__jp1[i*(m+1)+j] = 0
                else:
                    A1i__jp1[i*(m+1)+j] = -MU1ijpPoint5[i,j]/(dX1[j]+dX1p1[j])+Sigma11ij[i,j+1]/dX1p1[j]/(dX1[j]+dX1p1[j])
                # X2
                if i==0:
                    A2i__j__[i*(m+1)+j] = -MU2ipPoint5j[i,j]/(dX2[i]+dX2p1[i])-Sigma22ij[i,j]/dX2p1[i]/(dX2[i]+dX2p1[i])
                elif i==m:
                    A2i__j__[i*(m+1)+j] = MU2imPoint5j[i,j]/(dX2[i]+dX2p1[i])-Sigma22ij[i,j]/dX2[i]/(dX2[i]+dX2p1[i])
                else:
                    A2i__j__[i*(m+1)+j] = MU2imPoint5j[i,j]/(dX2[i]+dX2p1[i])-Sigma22ij[i,j]/dX2[i]/(dX2[i]+dX2p1[i]) \
                        + -MU2ipPoint5j[i,j]/(dX2[i]+dX2p1[i])-Sigma22ij[i,j]/dX2p1[i]/(dX2[i]+dX2p1[i])
                if i==0:
                    A2im1j__[i*(m+1)+j] = 0
                else:
                    A2im1j__[i*(m+1)+j] = MU2imPoint5j[i,j]/(dX2[i]+dX2p1[i])+Sigma22ij[i-1,j]/dX2[i]/(dX2[i]+dX2p1[i])
                if i==m:
                    A2ip1j__[i*(m+1)+j] = 0
                else:
                    A2ip1j__[i*(m+1)+j] = -MU2ipPoint5j[i,j]/(dX2[i]+dX2p1[i])+Sigma22ij[i+1,j]/dX2p1[i]/(dX2[i]+dX2p1[i])


                A0P[i*(m+1)+j] += A0i__j__[i*(m+1)+j]*vecpX[i*(m+1)+j]
                if i==0:
                    A0P[i*(m+1)+j] += A0im1j__[i*(m+1)+j]*vecpX[(i-0)*(m+1)+j]
                else:
                    A0P[i*(m+1)+j] += A0im1j__[i*(m+1)+j]*vecpX[(i-1)*(m+1)+j]
                if i==m:
                    A0P[i*(m+1)+j] += A0ip1j__[i*(m+1)+j]*vecpX[(i+0)*(m+1)+j]
                else:
                    A0P[i*(m+1)+j] += A0ip1j__[i*(m+1)+j]*vecpX[(i+1)*(m+1)+j]
                if j==0:
                    A0P[i*(m+1)+j] += A0i__jm1[i*(m+1)+j]*vecpX[i*(m+1)+(j-0)]
                else:
                    A0P[i*(m+1)+j] += A0i__jm1[i*(m+1)+j]*vecpX[i*(m+1)+(j-1)]
                if j==m:
                    A0P[i*(m+1)+j] += A0i__jp1[i*(m+1)+j]*vecpX[i*(m+1)+(j+0)]
                else:
                    A0P[i*(m+1)+j] += A0i__jp1[i*(m+1)+j]*vecpX[i*(m+1)+(j+1)]
                if i==0 and j==0:
                    A0P[i*(m+1)+j] += 0
                elif i==0:
                    A0P[i*(m+1)+j] += A0im1jm1[i*(m+1)+j]*vecpX[(i-0)*(m+1)+(j-1)]
                elif j==0:
                    A0P[i*(m+1)+j] += A0im1jm1[i*(m+1)+j]*vecpX[(i-1)*(m+1)+(j-0)]
                else:
                    A0P[i*(m+1)+j] += A0im1jm1[i*(m+1)+j]*vecpX[(i-1)*(m+1)+(j-1)]
                if i==0 and j==m:
                    A0P[i*(m+1)+j] += 0
                elif i==0:
                    A0P[i*(m+1)+j] += A0im1jp1[i*(m+1)+j]*vecpX[(i-0)*(m+1)+(j+1)]
                elif j==m:
                    A0P[i*(m+1)+j] += A0im1jp1[i*(m+1)+j]*vecpX[(i-1)*(m+1)+(j+0)]
                else:
                    A0P[i*(m+1)+j] += A0im1jp1[i*(m+1)+j]*vecpX[(i-1)*(m+1)+(j+1)]
                if i==m and j==0:
                    A0P[i*(m+1)+j] += 0
                elif i==m:
                    A0P[i*(m+1)+j] += A0ip1jm1[i*(m+1)+j]*vecpX[(i+0)*(m+1)+(j-1)]
                elif j==0:
                    A0P[i*(m+1)+j] += A0ip1jm1[i*(m+1)+j]*vecpX[(i+1)*(m+1)+(j-0)]
                else:
                    A0P[i*(m+1)+j] += A0ip1jm1[i*(m+1)+j]*vecpX[(i+1)*(m+1)+(j-1)]
                if i==m and j==m:
                    A0P[i*(m+1)+j] += 0
                elif i==m:
                    A0P[i*(m+1)+j] += A0ip1jp1[i*(m+1)+j]*vecpX[(i+0)*(m+1)+(j+1)]
                elif j==m:
                    A0P[i*(m+1)+j] += A0ip1jp1[i*(m+1)+j]*vecpX[(i+1)*(m+1)+(j+0)]
                else:
                    A0P[i*(m+1)+j] += A0ip1jp1[i*(m+1)+j]*vecpX[(i+1)*(m+1)+(j+1)]
                # X1
                A1P[i*(m+1)+j] += A1i__j__[i*(m+1)+j]*vecpX[i*(m+1)+j]
                if j==0:
                    A1P[i*(m+1)+j] += 0
                else:
                    A1P[i*(m+1)+j] += A1i__jm1[i*(m+1)+j]*vecpX[i*(m+1)+(j-1)]
                if j==m:
                    A1P[i*(m+1)+j] += 0
                else:
                    A1P[i*(m+1)+j] += A1i__jp1[i*(m+1)+j]*vecpX[i*(m+1)+(j+1)]
                # X2
                A2P[i*(m+1)+j] += A2i__j__[i*(m+1)+j]*vecpX[i*(m+1)+j]
                if i==0:
                    A2P[i*(m+1)+j] += 0
                else:
                    A2P[i*(m+1)+j] += A2im1j__[i*(m+1)+j]*vecpX[(i-1)*(m+1)+j]
                if i==m:
                    A2P[i*(m+1)+j] += 0
                else:
                    A2P[i*(m+1)+j] += A2ip1j__[i*(m+1)+j]*vecpX[(i+1)*(m+1)+j]

                A_P[i*(m+1)+j] += A0P[i*(m+1)+j]
                A_P[i*(m+1)+j] += A1P[i*(m+1)+j]
                A_P[i*(m+1)+j] += A2P[i*(m+1)+j]
        # HV scheme
        Y = vecpX + A_P*dt
        Y0 = np.copy(Y)

        a = 1-A1i__j__*dt/2
        b = -A1i__jm1*dt/2
        c = -A1i__jp1*dt/2
        f = Y-A1P*dt/2
        Y = TDMA(a,b,c,f)

        Y = Y.reshape((m+1,m+1)).ravel(order='F')
        A2i__j__ = A2i__j__.reshape((m+1,m+1)).ravel(order='F')
        A2im1j__ = A2im1j__.reshape((m+1,m+1)).ravel(order='F')
        A2ip1j__ = A2ip1j__.reshape((m+1,m+1)).ravel(order='F')
        A2P = A2P.reshape((m+1,m+1)).ravel(order='F')
        a = 1-A2i__j__*dt/2
        b = -A2im1j__*dt/2
        c = -A2ip1j__*dt/2
        f = Y-A2P*dt/2
        Y = TDMA(a,b,c,f)
        Y = Y.reshape((m+1,m+1)).ravel(order='F')
        A2i__j__ = A2i__j__.reshape((m+1,m+1)).ravel(order='F')
        A2im1j__ = A2im1j__.reshape((m+1,m+1)).ravel(order='F')
        A2ip1j__ = A2ip1j__.reshape((m+1,m+1)).ravel(order='F')
        A2P = A2P.reshape((m+1,m+1)).ravel(order='F')

        A_Y, A1Y, A2Y = cal_dt_P(Y, m,
            A0i__j__, A0im1j__, A0ip1j__, A0i__jm1, A0i__jp1, A0im1jm1, A0im1jp1, A0ip1jm1, A0ip1jp1,
            A1i__j__, A1i__jm1, A1i__jp1,
            A2i__j__, A2im1j__, A2ip1j__
        )

        Y = Y0 + (A_Y-A_P)*dt/2

        a = 1-A1i__j__*dt/2
        b = -A1i__jm1*dt/2
        c = -A1i__jp1*dt/2
        f = Y-A1Y*dt/2
        Y = TDMA(a,b,c,f)

        Y = Y.reshape((m+1,m+1)).ravel(order='F')
        A2i__j__ = A2i__j__.reshape((m+1,m+1)).ravel(order='F')
        A2im1j__ = A2im1j__.reshape((m+1,m+1)).ravel(order='F')
        A2ip1j__ = A2ip1j__.reshape((m+1,m+1)).ravel(order='F')
        A2Y = A2Y.reshape((m+1,m+1)).ravel(order='F')
        a = 1-A2i__j__*dt/2
        b = -A2im1j__*dt/2
        c = -A2ip1j__*dt/2
        f = Y-A2Y*dt/2
        Y = TDMA(a,b,c,f)
        Y = Y.reshape((m+1,m+1)).ravel(order='F')
        A2i__j__ = A2i__j__.reshape((m+1,m+1)).ravel(order='F')
        A2im1j__ = A2im1j__.reshape((m+1,m+1)).ravel(order='F')
        A2ip1j__ = A2ip1j__.reshape((m+1,m+1)).ravel(order='F')
        A2Y = A2Y.reshape((m+1,m+1)).ravel(order='F')

        vecpX = np.copy(Y)

        pX = vecpX.reshape((m+1,m+1))
        pX = np.maximum(pX, 0)
        Area = int2d(pX, mX1mPoint5, mX1pPoint5, mX2mPoint5, mX2pPoint5)
        pX = pX/Area
        vecpX = pX.ravel()
        # FV end

        t = t+dt
        count += 1

        EQt[0] = int2d(mX1*pX, mX1mPoint5, mX1pPoint5, mX2mPoint5, mX2pPoint5)
        EQt[1] = int2d(mX2*pX, mX1mPoint5, mX1pPoint5, mX2mPoint5, mX2pPoint5)
        yt[0] = int2d(y1*pX, mX1mPoint5, mX1pPoint5, mX2mPoint5, mX2pPoint5)
        yt[1] = int2d(y2*pX, mX1mPoint5, mX1pPoint5, mX2mPoint5, mX2pPoint5)

        if count % nCount1 == 0:
            print('time %d'%(count//nCount1), 'yt:', yt.T)
            ExT[count // nCount1, :] = yt.T
            EQT[count // nCount1, :] = EQt.T

        if count % nCount1 == 0:
            filePath1 = '%s/pQ%03d.pickle'%(resultDir, count//nCount1)
            f01 = open(filePath1, 'wb')
            pickle.dump(pX, f01)
            f01.close()

        if count >= nCount:
            break

    filePath1 = '%s/ExT.pickle'%(resultDir)
    f01 = open(filePath1, 'wb')
    pickle.dump(ExT, f01)
    f01.close()
    filePath1 = '%s/ExT.txt'%(resultDir)
    np.savetxt(filePath1, ExT, fmt='%.6f', delimiter=',')

    filePath1 = '%s/EQT.pickle'%(resultDir)
    f01 = open(filePath1, 'wb')
    pickle.dump(EQT, f01)
    f01.close()
    filePath1 = '%s/EQT.txt'%(resultDir)
    np.savetxt(filePath1, EQT, fmt='%.6f', delimiter=',')

def main():
    gameName = 'PD'
    A = np.array([[2,0],[3,1]])
    Qmin = -0.5
    Qmax = 3.5
    initPara = {(0,0): 1}
    # initPara = {(1,0): 1}

    d = 2

    initDist = 'pmf'

    alpha = 0.1
    tau = 2

    SLm = 1
    # SLm = 2
    # SLm = 5
    # SLm = -1

    tStart = 0
    T = 300

    dt = 0.1
    nQ_point1 = 5

    FV_ADI_2d(gameName, A, Qmin, Qmax, d, alpha, tau, T, dt, nQ_point1, initDist, initPara, tStart, SLm)
    print('done',datetime.datetime.now())


if __name__ == '__main__':
    main()




