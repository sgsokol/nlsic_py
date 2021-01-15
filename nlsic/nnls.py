"""
This module provides solvers for linear system
with inequality constraints. Theay are based on
nnls algorithm and software originaly designed
and (re)written in Fortran77 by [Lawson & al, 1974].

:nnls: solve Non Negative Least Square problem a*x~b, subject to x_i >= 0 for all indexes i
:ldp: solve Least Distance Problem x~0 subject to u*x>=c. LDP is solved by reducing this problem to nnls problem.
:lsi: solve Least Square problem with Inequality constraints a*x~b subject to u*x>=c. LSI problem is solved by reducing it to ldp problem. If matrices a and u are F-contiguous, they can be modified "in place".
:lsi_ln: the same as lsi but find least norm solution for rank-deficient matrix a.
"""
import numpy as np
teq = np.testing.assert_array_almost_equal
ar2f = np.asfortranarray
mat = np.mat
norm = np.linalg.norm
any = any

from nlsic.lapack_ssg import QR, ls_solve, tri_solve
from nlsic.nnls_f77 import nnlsr
from nlsic.nnls_f90 import nnls as nnlsf
class Obj:
    """
    tarnsform a dictionnary into object to allow access like obj.field
    """

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)
class nnlsError(Exception):
    pass
def cV(x):
    """converts x to column vector (as matrix with one column)"""
    return np.asmatrix(x).reshape(len(x), -1)
def nnls(a, b, indx=None, nsetp=None):
    """wrapper for nnlsf call
    
    :param a: m x n real matrix, m >= n
    :param b: m x 1
    :return: an object with fields:
    
        :x: solution vector of length n
        :rnorm: norm of residual ax-b
        :w: the dual solution vector of length n. w will satisfy w[i] = 0. for all i in set p  and w[i] <= 0. for all i in set z ,
        :indx: p and z set. indx[:nsetp] is a p-set indexes, while indx[nsetp:] is z-set indexes
        :nsetp: p and z set separator in indx
"""
    n = a.shape[1]
    if True or (indx is None or nsetp is None):
        # till weird bug in intializing nnlsr is not solved go always here
        ret = Obj(**dict(zip(("x", "rnorm", "w", "indx", "mode", "nsetp"),
                             nnlsf(a, b))))
        ret.x = cV(ret.x)
    else:
        indx += 1
        #print (type(a), type(b), type(indx), type(nsetp))
        ret = Obj(**dict(zip(("x", "rnorm", "w", "mode"),
                             nnlsr(a, b, indx, nsetp))))
        ret.indx = indx
        ret.nsetp = nsetp
    mode = ret.mode
    del(ret.mode)
    if mode == 3:
        raise nnlsException(
            "Iteration count exceeded.  more than 3*n iterations.")
    ret.indx -= 1
    return ret
def ldp(u, co, indx=None, nsetp=None):
    """solve Least Distance Problem: min(x**2) subject to u*x>=co
    by reducing it to nnls problem.
    Params:
    u - m x n real matrix, m <= n, u is of full rank
    co - m real vector
    indx - m int vector
    nsetp - 1 int vector
    Return a real vector of lenght n.
    indx and nsetp are modified in place.
    Raise an exception if constraints are unfeasible
    u and co are not modified.
    """
    m, n = u.shape
    if m == 0:
        # trivial case, no constraint
        return np.zeros((n,))
    e = np.vstack((u.T, co.ravel()))
    # f=np.vstack((np.zeros((n,1)),1.))
    f = ar2f(np.zeros((n+1, 1), dtype=np.float))
    f[n] = 1.
    resnnls = nnls(e.copy("f"), f, indx=indx, nsetp=nsetp)
    if resnnls.rnorm == 0.:
        raise nnlsError("Constraints are unfeasible")
    r = np.dot(e, resnnls.x)
    return r[:n]/((1.-r[n]))
def lsi(a, b, u=None, co=None, indx=None, nsetp=None):
    """
    solve linear Least Square problem (min ||a*x-b||)
    subject to Inequalities u*x>=co by reducing it to LDP problem
    Return x
    indx, nsetp are modified in place.
    a and b are modified in place too.
    Raise an exception if a is not of full rank
    """
    if u is None or co is None or u.shape[0] == 0:
        # no inequalities, solve just ls problem
        return ls_solve(ar2f(a), ar2f(b))
    m, n = a.shape
    a = mat(a)
    b = cV(b)
    aqr = QR(a)
    x0 = aqr.solve(ar2f(b))
    if aqr.rank < n:
        raise nnlsError("lsi: matrix a is not of full rank")
    u = mat(u)
    co = cV(co)
    cou = co-u*x0
    if all(cou <= 0.):
        # all inequalities are satisfied in a global minimum
        return x0
    # prepare variable change
    ut = tri_solve(aqr.qr, ar2f(u[:, aqr.pivot].T), trans="T").T
    xa = ldp(ut, cou, indx=indx, nsetp=nsetp)
    x = np.empty((n, 1), dtype=np.float, order="f")
    xa = tri_solve(aqr.qr, ar2f(xa))
    x[aqr.pivot] = xa
    x += x0
    # round errors can occur => slightly degrade residual but enforce inequalities
    # cou=co-u*x
    # if any(cou>0.):
    #    x=x+ldp(u, cou);
    return x
def lsi_ln(a, b, u=None, co=None, indx=None, nsetp=None):
    """
    solve linear Least Square problem (min ||a*x-b||)
    subject to Inequalities u*x>=co by reducing it to LDP problem
    If matrix a is not of full rank, a least norm solution is provided.
    Return x
    indx, nsetp are modified in place.
    a and b are modified in place too.
    """
    # if u is None or co is None or u.shape[0]==0:
    ineq = not (u is None or co is None or u.shape[0] == 0)
    if ineq:
        u = mat(u)
        co = cV(co)
    #    # no inequalities, solve just ls problem
    #    return ls_solve(ar2f(a), ar2f(b))
    m, n = a.shape
    a = mat(a) # modify 'a' in place to be able to return a decomposed to the caller level
    b = cV(b)
    aqr = QR(a)
    rdefic = aqr.rank < n
    if rdefic:
        # prepare least norm solution
        q = aqr.Q()[:, :aqr.rank]
        r = aqr.R()[:aqr.rank, :]
        tv = QR(r.T.copy("f"))
        #print(r.shape, tv.qr.shape)
        t = tv.R().T
        v = tv.Q()
        x = v*tri_solve(t, q.T*b, uplo="L")
        x0 = x.copy()
        x0[aqr.pivot] = x
    else:
        x0 = aqr.solve(b)
    if ineq:
        cou = co-u*x0
        if all(cou <= 0.):
            # all inequalities are satisfied in a global minimum
            return x0
        if rdefic:
            ut = tri_solve(
                t, ar2f((u[:, aqr.pivot]*v).T), uplo="L", trans="T").T
        else:
            ut = tri_solve(aqr.qr, ar2f(u[:, aqr.pivot].T), trans="T").T
    else:
        # no inequalities, return global minimum
        return x0
    # solve ldp
    xa = ldp(ut, cou, indx=indx, nsetp=nsetp)
    x = np.empty((n, 1), dtype=np.float, order="f")
    if rdefic:
        xa = v*tri_solve(t, xa, uplo="L")
    else:
        xa = tri_solve(aqr.qr, ar2f(xa))
    x[aqr.pivot] = xa
    x += x0
    # round errors can occur => slightly degrade residual but enforce inequalities
    # cou=co-u*x
    # if any(cou>0.):
    #    x=x+ldp(u, cou);
    return x
if __name__ == "__main__":
    from time import time
    n = 50
    m = 2*n
    decimal = np.finfo(float).precision-1-int(np.log10(n))
    print("decimal =", decimal)
    # must be F contigous if a copying is not desired
    a = mat(ar2f(np.random.rand(m, n)))
    x_true = cV(np.abs(np.random.rand(n)))
    b = a*x_true
    # ---test nnls
    t = time()
    ret = nnls(a.copy("f"), b.copy("f"))
    print("time for nnls:", time()-t)
    try:
        teq(ret.x, x_true, decimal)
    except:
        print("resid =", a*ret.x-b)
        print("diff=", (ret.x-x_true)[np.abs(ret.x-x_true) > 10**(-decimal)])
        raise

    # ---test ldp
    u = np.eye(n)
    co = x_true
    t = time()
    x = ldp(u, co)
    print("time for ldp:", time()-t)
    teq(x, x_true, decimal)

    # ---test lsi
    decimal -= 4
    print("decimal =", decimal)
    t = time()
    x = lsi(a, b, u, co)
    print("max diff=", np.max(np.abs(x-x_true)))
    # print "min u=", np.min(u*x-co)
    print("max resid =", np.max(np.abs(a*x-b)))
    # x += lsi(a, b-a*x, u, co-u*x) # not so much better
    # print "time for lsi:", time()-t
    # print "max diff2=", np.max(np.abs(x-x_true))
    # print "min u2=", np.min(u*x-co)
    # print "max resid =", np.max(np.abs(a*x-b))
    teq(x, x_true, decimal)

    # ---test lsi_ln
    print("decimal =", decimal)
    t = time()
    x = lsi_ln(a, b, u, co)
    print("max diff=", np.max(np.abs(x-x_true)))
    # print "min u=", np.min(u*x-co)
    print("max resid =", np.max(np.abs(a*x-b)))
    # x += lsi(a, b-a*x, u, co-u*x) # not so much better
    # print "time for lsi:", time()-t
    # print "max diff2=", np.max(np.abs(x-x_true))
    # print "min u2=", np.min(u*x-co)
    # print "max resid =", np.max(np.abs(a*x-b))
    teq(x, x_true, decimal)
