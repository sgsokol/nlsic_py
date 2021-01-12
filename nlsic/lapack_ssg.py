import numpy as np
ar2f = np.asfortranarray
import nlsic.fpylapack as la
# print fpylapack.__doc__
def isar(x):
    return isinstance(x, np.ndarray)
class lapack_ssg(Exception):
    pass
class QR:
    """QR class for storing and dealing with QR decomposition by lapack.
    Input matrix a and right hand side b are modified by QR() and solve()
    respectivly if they are in fortran storage else a copy is made.
    If you need just to solve a full column rank, overdetermined
    linear system a*x=b and don't need Q, R, rank and so on, use
    ls_solve(a, b) instead of this class.
    ls_solve() based on dgels lapack routine is slightly quiker for this task"""

    def __init__(self, a, rcond=np.finfo(float).eps*100):
        self.nrow, self.ncol = a.shape
        # make factorization
        # NB: q and r are permuted (permutation is given by pivot)
        self.qr, self.pivot, self.tau = la.dgeqp3(a)
        self.pivot -= 1
        d = np.diag(self.qr)
        self.rank = np.sum(np.abs(d) >= np.abs(d[0])*rcond)
        # packed upper triangular
        self.rp = ar2f(self.qr.T[np.tri(self.ncol, self.nrow, dtype=np.bool)])

    def Q(self):
        """get Q matrix from QR form returned by lapack"""
        try:
            q = self.q
        except AttributeError:
            self.Qc()
            q = self.qc[:, :self.ncol].copy("f")
        self.q = q
        return q

    def Qc(self):
        """get _complete_ (m x m) Q matrix from QR form returned by lapack"""
        try:
            qc = self.qc
        except AttributeError:
            #qc = self.qr.copy("f")
            #qc = np.resize(qc, (self.nrow, self.nrow))
            ##qc=la.dormqr(self.qr, self.tau, np.mat(np.eye(self.nrow)).T)
            #qc = la.dorgqr(qc, self.tau)
            qc=self.qy(np.eye(self.nrow)).copy("f")
        self.qc = qc
        return qc

    def R(self):
        """get R matrix from QR form returned by lapack"""
        try:
            r = self.r
        except AttributeError:
            n = self.ncol
            r = np.triu(self.qr[:n, ])
        self.r = r
        return r

    def solve(self, b=None):
        if b is None:
            b = np.mat(np.eye(self.nrow)).T
        s = la.dtrtrs(self.qr, la.dormqr(
            self.qr, self.tau, b, trans='T'))[:self.ncol, :]
        x = np.empty(s.shape, dtype=np.float)
        x[self.pivot, :] = s
        return x

    def qy(self, y):
        """multiply Q*y (y is modified in place if F-contiguous"""
        return la.dormqr(self.qr, self.tau, y, trans='N')

    def qty(self, y):
        """multiply Qt*y"""
        return la.dormqr(self.qr, self.tau, y, trans='T')[:self.ncol]

    def Null(self):
        """Claculate basis (column vectors) of Null space of a.T (nat)
        such that a.T*nat=0. rank(nat)=nrow(a)-rank(a)
        """
        try:
            return self.null
        except AttributeError:
            self.null = np.matrix(self.Qc()[:, self.rank:])
        return self.null
def ls_solve(a, b):
    """
    Wraper for lapack dgels routine.
    Solve over- or under-determined linear system a*x=b by QR or LR decomposition
    return x which has dimensions nrow(x)==ncol(a) and ncol(x)==ncol(b).
    Thus, multiple right hand sides in b are allowed.
    If a has been already QR factorized by QR(a), you can use qra.solve(b)
    where qra is a QR instance.
    If the system is under-determined, the least norm solution is returned.
    If a and b are F-contiguous they are modified "in place".
    You may want to protect them by copying before call.
    If a and b are _not_ F-contiguous they are copied inside the call to dgels
    (a warning is printed on stderr if -DF2PY_REPORT_ON_ARRAY_COPY was set
    during f2py compiling) so the original a and b are _not_ modified"""
    x = la.dgels(a, b)
    return x[:a.shape[1]]
def tri_solve(a, b, uplo='U', trans='N'):
    """Solve triangular system a*x=b or at*x=b (trans='N' or 'T')
    The matrix a can represent upper or lower triangular system (uplo='U' or 'L')
    Other part of rectangular matrix a is not referenced.
    b is modified "in place" if it is F-contiguous.
    """
    return la.dtrtrs(a, b, uplo=uplo, trans=trans)
if __name__ == "__main__":
    from time import time
    n = 100
    m = 2*n
    decimal = np.finfo(float).precision-1-int(np.log10(n))
    a = np.ones((m, n), order="F")
    a *= np.random.rand(m, n)
    a = np.asmatrix(a)
    x_true = ar2f(np.asmatrix(np.random.rand(n, 1)))
    b = ar2f(a*x_true)
    t = time()
    x = ls_solve(a.copy('F'), b.copy('F'))
    print("time for ls_solve()=", time()-t, "s")
    np.testing.assert_array_almost_equal(x, x_true, decimal)
    print("n=", n, "; max err=", np.max(np.abs(x-x_true)))
    # test if the result is the same
    t = time()
    x = QR(a.copy('F')).solve(b.copy('F'))
    print("time for QR().solve()=", time()-t, "s")
    print("n=", n, "; max err=", np.max(np.abs(x-x_true)))
    # print "x,x_true=", x, x_true
    np.testing.assert_array_almost_equal(x, x_true, decimal)
    t = time()
    x = QR(a.copy('F'))
    print("time for QR()=", time()-t, "s")
    qa = QR(a.copy('F'))
    t = time()
    x = qa.solve(b)
    print("time for qa.solve()=", time()-t, "s")
    # test Null space
    t = time()
    na = qa.Null()
    print("time for Null()=", time()-t, "s")
    np.testing.assert_array_almost_equal(
        a.T*qa.Null(), np.zeros((qa.ncol, qa.nrow-qa.rank)), decimal)
    np.testing.assert_array_almost_equal(QR(na).rank, qa.nrow-qa.rank, decimal)
