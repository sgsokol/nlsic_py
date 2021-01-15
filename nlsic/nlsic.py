"""A set of functions for solving Non-linear Least Squares with Inequality constraints.
Main function is ``nlsic(...)``
"""
import numpy as np
ar2f = np.asfortranarray
any = np.any
sqrt = np.sqrt
teq = np.testing.assert_array_almost_equal

import numpy.linalg as la
norm = la.norm

import nlsic.lapack_ssg as las
QR = las.QR
from nlsic.nnls import lsi, lsi_ln, ldp
class nlsicError(Exception):
    "custom error class"
    pass
class Obj:
    """
    tarnsform a dictionnary into object to allow access like o.field
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
def cV(x):
    """converts x to column vector (as matrix with one column)"""
    return np.matrix(x).reshape(len(x), -1)
def numdif(f, x, *kargs, **kwrags):
    """
    numericaly derivate vector function f at point x (by central diff)
    and return its Jacobian dfi_dxj
    """
    eps2 = np.finfo(float).eps
    eps = sqrt(eps2)
    n = len(x)
    xh = np.empty(n, dtype=np.float)
    for j in range(n):
        h = x[j]*eps
        h = eps if h <= eps2 else h
        xh.flat = x
        xh[j] += h
        fh = f(xh)
        if j == 0:
            jac = np.mat(np.empty((len(fh), n))).copy("f")
        jac[:, j].flat = fh
        xh[j] -= h+h
        jac[:, j].flat = (jac[:, j]-cV(f(xh)))/(h+h)
    return jac
def norm2(x):
    "L2 vector norm"
    return np.dot(x.flat, x.flat)
def nlsic(par, r, u=None, co=None, control={}, e=None, eco=None, flsi=lsi, *kargs, **kwargs):
    """solve non linear least square problem min of ||r(par,\*kargs, \*\*kwargs).res||_2
    with optional inequality constraints u*par>=co
    and optional equality constraints e*par=eco
    
    *Method* sequential lsi globalized by backtracking technique. If e, eco are not None, reduce jacobian before lsi() call.
    
    *Notes* If function r() return an object having attribute "jacobian"
    when cjac==True it is supposed to be equal to a matrix jacobian dr_dpar.
    Else numerical derivation is used for its construction.
    
    :param par: initial values for parameter vector (can be in non feasible domain)
        At return it contains the result vector (i.e. modified in-place).
    :type par: numpy array (float64)
    :param r: function calculating residual vector
        by a call to r(par, cjac=True|False, \*kargs, \*\*kwargs)
        where
        
            :par: is a current parameter vector,
            :cjac: is logical indicating if we need or not a Jacobean together with residuals.
            :\*kargs: positional parameters passed through to r()
            :\*\*kwargs: keyword=value parameters passed through to r()
            
        The call to r() must return an oject having an attribute "res"
        containing the residual vector and optionnaly an attribute "jacobian"
        when cjac is set to TRUE.
        
    :type r: function
    :param u: linear inequality constraint matrix in u\*par>=co. Defaults to None.
    :type u: numpy matrix (float64), optional
    :param co: inequality constraints vector
    :type co: numpy vector (float64), optional
    :param controls: control parameters (=default values which is assumed if an entry is missing):
        
        :tolx=1.e-7: error on L2 norm of the iteration step sqrt(pt\*p).
        :maxit=100: maximum of newton iterations
        :btstart=1.: (0;1) starting value for backtracking
        :btfrac=0.5: (0;1) by this value we diminish the step till tight up to the quadratic model of norm reduction in backtrack (bt) iterations
        :btdesc=0.5: (0;1) how good we have to tight up to the quadratic model. 0-we are very relax, 1 we are very close (may need many bt iterations)
        :btmaxit=15: maximum of backtrack iterations
        :trace=0: print tracing information on stdout
        
        Defaults to empty dict {} which is equivalent to the above default values.
        
    :type controls: dict, optional
    :param e: linear equality constraint matrix in e\*par=eco. Defaults to None.
    :type e: numpy matrix (float64), optional
    :param eco: equality constraints vector
    :type eco: numpy vector (float64), optional
    :param flsi: solution of least squares with inequality constraints. Defaults to nlsic.lsi. Can be nlsic.lsi_ln if least norm increment is required.
    :type flsi: function
    :param kargs: optional positional arguments to be passed through to r function
    :type kargs: list, optional
    :param kwargs: optional keyword=value parameters to pass through to r function
    :type kwargs: dict, optional
    :return: result Object with the following fields:
    
        :par: resulting parameter vector
        :laststep: last increment after a possible back-tracking contraction
        :normp: L2 norm of the last increment before back-tracking
        :res: last residual vector
        :prevres: previous residual vector
        :indx: vector of p and z sets. indx[:nsetp] is a p-set indexes, while indx[nsetp:] is z-set indexes
        :nsetp: size of p-set.
        :it: number of non-linear iterations
        :btit: number of back-tracking iterations at the last non linear iteration
        :error: execution error code: 0 means no error occurred
        :a: last Jacobean calculated
        :nte: null-space basis matrix if e is provided
        :mes: if error != 0, then str with informative message.
        
    :rtype: Object
"""
    n = len(par)
    m, co = (len(co), cV(co)) if co is not None else (0, np.empty((0, n)))
    par = cV(par)

    # predefined controls
    con = {"tolx": 1.e-7, "maxit": 100, "btstart": 1., "btfrac": 0.5, "btdesc": 0.5,
           "btmaxit": 15, "trace": 0}
    nmsC = set(con.keys())
    namc = set(control.keys())
    con.update(control)
    con = Obj(**con)

    noNms = namc - nmsC
    if noNms:
        raise nlsicError("unknown names in control: ", ", ".join(noNms))
    if e is not None:
        et = np.matrix(e).reshape((-1, n), order="F").T
        eco = cV(eco)
    econstr = e is not None and eco is not None and e.shape[0] > 0 and e.shape[1] == n
    # print "e, eco=", e, eco
    # make u matrix if u is None
    if u is None:
        u = np.zeros((0, n))
        co = np.zeros((0,))
    u = np.matrix(u)
    co = np.matrix(co)
    # prepare indx and nsetp for nnls
    indx = np.empty(m, dtype=np.int)
    nsetp = np.zeros(1, dtype=np.int)
    if econstr:
        # affine transform epar -> par s.t. e*par=eco
        # particular solution
        from nlsic.lapack_ssg import QR, tri_solve
        et = QR(ar2f(et))
        tmp=np.zeros((e.shape[1],))
        tmp[:len(eco)]=tri_solve(et.qr, eco[et.pivot], trans='T').copy()
        parp = cV(et.qy(tmp))
        # par=parp+Null(t(e))%*%epar, where epar is new parameter vector
        nte = et.Null()
        ue = u.dot(nte)
        epar = nte.T.dot(par-parp)
        par[:] = nte.dot(epar)+parp
    else:
        nte=None

    cou = co-u.dot(par)
    if any(cou > 0.):
        # project par into feasible domain
        # just in case if residual function is defined only on feasible domain
        if econstr:
            x = cV(ldp(ue, cou))
            par[:] = par+nte.dot(x)
        else:
            x = cV(ldp(u, cou))
            par[:] = par+x

    # newton globalized iterations
    it = 0
    btit = 0
    converged = False
    #import pdb; pdb.set_trace()
    while (not converged) and it < con.maxit:
        if it == 0:
            # residual vector
            lres = r(par, cjac=True, *kargs, **kwargs)
            res = lres.res
            b = -res.reshape(res.size, 1)
            norb = norm(b)
            norres = norb
            k = 0
            if con.trace:
                print("it={}\tres={}".format(it, norb))
        else:
            b = -res
            norb = norres
        # jacobian
        try:
            a = lres.jacobian
            if a is None:
                a = r(par, cjac=TRUE, *kargs, **kwargs).jacobian
        except AttributeError:
            a = numdif(lambda x: r(x, cjac=False, *kargs, **kwargs).res, par)
        # solve linear ls
        if econstr:
            ajac=a.dot(nte)
            p = flsi(ajac.copy("f"), b.copy("f"), ue, co-u.dot(par),
                     indx=indx, nsetp=nsetp)  # newton direction
            p = cV(nte.dot(p))
        else:
            ajac=a
            p = cV(flsi(ajac.copy("f"), b.copy("f"), u, co-u.dot(par),
                     indx=indx, nsetp=nsetp))  # newton direction
        # print "p=", (p).T
        # print nsetp
        # nsetp.flat=0
        normp = norm(p)
        converged = (normp <= con.tolx)
        if converged:
            # no need for backtracking at this stage
            laststep = p
            par = par+laststep
            it = it+1
            btit = 0
            resdecr = None
            res = r(par, cjac=False, *kargs, **kwargs).res
            norres = norm(res)
            if con.trace and (it < 10 or not it % 10):
                print("it={}\tres={}\tnormstep={}\tbtk={}".format(
                    it, norres, normp, k/con.btfrac))
            break
        ap = a.dot(p)
        n2ap = norm2(ap)
        k = con.btstart  # fraction of p
        # backtrack iterations
        btit = 0
        descending = False
        while (not descending) and btit < con.btmaxit:
            laststep = k*p
            lres = r(par+laststep, cjac=False, *kargs, **kwargs)
            res = lres.res
            res = res.reshape(res.size, 1)
            norres = norm(res)
            # print "par+%3.2f*p="%k, (par+laststep).T
            # print "norres=", norres
            scaprod = np.dot(ap.T, (res+b))[0,0]
            descending = (
                scaprod >= 0. and
                sqrt(scaprod) >= con.btdesc*sqrt(n2ap*k) and
                norres < norb
            )
            k = k*con.btfrac
            btit = btit+1
        par[:] = par+laststep
        it = it+1
        if con.trace and (it < 10 or not it % 10):
            print("it={}\tres={}\tnormstep={}\tbtk={}".format(
                it, norres, normp, k/con.btfrac))
    if con.trace and not (it < 10 or not it % 10):
        print("it={}\tres={}\tnormstep={}\tbtk={}".format(
            it, norres, normp, k/con.btfrac))
    mes = []
    if it >= con.maxit:
        mes = ["Maximal non linear iteration number is achieved"]
    if btit >= con.btmaxit:
        mes.append("Maximal backtrack iteration number is achieved")
    # restore names
    return Obj(
        par=cV(par),
        laststep=cV(laststep),
        normp=normp,
        res=cV(res),
        prevres=-cV(b),
        indx=indx,
        nsetp=nsetp,
        it=it,
        btit=btit,
        error=0,
        a=ajac,
        nte=nte,
        mes=mes)

if __name__ == "__main__":
    from time import time
    # test numdif()

    def f(x):
        return x[1:]**2-x[:-1]**2
    x = np.arange(5)
    # print numdif(f, x)

    # test nlsic
    def r(par, cjac):
        n = len(par)
        res = np.sin(np.pi*0.25*(par.ravel()-x_true.ravel()))
        return Obj(res=res)
    n = 500
    par = cV(np.zeros(n))
    x_true = (np.arange(n)-n/2.5)/(n+1.)
    # print "x_true=", x_true
    u = np.eye(n, dtype=np.float)
    co = np.zeros((n, 1), dtype=np.float)
    # decimal=np.finfo(float).precision-1-int(np.log10(n))
    decimal = 4
    print("n, decimal=", n, decimal)
    t = time()
    ret = nlsic(par, r, u=u, co=co,
                control={"trace": 1, "tolx": 10.**(-decimal), "maxit": 5, "btmaxit": 15})
    print("time for min r=", time()-t)
    # print ret.__dict__
    teq(ret.par.flat, np.maximum(x_true, 0.).flat, decimal)
