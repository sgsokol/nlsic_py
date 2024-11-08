/* File: nnls_f77module.c
 * This file is auto-generated with f2py (version:2.0.1).
 * f2py is a Fortran to Python Interface Generator (FPIG), Second Edition,
 * written by Pearu Peterson <pearu@cens.ioc.ee>.
 * Generation date: Mon Nov  4 14:00:50 2024
 * Do not edit this file directly unless you know what you are doing!!!
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif /* PY_SSIZE_T_CLEAN */

/* Unconditionally included */
#include <Python.h>
#include <numpy/npy_os.h>

/*********************** See f2py2e/cfuncs.py: includes ***********************/
#include "fortranobject.h"
#include <math.h>

/**************** See f2py2e/rules.py: mod_rules['modulebody'] ****************/
static PyObject *nnls_f77_error;
static PyObject *nnls_f77_module;

/*********************** See f2py2e/cfuncs.py: typedefs ***********************/
/*need_typedefs*/

/****************** See f2py2e/cfuncs.py: typedefs_generated ******************/
/*need_typedefs_generated*/

/********************** See f2py2e/cfuncs.py: cppmacros **********************/

#if defined(PREPEND_FORTRAN)
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F
#else
#define F_FUNC(f,F) _##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F##_
#else
#define F_FUNC(f,F) _##f##_
#endif
#endif
#else
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F
#else
#define F_FUNC(f,F) f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F##_
#else
#define F_FUNC(f,F) f##_
#endif
#endif
#endif
#if defined(UNDERSCORE_G77)
#define F_FUNC_US(f,F) F_FUNC(f##_,F##_)
#else
#define F_FUNC_US(f,F) F_FUNC(f,F)
#endif


/* See fortranobject.h for definitions. The macros here are provided for BC. */
#define rank f2py_rank
#define shape f2py_shape
#define fshape f2py_shape
#define len f2py_len
#define flen f2py_flen
#define slen f2py_slen
#define size f2py_size


#define CHECKSCALAR(check,tcheck,name,show,var)\
    if (!(check)) {\
        char errstring[256];\
        sprintf(errstring, "%s: "show, "("tcheck") failed for "name, var);\
        PyErr_SetString(nnls_f77_error,errstring);\
        /*goto capi_fail;*/\
    } else 

#ifdef DEBUGCFUNCS
#define CFUNCSMESS(mess) fprintf(stderr,"debug-capi:"mess);
#define CFUNCSMESSPY(mess,obj) CFUNCSMESS(mess) \
    PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\
    fprintf(stderr,"\n");
#else
#define CFUNCSMESS(mess)
#define CFUNCSMESSPY(mess,obj)
#endif


#ifndef max
#define max(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) ((a < b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? (a) : (b))
#endif


/************************ See f2py2e/cfuncs.py: cfuncs ************************/

static int
int_from_pyobj(int* v, PyObject *obj, const char *errmess)
{
    PyObject* tmp = NULL;

    if (PyLong_Check(obj)) {
        *v = Npy__PyLong_AsInt(obj);
        return !(*v == -1 && PyErr_Occurred());
    }

    tmp = PyNumber_Long(obj);
    if (tmp) {
        *v = Npy__PyLong_AsInt(tmp);
        Py_DECREF(tmp);
        return !(*v == -1 && PyErr_Occurred());
    }

    if (PyComplex_Check(obj)) {
        PyErr_Clear();
        tmp = PyObject_GetAttrString(obj,"real");
    }
    else if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        /*pass*/;
    }
    else if (PySequence_Check(obj)) {
        PyErr_Clear();
        tmp = PySequence_GetItem(obj, 0);
    }

    if (tmp) {
        if (int_from_pyobj(v, tmp, errmess)) {
            Py_DECREF(tmp);
            return 1;
        }
        Py_DECREF(tmp);
    }

    {
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
            err = nnls_f77_error;
        }
        PyErr_SetString(err, errmess);
    }
    return 0;
}


static int
double_from_pyobj(double* v, PyObject *obj, const char *errmess)
{
    PyObject* tmp = NULL;
    if (PyFloat_Check(obj)) {
        *v = PyFloat_AsDouble(obj);
        return !(*v == -1.0 && PyErr_Occurred());
    }

    tmp = PyNumber_Float(obj);
    if (tmp) {
        *v = PyFloat_AsDouble(tmp);
        Py_DECREF(tmp);
        return !(*v == -1.0 && PyErr_Occurred());
    }

    if (PyComplex_Check(obj)) {
        PyErr_Clear();
        tmp = PyObject_GetAttrString(obj,"real");
    }
    else if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        /*pass*/;
    }
    else if (PySequence_Check(obj)) {
        PyErr_Clear();
        tmp = PySequence_GetItem(obj, 0);
    }

    if (tmp) {
        if (double_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}
        Py_DECREF(tmp);
    }
    {
        PyObject* err = PyErr_Occurred();
        if (err==NULL) err = nnls_f77_error;
        PyErr_SetString(err,errmess);
    }
    return 0;
}


/********************* See f2py2e/cfuncs.py: userincludes *********************/
/*need_userincludes*/

/********************* See f2py2e/capi_rules.py: usercode *********************/


/* See f2py2e/rules.py */
extern void F_FUNC(nnlsr,nnlsr)(double*,int*,int*,int*,double*,double*,double*,double*,double*,int*,int*,int*);
/*eof externroutines*/

/******************** See f2py2e/capi_rules.py: usercode1 ********************/


/******************* See f2py2e/cb_rules.py: buildcallback *******************/
/*need_callbacks*/

/*********************** See f2py2e/rules.py: buildapi ***********************/

/*********************************** nnlsr ***********************************/
static char doc_f2py_rout_nnls_f77_nnlsr[] = "\
nnlsr(A,B,X,RNORM,W,ZZ,INDEX,MODE,NSETP,[MDA,M,N])\n\nWrapper for ``nnlsr``.\
\n\nParameters\n----------\n"
"A : input rank-2 array('d') with bounds (MDA,N)\n"
"B : input rank-1 array('d') with bounds (M)\n"
"X : input rank-1 array('d') with bounds (N)\n"
"RNORM : input float\n"
"W : input rank-1 array('d') with bounds (N)\n"
"ZZ : input rank-1 array('d') with bounds (M)\n"
"INDEX : input rank-1 array('i') with bounds (N)\n"
"MODE : input int\n"
"NSETP : input int\n"
"\nOther Parameters\n----------------\n"
"MDA : input int, optional\n    Default: shape(A, 0)\n"
"M : input int, optional\n    Default: shape(B, 0)\n"
"N : input int, optional\n    Default: shape(A, 1)";
/* extern void F_FUNC(nnlsr,nnlsr)(double*,int*,int*,int*,double*,double*,double*,double*,double*,int*,int*,int*); */
static PyObject *f2py_rout_nnls_f77_nnlsr(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(double*,int*,int*,int*,double*,double*,double*,double*,double*,int*,int*,int*)) {
    PyObject * volatile capi_buildvalue = NULL;
    volatile int f2py_success = 1;
/*decl*/

    double *A = NULL;
    npy_intp A_Dims[2] = {-1, -1};
    const int A_Rank = 2;
    PyArrayObject *capi_A_as_array = NULL;
    int capi_A_intent = 0;
    PyObject *A_capi = Py_None;
    int MDA = 0;
    PyObject *MDA_capi = Py_None;
    int M = 0;
    PyObject *M_capi = Py_None;
    int N = 0;
    PyObject *N_capi = Py_None;
    double *B = NULL;
    npy_intp B_Dims[1] = {-1};
    const int B_Rank = 1;
    PyArrayObject *capi_B_as_array = NULL;
    int capi_B_intent = 0;
    PyObject *B_capi = Py_None;
    double *X = NULL;
    npy_intp X_Dims[1] = {-1};
    const int X_Rank = 1;
    PyArrayObject *capi_X_as_array = NULL;
    int capi_X_intent = 0;
    PyObject *X_capi = Py_None;
    double RNORM = 0;
    PyObject *RNORM_capi = Py_None;
    double *W = NULL;
    npy_intp W_Dims[1] = {-1};
    const int W_Rank = 1;
    PyArrayObject *capi_W_as_array = NULL;
    int capi_W_intent = 0;
    PyObject *W_capi = Py_None;
    double *ZZ = NULL;
    npy_intp ZZ_Dims[1] = {-1};
    const int ZZ_Rank = 1;
    PyArrayObject *capi_ZZ_as_array = NULL;
    int capi_ZZ_intent = 0;
    PyObject *ZZ_capi = Py_None;
    int *INDEX = NULL;
    npy_intp INDEX_Dims[1] = {-1};
    const int INDEX_Rank = 1;
    PyArrayObject *capi_INDEX_as_array = NULL;
    int capi_INDEX_intent = 0;
    PyObject *INDEX_capi = Py_None;
    int MODE = 0;
    PyObject *MODE_capi = Py_None;
    int NSETP = 0;
    PyObject *NSETP_capi = Py_None;
    static char *capi_kwlist[] = {"A","B","X","RNORM","W","ZZ","INDEX","MODE","NSETP","MDA","M","N",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
    if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
        "OOOOOOOOO|OOO:nnls_f77.nnlsr",\
        capi_kwlist,&A_capi,&B_capi,&X_capi,&RNORM_capi,&W_capi,&ZZ_capi,&INDEX_capi,&MODE_capi,&NSETP_capi,&MDA_capi,&M_capi,&N_capi))
        return NULL;
/*frompyobj*/
    /* Processing variable MODE */
        f2py_success = int_from_pyobj(&MODE,MODE_capi,"nnls_f77.nnlsr() 8th argument (MODE) can't be converted to int");
    if (f2py_success) {
    /* Processing variable NSETP */
        f2py_success = int_from_pyobj(&NSETP,NSETP_capi,"nnls_f77.nnlsr() 9th argument (NSETP) can't be converted to int");
    if (f2py_success) {
    /* Processing variable A */
    ;
    capi_A_intent |= F2PY_INTENT_IN;
    const char * capi_errmess = "nnls_f77.nnls_f77.nnlsr: failed to create array from the 1st argument `A`";
    capi_A_as_array = ndarray_from_pyobj(  NPY_DOUBLE,1,A_Dims,A_Rank,  capi_A_intent,A_capi,capi_errmess);
    if (capi_A_as_array == NULL) {
        PyObject* capi_err = PyErr_Occurred();
        if (capi_err == NULL) {
            capi_err = nnls_f77_error;
            PyErr_SetString(capi_err, capi_errmess);
        }
    } else {
        A = (double *)(PyArray_DATA(capi_A_as_array));

    /* Processing variable B */
    ;
    capi_B_intent |= F2PY_INTENT_IN;
    const char * capi_errmess = "nnls_f77.nnls_f77.nnlsr: failed to create array from the 2nd argument `B`";
    capi_B_as_array = ndarray_from_pyobj(  NPY_DOUBLE,1,B_Dims,B_Rank,  capi_B_intent,B_capi,capi_errmess);
    if (capi_B_as_array == NULL) {
        PyObject* capi_err = PyErr_Occurred();
        if (capi_err == NULL) {
            capi_err = nnls_f77_error;
            PyErr_SetString(capi_err, capi_errmess);
        }
    } else {
        B = (double *)(PyArray_DATA(capi_B_as_array));

    /* Processing variable RNORM */
        f2py_success = double_from_pyobj(&RNORM,RNORM_capi,"nnls_f77.nnlsr() 4th argument (RNORM) can't be converted to double");
    if (f2py_success) {
    /* Processing variable M */
    if (M_capi == Py_None) M = shape(B, 0); else
        f2py_success = int_from_pyobj(&M,M_capi,"nnls_f77.nnlsr() 2nd keyword (M) can't be converted to int");
    if (f2py_success) {
    CHECKSCALAR(shape(B, 0) == M,"shape(B, 0) == M","2nd keyword M","nnlsr:M=%d",M) {
    /* Processing variable MDA */
    if (MDA_capi == Py_None) MDA = shape(A, 0); else
        f2py_success = int_from_pyobj(&MDA,MDA_capi,"nnls_f77.nnlsr() 1st keyword (MDA) can't be converted to int");
    if (f2py_success) {
    CHECKSCALAR(shape(A, 0) == MDA,"shape(A, 0) == MDA","1st keyword MDA","nnlsr:MDA=%d",MDA) {
    /* Processing variable N */
    if (N_capi == Py_None) N = shape(A, 1); else
        f2py_success = int_from_pyobj(&N,N_capi,"nnls_f77.nnlsr() 3rd keyword (N) can't be converted to int");
    if (f2py_success) {
    CHECKSCALAR(shape(A, 1) == N,"shape(A, 1) == N","3rd keyword N","nnlsr:N=%d",N) {
    /* Processing variable INDEX */
    INDEX_Dims[0]=N;
    capi_INDEX_intent |= F2PY_INTENT_IN;
    const char * capi_errmess = "nnls_f77.nnls_f77.nnlsr: failed to create array from the 7th argument `INDEX`";
    capi_INDEX_as_array = ndarray_from_pyobj(  NPY_INT,1,INDEX_Dims,INDEX_Rank,  capi_INDEX_intent,INDEX_capi,capi_errmess);
    if (capi_INDEX_as_array == NULL) {
        PyObject* capi_err = PyErr_Occurred();
        if (capi_err == NULL) {
            capi_err = nnls_f77_error;
            PyErr_SetString(capi_err, capi_errmess);
        }
    } else {
        INDEX = (int *)(PyArray_DATA(capi_INDEX_as_array));

    /* Processing variable W */
    W_Dims[0]=N;
    capi_W_intent |= F2PY_INTENT_IN;
    const char * capi_errmess = "nnls_f77.nnls_f77.nnlsr: failed to create array from the 5th argument `W`";
    capi_W_as_array = ndarray_from_pyobj(  NPY_DOUBLE,1,W_Dims,W_Rank,  capi_W_intent,W_capi,capi_errmess);
    if (capi_W_as_array == NULL) {
        PyObject* capi_err = PyErr_Occurred();
        if (capi_err == NULL) {
            capi_err = nnls_f77_error;
            PyErr_SetString(capi_err, capi_errmess);
        }
    } else {
        W = (double *)(PyArray_DATA(capi_W_as_array));

    /* Processing variable X */
    X_Dims[0]=N;
    capi_X_intent |= F2PY_INTENT_IN;
    const char * capi_errmess = "nnls_f77.nnls_f77.nnlsr: failed to create array from the 3rd argument `X`";
    capi_X_as_array = ndarray_from_pyobj(  NPY_DOUBLE,1,X_Dims,X_Rank,  capi_X_intent,X_capi,capi_errmess);
    if (capi_X_as_array == NULL) {
        PyObject* capi_err = PyErr_Occurred();
        if (capi_err == NULL) {
            capi_err = nnls_f77_error;
            PyErr_SetString(capi_err, capi_errmess);
        }
    } else {
        X = (double *)(PyArray_DATA(capi_X_as_array));

    /* Processing variable ZZ */
    ZZ_Dims[0]=M;
    capi_ZZ_intent |= F2PY_INTENT_IN;
    const char * capi_errmess = "nnls_f77.nnls_f77.nnlsr: failed to create array from the 6th argument `ZZ`";
    capi_ZZ_as_array = ndarray_from_pyobj(  NPY_DOUBLE,1,ZZ_Dims,ZZ_Rank,  capi_ZZ_intent,ZZ_capi,capi_errmess);
    if (capi_ZZ_as_array == NULL) {
        PyObject* capi_err = PyErr_Occurred();
        if (capi_err == NULL) {
            capi_err = nnls_f77_error;
            PyErr_SetString(capi_err, capi_errmess);
        }
    } else {
        ZZ = (double *)(PyArray_DATA(capi_ZZ_as_array));

/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
                (*f2py_func)(A,&MDA,&M,&N,B,X,&RNORM,W,ZZ,INDEX,&MODE,&NSETP);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
        if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
        CFUNCSMESS("Building return value.\n");
        capi_buildvalue = Py_BuildValue("");
/*closepyobjfrom*/
/*end of closepyobjfrom*/
        } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
    if((PyObject *)capi_ZZ_as_array!=ZZ_capi) {
        Py_XDECREF(capi_ZZ_as_array); }
    }  /* if (capi_ZZ_as_array == NULL) ... else of ZZ */
    /* End of cleaning variable ZZ */
    if((PyObject *)capi_X_as_array!=X_capi) {
        Py_XDECREF(capi_X_as_array); }
    }  /* if (capi_X_as_array == NULL) ... else of X */
    /* End of cleaning variable X */
    if((PyObject *)capi_W_as_array!=W_capi) {
        Py_XDECREF(capi_W_as_array); }
    }  /* if (capi_W_as_array == NULL) ... else of W */
    /* End of cleaning variable W */
    if((PyObject *)capi_INDEX_as_array!=INDEX_capi) {
        Py_XDECREF(capi_INDEX_as_array); }
    }  /* if (capi_INDEX_as_array == NULL) ... else of INDEX */
    /* End of cleaning variable INDEX */
    } /*CHECKSCALAR(shape(A, 1) == N)*/
    } /*if (f2py_success) of N*/
    /* End of cleaning variable N */
    } /*CHECKSCALAR(shape(A, 0) == MDA)*/
    } /*if (f2py_success) of MDA*/
    /* End of cleaning variable MDA */
    } /*CHECKSCALAR(shape(B, 0) == M)*/
    } /*if (f2py_success) of M*/
    /* End of cleaning variable M */
    } /*if (f2py_success) of RNORM*/
    /* End of cleaning variable RNORM */
    if((PyObject *)capi_B_as_array!=B_capi) {
        Py_XDECREF(capi_B_as_array); }
    }  /* if (capi_B_as_array == NULL) ... else of B */
    /* End of cleaning variable B */
    if((PyObject *)capi_A_as_array!=A_capi) {
        Py_XDECREF(capi_A_as_array); }
    }  /* if (capi_A_as_array == NULL) ... else of A */
    /* End of cleaning variable A */
    } /*if (f2py_success) of NSETP*/
    /* End of cleaning variable NSETP */
    } /*if (f2py_success) of MODE*/
    /* End of cleaning variable MODE */
/*end of cleanupfrompyobj*/
    if (capi_buildvalue == NULL) {
/*routdebugfailure*/
    } else {
/*routdebugleave*/
    }
    CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
    return capi_buildvalue;
}
/******************************** end of nnlsr ********************************/
/*eof body*/

/******************* See f2py2e/f90mod_rules.py: buildhooks *******************/
/*need_f90modhooks*/

/************** See f2py2e/rules.py: module_rules['modulebody'] **************/

/******************* See f2py2e/common_rules.py: buildhooks *******************/

/*need_commonhooks*/

/**************************** See f2py2e/rules.py ****************************/

static FortranDataDef f2py_routine_defs[] = {
    {"nnlsr",-1,{{-1}},0,0,(char *)  F_FUNC(nnlsr,nnlsr),  (f2py_init_func)f2py_rout_nnls_f77_nnlsr,doc_f2py_rout_nnls_f77_nnlsr},

/*eof routine_defs*/
    {NULL}
};

static PyMethodDef f2py_module_methods[] = {

    {NULL,NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "nnls_f77",
    NULL,
    -1,
    f2py_module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_nnls_f77(void) {
    int i;
    PyObject *m,*d, *s, *tmp;
    m = nnls_f77_module = PyModule_Create(&moduledef);
    Py_SET_TYPE(&PyFortran_Type, &PyType_Type);
    import_array();
    if (PyErr_Occurred())
        {PyErr_SetString(PyExc_ImportError, "can't initialize module nnls_f77 (failed to import numpy)"); return m;}
    d = PyModule_GetDict(m);
    s = PyUnicode_FromString("2.0.1");
    PyDict_SetItemString(d, "__version__", s);
    Py_DECREF(s);
    s = PyUnicode_FromString(
        "This module 'nnls_f77' is auto-generated with f2py (version:2.0.1).\nFunctions:\n"
"    nnlsr(A,B,X,RNORM,W,ZZ,INDEX,MODE,NSETP,MDA=shape(A, 0),M=shape(B, 0),N=shape(A, 1))\n"
".");
    PyDict_SetItemString(d, "__doc__", s);
    Py_DECREF(s);
    s = PyUnicode_FromString("2.0.1");
    PyDict_SetItemString(d, "__f2py_numpy_version__", s);
    Py_DECREF(s);
    nnls_f77_error = PyErr_NewException ("nnls_f77.error", NULL, NULL);
    /*
     * Store the error object inside the dict, so that it could get deallocated.
     * (in practice, this is a module, so it likely will not and cannot.)
     */
    PyDict_SetItemString(d, "_nnls_f77_error", nnls_f77_error);
    Py_DECREF(nnls_f77_error);
    for(i=0;f2py_routine_defs[i].name!=NULL;i++) {
        tmp = PyFortranObject_NewAsAttr(&f2py_routine_defs[i]);
        PyDict_SetItemString(d, f2py_routine_defs[i].name, tmp);
        Py_DECREF(tmp);
    }

/*eof initf2pywraphooks*/
/*eof initf90modhooks*/

/*eof initcommonhooks*/


#ifdef F2PY_REPORT_ATEXIT
    if (! PyErr_Occurred())
        on_exit(f2py_report_on_exit,(void*)"nnls_f77");
#endif
    return m;
}
#ifdef __cplusplus
}
#endif
