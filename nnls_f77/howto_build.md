```
python3.12 -m numpy.f2py -h nnls_f77.pyf nnlsr.f
python3.12 -m numpy.f2py nnls_f77.pyf nnlsr.f -m nnls_f77
```

then use `meson` on nnls_f77module.c, nnlsr.f and nnls.f
