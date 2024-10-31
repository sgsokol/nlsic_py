```
python3 -m numpy.f2py nnls_f77.pyf nnlsr.f -m nnls_f7
```

then use `meson` on nnls_f77module.c, nnlsr.f and nnls.f
