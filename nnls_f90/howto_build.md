```
python3 -m numpy.f2py nnls.f90 -m nnls_f90
```

then use `meson` on nnls_f90module.c, nnls.f90 and nnls_f90-f2pywrappers2.f90
