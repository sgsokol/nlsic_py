NLSIC stands for "Non-linear Least Squares with Inequality Constraints". This algorithm was first published in Sokol et al. (2012) <doi:10.1093/bioinformatics/btr716>. His advantage is in combination of the following features:

    * Jacobi matrix *J* is not squared, i.e. *J^tJ* is not calculated, thus preserving the solution from condition number degradation. Instead, QR decomposition is used;
    * taking into account inequality constraints is based on a well known NNLS algorithm by Lawson and Hanson;
    * convergence in non-linear iterations is globalized by back-tracking method. I.e. with enough iterations, the process will reach a convergence point but no guaranty is given that it will be a global minimum;
    * if *J* is rank deficient, a least norm (also known as "minimum norm") increment is used. To find such increment exactly (not just a regularized one) in presence of inequality constraints is not a trivial task. We use our own algorithm implemented in ``lsi_ln(...)`` function;
    * if an initial approximation for a solution vector is outside of feasibility domain (FD) defined by inequality constraints then it is first projected on the FD (cf. ``ldp(...)``) and the non-linear iterations start afterward.
    
This python implementation is an adaptation of the R one which is distributed with `influx_si <https://metasys.insa-toulouse.fr/software/influx/>`_ software. The original R version is in advance on this one. So users desiring to catch with the latest developments are invited to switch to the R version.

The main function to call is ``nlsic(...)``.
