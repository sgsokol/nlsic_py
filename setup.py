#!/usr/bin/env python3

# run with e.g.
# python setup.py build_src build_ext -b ./dist
import setuptools
from numpy.distutils.core import setup as nps

with open("README.rst", "r", encoding="utf-8") as f:
    long_description = f.read()
with open("version.txt", "r") as f:
    version = f.read().rstrip()
def get_library_dirs():
    from numpy import __config__
    attrs = ['blas_mkl_info', 'blas_opt_info', 'lapack_mkl_info',
             'lapack_opt_info', 'openblas_lapack_info']
    libdirs = set()
    for attr in attrs:
        obj = getattr(__config__, attr, {}).get('library_dirs')
        if obj is not None:
            libdirs.update(obj)
    return list(libdirs)
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    res={}
    config = Configuration('', parent_package, top_path)
    config.add_extension('fpylapack',
                         sources=['fpylapack/fpylapack.pyf'],
                         libraries=['lapack'],
                         library_dirs=get_library_dirs()
                         )
    res=config.todict()
    #print("res fpy=", res)
    #config = Configuration('', parent_package, top_path)
    config.add_extension('nnls_f77',
                         sources=['nnls_f77/nnls_f77.pyf', 'nnls_f77/nnlsr.f', 'nnls_f77/nnls.f'])
    res.update(config.todict())
    #print("res f77=", res)
    #config = Configuration('', parent_package, top_path)
    config.add_extension('nnls_f90',
                         sources=['nnls_f90/nnls.f90'])
    res.update(config.todict())
    del(res["name"])
    #print("res f90=", res)
    return res

if __name__ == '__main__':
    d = configuration(top_path='', parent_package='nlsic')
    #print("d=", d)
#    setuptools.setup(
    nps(
        name = 'nlsic',
        version = version,
        setup_requires = ['numpy'],
        install_requires = ['numpy'],
        url = 'https://github.com/sgsokol/nlsic_py/',
        description = 'Non-linear Least Squares with Inequality Constraints',
        keywords = ['least squares'],
        license = 'GNU General Public License v2 or later (GPLv2+)',
        long_description = long_description,
        long_description_content_type = 'text/x-rst',
        author = 'Serguei Sokol',
        author_email = 'sokol@insa-toulouse.fr',
        packages = ['nlsic'],
        py_modules = ['nlsic.nnls', 'nlsic.lapack_ssg'],
        **d,
    )
