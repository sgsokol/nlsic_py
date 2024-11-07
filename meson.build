project(
    'nlsic',
    ['c', 'fortran'],
    version: configure_file(input: 'version.txt', output: '-', copy: true),
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

ladep = dependency('lapack')

incdir_py = run_command(py,
  ['-c', 'import sysconfig; print(sysconfig.get_path("include"))'],
  check : true
).stdout().strip()

incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()
fortranobject_c = incdir_f2py / 'fortranobject.c'

inc_np = include_directories(incdir_numpy)
inc_py = include_directories(incdir_py)
np_dep = declare_dependency(include_directories: inc_np)

incdir_f2py = incdir_numpy / '..' / '..' / 'f2py' / 'src'
inc_f2py = include_directories(incdir_f2py)
fortranobject_c = incdir_f2py / 'fortranobject.c'

inc_np = include_directories(incdir_numpy, incdir_f2py)
#add_project_arguments('-I', incdir_numpy, language : 'c')

fobj_lib=static_library('fobj',
    fortranobject_c,
    include_directories: [
                 inc_py,
                 inc_np,
                 ],
    dependencies: [py_dep, np_dep],
)

py.extension_module(
    'nnls_f77',
    ['nnls_f77/nnls_f77module.c', 'nnls_f77/nnlsr.f', 'nnls_f77/nnls.f'],
    include_directories: [
                     inc_np,
                     ],
    link_with: [fobj_lib],
    dependencies: [py_dep, ladep, np_dep],
    install: true,
    subdir: 'nlsic',
)
py.extension_module(
    'nnls_f90',
    ['nnls_f90/nnls_f90module.c', 'nnls_f90/nnls.f90', 'nnls_f90/nnls_f90-f2pywrappers2.f90'],
    include_directories: [
                     inc_np,
                     ],
    link_with: [fobj_lib],
    dependencies: [py_dep, ladep, np_dep],
    install: true,
    subdir: 'nlsic',
)
py.extension_module(
    'fpylapack',
    ['fpylapackmodule.c'],
    include_directories: [
                     inc_np,
                     ],
    link_with: [fobj_lib],
    dependencies: [py_dep, ladep, np_dep],
    install: true,
    subdir: 'nlsic',
)
py.install_sources(
    ['nlsic/__init__.py', 'nlsic/nlsic.py', 'nlsic/lapack_ssg.py', 'nlsic/nnls.py', 'version.txt'],
    pure: false,
    subdir: 'nlsic',
)
