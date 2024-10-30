project(
    'nlsic',
    ['c', 'fortran'],
    version: run_command('cat', 'version.txt', check: true).stdout().strip(),
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()
ladep = dependency('lapack')
npdep = dependency('numpy')
incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy)
np_dep = declare_dependency(include_directories: inc_np)

incdir_f2py = incdir_numpy / '..' / '..' / 'f2py' / 'src'
inc_f2py = include_directories(incdir_f2py)
fortranobject_c = incdir_f2py / 'fortranobject.c'

inc_np = include_directories(incdir_numpy, incdir_f2py)
#add_project_arguments('-I', incdir_numpy, language : 'c')

py.extension_module(
    'nnls_f77',
    ['nnls_f77/nnlsr.f', 'nnls_f77/nnls.f'],
    include_directories: [
                     inc_np,
                     ],
    install: true,
)
py.extension_module(
    'nnls_f90',
    'nnls_f90/nnls.f90',
    include_directories: [
                     inc_np,
                     ],
    install: true,
)
py.extension_module(
    'fpylapack',
    'fpylapackmodule.c',
    include_directories: [
                     inc_np,
                     ],
    dependencies: [py_dep, ladep, np_dep],
    install: true,
)
