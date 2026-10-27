from pathlib import Path
import sys, os, json, subprocess

ENV = Path('/content/paper3_ctu_warning_pilot_env')
PYTHON = ENV / 'bin' / 'python'
VERSIONS = {'numpy':'2.3.5', 'scipy':'1.17.0', 'scikit-learn':'1.8.0', 'joblib':'1.5.3', 'threadpoolctl':'3.6.0'}

# Colab may not provide ensurepip. Also repairs an incomplete environment
# left by the earlier setup attempt, without clearing installed packages.
created = subprocess.run(
    [sys.executable, '-m', 'venv', '--without-pip', str(ENV)],
    text=True, capture_output=True
)
if created.returncode != 0:
    print(created.stdout)
    print(created.stderr)
    raise RuntimeError('Environment creation failed. Send the full message printed above.')

probe = "import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in " + repr(list(VERSIONS)) + "}))"
check = subprocess.run([str(PYTHON), '-c', probe], text=True, capture_output=True)
ready = check.returncode == 0 and json.loads(check.stdout) == VERSIONS
if not ready:
    # Run Colab's existing pip against the isolated interpreter; no ensurepip needed.
    command = [sys.executable, '-m', 'pip', '--python', str(PYTHON),
               'install', '--disable-pip-version-check', '--only-binary=:all:']
    command += [n+'=='+v for n,v in VERSIONS.items()]
    subprocess.run(command, check=True)
check = subprocess.run([str(PYTHON), '-c', probe], text=True, capture_output=True, check=True)
if json.loads(check.stdout) != VERSIONS:
    raise RuntimeError('Isolated dependency versions do not match.')
RUN_ENV = dict(os.environ)
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    RUN_ENV[key] = '1'
print('Isolated dependencies ready:', check.stdout.strip())
