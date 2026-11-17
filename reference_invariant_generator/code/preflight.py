from study import *
import subprocess
assert not (OUT/'models.json').exists(),'Preflight may not overwrite existing protocol.'
r=subprocess.run([sys.executable,'-m','pytest','tests','-q'],cwd=ROOT);assert r.returncode==0
protocol();dump('smoke_pass.json',dict(tests_passed=True,before_training=True,exact_G4_and_gradient_verified=True))
