"""Run the complete assignment pipeline in the required order."""
import subprocess,sys
steps=[[sys.executable,'src/prepare.py'],[sys.executable,'src/grad_check.py'],[sys.executable,'src/train.py'],[sys.executable,'src/test_model.py']]
for cmd in steps:
    print('\n>>>',' '.join(cmd),flush=True)
    subprocess.run(cmd,check=True)
