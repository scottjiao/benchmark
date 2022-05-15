
import jstyleson as json
import os

import subprocess
f=open("./.vscode/launch.json",'r')
js=json.load(f)
f.close()
print(js)
args=js['configurations'][0]['args']
print(args)
s="python run_analysis.py"
for i in args:
    s+=" "+i
subprocess.run(s,shell=True)

