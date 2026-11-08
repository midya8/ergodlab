import json
import os
import subprocess
import sys
import tempfile
import unittest
from ergodlab.demo import demo,occupancy_model


class CLI(unittest.TestCase):
    def run_cli(self,*args):return subprocess.run([sys.executable,'-m','ergodlab',*args],capture_output=True,text=True)
    def test_demo(self):
        p=self.run_cli('demo');self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout),demo())
    def test_caller_model(self):
        p=self.run_cli('audit','examples/occupancy.json','--block-limit','3');self.assertEqual(p.returncode,0,p.stderr);a=json.loads(p.stdout);self.assertEqual(len(a['variance']['stationary_blocks']),3);self.assertEqual(a['gradient']['total'],[[1,18]])
    def test_strict_input(self):
        payload={'version':1,'model':occupancy_model()}
        bad=[{},dict(payload,version=True),dict(payload,version=2),dict(payload,extra=1),dict(payload,model=None)]
        with tempfile.TemporaryDirectory() as d:
            path=os.path.join(d,'input.json')
            for item in bad:
                with open(path,'w') as f:json.dump(item,f)
                p=self.run_cli('audit',path);self.assertEqual(p.returncode,2);self.assertFalse(p.stdout);self.assertNotIn('Traceback',p.stderr)
            for text in ['{','NaN','{"version":1,"model":NaN}','{"version":1,"version":1}', ' '*65537, '['*1100]:
                with open(path,'w') as f:f.write(text)
                p=self.run_cli('audit',path);self.assertEqual(p.returncode,2);self.assertFalse(p.stdout)
    def test_invalid_path_and_options(self):
        for args in [('audit','/definitely-not-a-file'),('audit','examples/occupancy.json','--block-limit','0')]:
            p=self.run_cli(*args);self.assertEqual(p.returncode,2);self.assertFalse(p.stdout)
