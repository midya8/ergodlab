import base64
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import zlib
from examples.render_audit import render
from ergodlab.demo import demo


class Figure(unittest.TestCase):
    def test_complete_metadata_and_regeneration(self):
        d=demo();value=render(d);self.assertEqual(value,Path('assets/stationary-audit.svg').read_text());root=ET.fromstring(value);m=root.find('{http://www.w3.org/2000/svg}metadata');self.assertEqual(json.loads(zlib.decompress(base64.b64decode(m.text))),d)
    def test_every_point_and_limit_geometry(self):
        d=demo();root=ET.fromstring(render(d));dots=root.findall('{http://www.w3.org/2000/svg}circle');self.assertEqual(len(dots),32)
        for dot in dots:
            n=int(dot.attrib['data-length']);b=d[dot.attrib['data-fixture']]['variance']['stationary_blocks'][n-1];self.assertEqual(float(dot.attrib['cx']),75+39*(n-1));self.assertAlmostEqual(float(dot.attrib['cy']),495-160*float(F(*b['variance_per_step'])),places=7)
        paths=[x for x in root.findall('{http://www.w3.org/2000/svg}path') if 'data-limit' in x.attrib];self.assertEqual(len(paths),2)
        for p in paths:
            y=495-160*float(F(*d[p.attrib['data-limit']]['variance']['asymptotic_per_step']));self.assertEqual(p.attrib['d'],f'M 75 {y:.8f} H 680')
    def test_changed_undisplayed_data_rejected(self):
        d=deepcopy(demo());d['occupancy']['tree_reference']['root_tree_weights'][0][0]+=1
        with self.assertRaises(ValueError):render(d)
