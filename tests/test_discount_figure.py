import base64
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import zlib
from examples.render_discount import render
from ergodlab.discount_demo import discount_demo


class DiscountFigure(unittest.TestCase):
    def test_regeneration_full_metadata(self):
        d=discount_demo();s=render(d);self.assertEqual(s,Path('assets/discount-audit.svg').read_text());m=ET.fromstring(s).find('{http://www.w3.org/2000/svg}metadata');self.assertEqual(json.loads(zlib.decompress(base64.b64decode(m.text))),d)
    def test_every_table_cell_and_geometry(self):
        d=discount_demo();root=ET.fromstring(render(d));cells=[x for x in root.findall('{http://www.w3.org/2000/svg}text') if 'data-row' in x.attrib];self.assertEqual(len(cells),20)
        for cell in cells:
            i=int(cell.attrib['data-row']);j=int(cell.attrib['data-col']);row=d['discounts'][i]
            def v(x):return '['+', '.join(str(F(*a)) for a in x)+']'
            expected=[str(F(*row['beta']))+(' (limit)' if row['is_average_reward_limit'] else ''),v(row['stationary_gradient_proxy']),v(row['fixed_start_objective_gradient']),str(F(*row['proxy_curl'][0][1]))][j]
            self.assertEqual(cell.text,expected);self.assertEqual(int(cell.attrib['x']),[42,170,465,820][j]);self.assertEqual(int(cell.attrib['y']),269+56*i)
    def test_undisplayed_mutation_rejected(self):
        d=deepcopy(discount_demo());d['discounts'][2]['centered_resolvent_derivatives'][0][0][0]+=1
        with self.assertRaises(ValueError):render(d)
