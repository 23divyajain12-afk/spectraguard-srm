import subprocess
import sys
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]

class WebUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = subprocess.Popen([sys.executable, 'src/app/web_ui.py', '--port', '8765'], cwd=ROOT)
        time.sleep(.25)
    @classmethod
    def tearDownClass(cls): cls.server.terminate(); cls.server.wait(timeout=3)
    def test_independent_pages(self):
        for route,marker in {'/':b'Mission overview','/scene':b'Explore Scene','/process':b'Processing overview','/results':b'Artifact comparison'}.items():
            with urlopen('http://127.0.0.1:8765'+route) as response:
                self.assertEqual(response.status,200);self.assertIn(marker,response.read())
    def test_available_assets_and_unknown_routes(self):
        for route in ('/scene.png','/result/bicubic.png'):
            with urlopen('http://127.0.0.1:8765'+route) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers.get_content_type(), 'image/png')
        with self.assertRaises(HTTPError) as caught:
            urlopen('http://127.0.0.1:8765/unknown')
        self.assertEqual(caught.exception.code, 404)
    def test_navigation_and_aoi_restriction(self):
        with urlopen('http://127.0.0.1:8765/scene') as response: page=response.read().lower()
        self.assertIn(b'href="/"',page);self.assertNotIn(b'draw aoi',page);self.assertNotIn(b'edit aoi',page)
