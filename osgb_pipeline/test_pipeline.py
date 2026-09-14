"""Regression: refinement excludes coarse mesh and retains parent transforms."""
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np

base=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='osgb-regression-') as directory:
    fixture=Path(directory)/'input'
    output=Path(directory)/'output'
    subprocess.run([str(base/'build/make_fixture'),str(fixture)],check=True)
    subprocess.run([str(base/'build/extract_osgb'),str(fixture/'Block.osgb'),str(output),
                    '100','0','100','200','300','42'],check=True)
    stats=json.loads((output/'stats.json').read_text())
    assert stats['triangles']==1 and stats['transforms']==2,stats
    vertices=np.array([[float(x) for x in line.split()[1:]]
                       for line in (output/'model.obj').read_text().splitlines() if line.startswith('v ')])
    np.testing.assert_allclose(vertices,[[110,220,330],[112,220,330],[110,222,330]])
    points=np.fromfile(output/'points.bin',dtype='<f8').reshape(-1,3)
    assert len(points)==200
    assert np.all(points[:,:2]>=np.array([110,220]))
    assert np.all((points[:,0]-110)+(points[:,1]-220)<=2+1e-10)
    print('PASS: PagedLOD refinement, inherited scale/translation, origin, area sampling')
