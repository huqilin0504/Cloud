"""Check exported OBJ structure and root-driven source coverage."""
import csv
import json
from pathlib import Path
import sys
import numpy as np

folder=Path(sys.argv[1])
stats=json.loads((folder/'stats.json').read_text())
qc=json.loads((folder/'qc.json').read_text())
vertices=faces=groups=0
minimum=np.full(3,np.inf);maximum=-minimum
with (folder/'model.obj').open() as stream:
    for line in stream:
        if line.startswith('v '):
            xyz=np.array(list(map(float,line.split()[1:])))
            assert xyz.shape==(3,) and np.isfinite(xyz).all()
            minimum=np.minimum(minimum,xyz);maximum=np.maximum(maximum,xyz)
            vertices+=1
        elif line.startswith('f '):
            indices=list(map(int,line.split()[1:]))
            assert len(indices)==3 and min(indices)>=1 and max(indices)<=vertices
            faces+=1
        elif line.startswith('o '):groups+=1
with (folder/'geometry.tsv').open() as stream:
    rows=list(csv.DictReader(stream,delimiter='\t'))
sources={r['source'] for r in rows}
assert vertices==sum(int(r['vertices']) for r in rows)
assert faces==stats['triangles']==sum(int(r['triangles']) for r in rows)
assert groups==stats['geometries']==len(rows)
assert np.all(np.array(qc['minimum'])>=minimum-.001)
assert np.all(np.array(qc['maximum'])<=maximum+.001)
result=dict(unique_leaf_files=len(sources),blocks=sorted({Path(s).parent.name for s in sources}),
            obj_vertices=vertices,obj_triangles=faces,obj_groups=groups,
            obj_minimum=minimum.tolist(),obj_maximum=maximum.tolist(),
            las_within_obj_bounds=True,face_indices_valid=True)
(folder/'mesh_qc.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
