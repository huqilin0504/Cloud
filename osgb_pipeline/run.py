"""Root-driven OSGB -> geometry OBJ and sampled XYZ LAS (no texture/color)."""
import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import laspy
import numpy as np
from pyproj import CRS

BASE = Path(__file__).resolve().parent
DEFAULT = BASE.parent / '科研 现场资料/导流洞进口危岩体 航拍三维/危岩体/Block.osgb'
DEFAULT_DENSITY = 600.0

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--density', type=float, default=DEFAULT_DENSITY)
    p.add_argument('--max-geometries', type=int, default=0)
    p.add_argument('--origin', choices=['keep', 'add'], required=True,
                   help='keep preserves model coordinates; add applies metadata SRSOrigin once')
    args = p.parse_args()
    if not np.isfinite(args.density) or args.density <= 0 or args.max_geometries < 0:
        p.error('Invalid density or geometry limit')
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists():
        p.error('Output must be a new directory; existing outputs are never reused')
    meta = ET.parse(root.parent / 'metadata.xml').getroot()
    origin = np.array([float(v) for v in meta.findtext('.//SRSOrigin').split(',')])
    if origin.shape != (3,) or not np.isfinite(origin).all():
        p.error('Invalid SRSOrigin')
    srs = meta.findtext('.//SRS', '').strip()
    shift = origin if args.origin == 'add' else np.zeros(3)
    subprocess.run([str(BASE/'build/extract_osgb'), str(root), str(output),
                    str(args.density), str(args.max_geometries), *map(str, shift), '42'], check=True)
    raw = output/'points.bin'
    if raw.stat().st_size % 24:
        raise RuntimeError('Truncated XYZ binary')
    xyz = np.memmap(raw, dtype='<f8', mode='r').reshape(-1, 3)
    header = laspy.LasHeader(point_format=0, version='1.4')
    header.scales = [0.001]*3
    header.offsets = xyz.min(axis=0)
    if ((xyz.max(axis=0)-header.offsets)/header.scales > 2147483647).any():
        raise RuntimeError('Coordinate extent exceeds chosen LAS scale')
    if srs and srs.upper() != 'LOCAL':
        header.add_crs(CRS.from_user_input(srs))
    target = output/'cloud.las'
    partial = output/'cloud.las.part'
    with laspy.open(partial, mode='w', header=header) as writer:
        for start in range(0, len(xyz), 500000):
            chunk = xyz[start:start+500000]
            points = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=header)
            points.x, points.y, points.z = chunk.T
            writer.write_points(points)
    # Independent library readback checks every coordinate, not only header size.
    count, error = 0, 0.0
    with laspy.open(partial) as reader:
        for points in reader.chunk_iterator(500000):
            restored = np.column_stack((points.x, points.y, points.z))
            error = max(error, float(np.max(np.abs(restored-xyz[count:count+len(points)]))))
            count += len(points)
        if count != len(xyz) or error > 0.000501:
            raise RuntimeError('LAS readback failed')
    partial.rename(target)
    report = dict(root=str(root), origin_mode=args.origin, metadata_origin=origin.tolist(),
                  srs=srs, assigned_crs=(srs if srs.upper() != 'LOCAL' else None),
                  points=count, minimum=xyz.min(axis=0).tolist(), maximum=xyz.max(axis=0).tolist(),
                  max_roundtrip_error=error, density=args.density,
                  geometry_limit=args.max_geometries, texture=False, rgb=False)
    (output/'qc.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
