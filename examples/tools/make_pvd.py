"""Group an existing folder of time-named .vti files into a single ParaView dataset.

WHY: the solver names its VTK output ``output_<simulation time>.vti`` (e.g.
``output_352.25121656.vti``). ParaView only auto-groups a file series when the names look
like ``name_<digits>.ext``, so the embedded float time makes every file load as a SEPARATE
dataset. This writes an ``output.pvd`` collection that indexes them all with their REAL
timesteps -- open that one file and you get a single dataset with a working time slider.

The solver now writes output.pvd automatically, but this tool regenerates it for runs that
are already finished (or still in progress -- just re-run it to pick up new files).

Usage:
    python examples/tools/make_pvd.py <result_folder_or_output_folder> [...]
    python examples/tools/make_pvd.py SupersonicCylinderPonD_1
    python examples/tools/make_pvd.py MovingCylinderPonD_1/output --name density
"""

import argparse
import re
import sys
from pathlib import Path

_TIME_RE = re.compile(r"^(?P<stem>.+?)_(?P<time>-?\d+(?:\.\d+)?)\.vti$")


def find_output_dir(path: Path) -> Path:
    """Accept either the run folder or its 'output' subfolder."""
    if path.is_dir() and (path / "output").is_dir() and any((path / "output").glob("*.vti")):
        return path / "output"
    return path


def collect(output_dir: Path, stem_filter=None):
    """Return [(time, filename)] for every time-named .vti, sorted by time."""
    entries = []
    for f in output_dir.glob("*.vti"):
        m = _TIME_RE.match(f.name)
        if not m:
            continue
        if stem_filter and m.group("stem") != stem_filter:
            continue
        entries.append((float(m.group("time")), f.name))
    entries.sort(key=lambda e: e[0])
    return entries


def write_pvd(output_dir: Path, entries, pvd_name="output.pvd") -> Path:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for t, fn in entries:
        lines.append(f'    <DataSet timestep="{t:.8f}" group="" part="0" file="{fn}"/>')
    lines += ["  </Collection>", "</VTKFile>"]
    path = output_dir / pvd_name
    path.write_text("\n".join(lines) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="run folder(s) or their output/ subfolder(s)")
    ap.add_argument("--name", default=None,
                    help="only include files with this stem (default: all, e.g. 'output')")
    ap.add_argument("--pvd", default="output.pvd", help="collection filename to write")
    args = ap.parse_args()

    for raw in args.folders:
        d = find_output_dir(Path(raw))
        if not d.is_dir():
            print(f"!! not a folder: {d}", file=sys.stderr)
            continue
        entries = collect(d, args.name)
        if not entries:
            print(f"!! no time-named .vti files in {d}", file=sys.stderr)
            continue
        path = write_pvd(d, entries, args.pvd)
        print(f"{path}  <- {len(entries)} timesteps, t = {entries[0][0]:.4f} .. {entries[-1][0]:.4f}")
        print("   open THIS file in ParaView (not the .vti files) for one grouped dataset.")


if __name__ == "__main__":
    main()
