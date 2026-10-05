from pathlib import Path
import subprocess
import shutil

ROOT = Path(__file__).resolve().parents[1]

PPDATA = (
    ROOT
    / "cocopp_postprocessing"
    / "ppdata"
    / "DMGSO_NELDE_POWEL_COBYL_COMPA_092916h4343"
)

OUT = ROOT / "figure_export" / "png_300dpi"
OUT.mkdir(parents=True, exist_ok=True)

dimensions = [2, 5, 10, 20, 40]

pdftocairo = shutil.which("pdftocairo")
if pdftocairo is None:
    raise RuntimeError("pdftocairo was not found on PATH.")

print("=" * 72)
print("COCO/BBOB ECDF -> PNG 300 dpi")
print("Source: native COCOpp PDF output")
print(f"Converter: {pdftocairo}")
print("=" * 72)

for d in dimensions:
    src = PPDATA / f"pprldmany_{d:02d}D_noiselessall.pdf"
    dst = OUT / f"COCO_BBOB_ECDF_{d:02d}D_300dpi.png"

    if not src.exists():
        raise FileNotFoundError(src)

    # pdftocairo appends ".png" automatically when the output
    # prefix is supplied, so use the destination without suffix.
    prefix = dst.with_suffix("")

    cmd = [
        pdftocairo,
        "-png",
        "-singlefile",
        "-r", "300",
        str(src),
        str(prefix),
    ]

    subprocess.run(cmd, check=True)

    if not dst.exists():
        raise RuntimeError(f"Expected output was not created: {dst}")

    print(
        f"[OK] D={d:2d} | "
        f"{dst.name} | "
        f"{dst.stat().st_size:,} bytes"
    )

print("=" * 72)
print("Output:", OUT.resolve())
print("=" * 72)
