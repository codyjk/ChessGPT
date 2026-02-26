"""
Lichess PGN database downloader.

Downloads monthly game databases from https://database.lichess.org/standard/
and decompresses them. Files are .pgn.zst (Zstandard compressed PGN).

January 2013 is the smallest file (~121K games, ~40MB compressed) — ideal for
development. Larger months (2020+) can be 5-20GB compressed.
"""

import urllib.request
from pathlib import Path

import zstandard


def download_pgn(year: int, month: int, output_dir: str) -> Path:
    """
    Download and decompress a Lichess monthly PGN database.

    Downloads from database.lichess.org, streams through zstandard decompressor,
    writes plain PGN to output_dir.

    Returns path to the decompressed PGN file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"lichess_db_standard_rated_{year:04d}-{month:02d}.pgn"
    zst_filename = f"{filename}.zst"
    url = f"https://database.lichess.org/standard/{zst_filename}"
    pgn_path = output_path / filename

    if pgn_path.exists():
        print(f"Already exists: {pgn_path}")
        return pgn_path

    print(f"Downloading {url}...")
    zst_path = output_path / zst_filename
    urllib.request.urlretrieve(url, zst_path)

    print(f"Decompressing {zst_path}...")
    dctx = zstandard.ZstdDecompressor()
    with open(zst_path, "rb") as ifh, open(pgn_path, "wb") as ofh:
        dctx.copy_stream(ifh, ofh)

    zst_path.unlink()
    print(f"Saved to {pgn_path}")
    return pgn_path
