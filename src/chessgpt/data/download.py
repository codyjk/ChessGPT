"""
Lichess PGN database downloader.

Downloads monthly game databases from https://database.lichess.org/standard/
and decompresses them. Files are .pgn.zst (Zstandard compressed PGN).

January 2013 is the smallest file (~121K games, ~40MB compressed) -- ideal for
development. Larger months (2020+) can be 5-20GB compressed.
"""

import urllib.request
from pathlib import Path

import zstandard


def lichess_url(year: int, month: int) -> str:
    """Return the Lichess download URL for a given year/month."""
    zst_filename = f"lichess_db_standard_rated_{year:04d}-{month:02d}.pgn.zst"
    return f"https://database.lichess.org/standard/{zst_filename}"


def zst_filename(year: int, month: int) -> str:
    """Return the .zst filename for a given year/month."""
    return f"lichess_db_standard_rated_{year:04d}-{month:02d}.pgn.zst"


def pgn_filename(year: int, month: int) -> str:
    """Return the .pgn filename for a given year/month."""
    return f"lichess_db_standard_rated_{year:04d}-{month:02d}.pgn"


def download_zst(year: int, month: int, output_dir: str) -> Path:
    """
    Download a Lichess monthly .pgn.zst without decompressing.

    Returns path to the downloaded .zst file. Skips download if file already exists.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    zst_path = output_path / zst_filename(year, month)

    if zst_path.exists():
        print(f"Already exists: {zst_path}")
        return zst_path

    url = lichess_url(year, month)
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, zst_path)
    print(f"Saved to {zst_path}")
    return zst_path


def decompress_zst(zst_path: str | Path) -> Path:
    """
    Decompress a .pgn.zst file to .pgn, then delete the .zst.

    Returns path to the decompressed .pgn file.
    """
    zst_path = Path(zst_path)
    if not zst_path.exists():
        raise FileNotFoundError(f"File not found: {zst_path}")

    # Strip .zst suffix to get .pgn path
    pgn_path = zst_path.with_suffix("")
    if not pgn_path.suffix == ".pgn":
        raise ValueError(f"Expected .pgn.zst file, got: {zst_path}")

    print(f"Decompressing {zst_path}...")
    dctx = zstandard.ZstdDecompressor()
    with open(zst_path, "rb") as ifh, open(pgn_path, "wb") as ofh:
        dctx.copy_stream(ifh, ofh)

    zst_path.unlink()
    print(f"Decompressed to {pgn_path}")
    return pgn_path


def download_pgn(year: int, month: int, output_dir: str) -> Path:
    """
    Download and decompress a Lichess monthly PGN database.

    Downloads from database.lichess.org, decompresses .zst to .pgn, deletes .zst.
    Returns path to the decompressed PGN file.
    """
    output_path = Path(output_dir)
    pgn_path = output_path / pgn_filename(year, month)

    if pgn_path.exists():
        print(f"Already exists: {pgn_path}")
        return pgn_path

    zst_path = download_zst(year, month, output_dir)
    return decompress_zst(zst_path)
