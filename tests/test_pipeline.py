"""Tests for the AWS Lambda data pipeline.

Tests are split into categories:
- download_zst / decompress_zst (library functions)
- .zst streaming in PGN parsing
- Lambda handlers (mocked S3)
- CLI --cloud flag validation
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import zstandard

from chessgpt.data.download import (
    decompress_zst,
    download_zst,
    pgn_filename,
    zst_filename,
)
from chessgpt.pgn.utils import _parse_pgn_lines, process_raw_games_from_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_PGN = textwrap.dedent("""\
    [Event "Rated Blitz game"]
    [WhiteElo "2000"]
    [BlackElo "1900"]

    1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0
""")


def _make_zst(pgn_content: str, path: Path) -> Path:
    """Compress PGN content to a .zst file."""
    cctx = zstandard.ZstdCompressor()
    compressed = cctx.compress(pgn_content.encode("utf-8"))
    path.write_bytes(compressed)
    return path


# ---------------------------------------------------------------------------
# download_zst / decompress_zst
# ---------------------------------------------------------------------------


class TestDownloadZst:
    def test_skips_existing(self, tmp_path: Path) -> None:
        """download_zst returns immediately if .zst already exists."""
        zst_path = tmp_path / zst_filename(2013, 1)
        zst_path.write_text("fake")

        result = download_zst(2013, 1, str(tmp_path))
        assert result == zst_path
        assert result.read_text() == "fake"  # not overwritten

    @patch("chessgpt.data.download.urllib.request.urlretrieve")
    def test_downloads_new(self, mock_retrieve: MagicMock, tmp_path: Path) -> None:
        """download_zst downloads .zst and returns path."""

        def fake_download(url: str, path: str) -> None:
            Path(path).write_text("zst-data")

        mock_retrieve.side_effect = fake_download

        result = download_zst(2013, 1, str(tmp_path))
        assert result.exists()
        assert result.suffix == ".zst"
        assert result.read_text() == "zst-data"
        mock_retrieve.assert_called_once()


class TestDecompressZst:
    def test_decompresses_and_deletes(self, tmp_path: Path) -> None:
        """decompress_zst creates .pgn, deletes .zst, content matches."""
        zst_path = tmp_path / "test.pgn.zst"
        _make_zst("hello pgn\n", zst_path)

        pgn_path = decompress_zst(zst_path)
        assert pgn_path.exists()
        assert pgn_path.suffix == ".pgn"
        assert pgn_path.read_text() == "hello pgn\n"
        assert not zst_path.exists()

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            decompress_zst(tmp_path / "nonexistent.pgn.zst")

    def test_bad_extension_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "test.txt.zst"
        bad.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Expected .pgn.zst"):
            decompress_zst(bad)


class TestFilenameHelpers:
    def test_zst_filename(self) -> None:
        assert zst_filename(2017, 3) == "lichess_db_standard_rated_2017-03.pgn.zst"

    def test_pgn_filename(self) -> None:
        assert pgn_filename(2017, 12) == "lichess_db_standard_rated_2017-12.pgn"


# ---------------------------------------------------------------------------
# .zst streaming
# ---------------------------------------------------------------------------


class TestParsePgnLines:
    def test_parses_lines(self) -> None:
        """_parse_pgn_lines works on any iterable of strings."""
        lines = SAMPLE_PGN.splitlines(keepends=True)
        games = list(_parse_pgn_lines(lines))
        assert len(games) == 1
        assert "WhiteElo" in games[0].metadata[1]
        assert "1-0" in games[0].moves

    def test_empty_input(self) -> None:
        games = list(_parse_pgn_lines([]))
        assert games == []

    def test_multiple_games(self) -> None:
        two_games = SAMPLE_PGN + "\n" + SAMPLE_PGN
        lines = two_games.splitlines(keepends=True)
        games = list(_parse_pgn_lines(lines))
        assert len(games) == 2


class TestZstStreaming:
    def test_process_raw_games_from_zst(self, tmp_path: Path) -> None:
        """process_raw_games_from_file handles .pgn.zst files."""
        zst_path = tmp_path / "test.pgn.zst"
        _make_zst(SAMPLE_PGN, zst_path)

        games = list(process_raw_games_from_file(str(zst_path)))
        assert len(games) == 1
        assert "1-0" in games[0].moves


# ---------------------------------------------------------------------------
# Prepare pipeline with .zst input
# ---------------------------------------------------------------------------


class TestPrepareWithZst:
    def test_prepare_from_zst(self, tmp_path: Path) -> None:
        """prepare_training_data works end-to-end with .zst input."""
        from chessgpt.data.prepare import prepare_training_data

        zst_path = tmp_path / "test.pgn.zst"
        _make_zst(SAMPLE_PGN, zst_path)
        output_csv = tmp_path / "output.csv"

        counts = prepare_training_data(
            input_pgn=str(zst_path),
            output_csv=str(output_csv),
            min_elo=1000,
            min_moves=2,
        )
        assert output_csv.exists()
        assert counts["total"] == 1
        assert counts["written"] == 1


# ---------------------------------------------------------------------------
# Lambda handlers (mocked S3)
# ---------------------------------------------------------------------------


class TestDownloadHandler:
    @patch("chessgpt.lambdas.download_handler.boto3")
    @patch("chessgpt.lambdas.download_handler.download_zst")
    def test_handler(self, mock_download: MagicMock, mock_boto3: MagicMock) -> None:
        from chessgpt.lambdas.download_handler import handler

        mock_download.return_value = Path("/tmp/lichess_db_standard_rated_2013-01.pgn.zst")
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        result = handler({"year": 2013, "month": 1, "bucket": "test-bucket"}, None)

        mock_download.assert_called_once_with(2013, 1, "/tmp")
        mock_s3.upload_file.assert_called_once()
        assert result["status"] == "ok"
        assert "raw/" in result["s3_key"]


class TestPrepareHandler:
    @patch("chessgpt.lambdas.prepare_handler.boto3")
    @patch("chessgpt.lambdas.prepare_handler.prepare_training_data")
    @patch("chessgpt.lambdas.prepare_handler.print_stats")
    def test_handler(
        self, mock_stats: MagicMock, mock_prepare: MagicMock, mock_boto3: MagicMock
    ) -> None:
        from chessgpt.lambdas.prepare_handler import handler

        mock_prepare.return_value = {"total": 100, "written": 80}
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        # Create the CSV that prepare would create so upload_file doesn't fail
        result = handler(
            {"year": 2013, "month": 1, "bucket": "test-bucket"},
            None,
        )

        mock_s3.download_file.assert_called_once()
        mock_prepare.assert_called_once()
        mock_s3.upload_file.assert_called_once()
        assert result["status"] == "ok"
        assert "prepared/" in result["s3_key"]


# ---------------------------------------------------------------------------
# CLI flag validation
# ---------------------------------------------------------------------------


class TestCLICloudFlags:
    def test_download_cloud_requires_bucket(self) -> None:
        """--cloud without --bucket should fail."""
        import sys

        from chessgpt.cli.download import main

        argv = ["chessgpt-download", "--year", "2013", "--month", "1", "--cloud"]
        with patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit):
                main()

    @patch("chessgpt.cli.download._invoke_lambda")
    def test_download_cloud_invokes_lambda(self, mock_invoke: MagicMock) -> None:
        """--cloud with --bucket invokes Lambda."""
        import sys

        from chessgpt.cli.download import main

        argv = [
            "chessgpt-download",
            "--year",
            "2013",
            "--month",
            "1",
            "--cloud",
            "--bucket",
            "my-bucket",
        ]
        with patch.object(sys, "argv", argv):
            main()

        mock_invoke.assert_called_once_with(2013, 1, "my-bucket", "chessgpt-download")

    def test_prepare_cloud_requires_bucket(self) -> None:
        """--cloud without --bucket should fail."""
        import sys

        from chessgpt.cli.prepare import main

        with patch.object(
            sys,
            "argv",
            ["chessgpt-prepare", "--cloud", "--year", "2013", "--month", "1"],
        ):
            with pytest.raises(SystemExit):
                main()

    def test_prepare_merge_requires_year(self) -> None:
        """--merge-from-s3 without --year should fail."""
        import sys

        from chessgpt.cli.prepare import main

        with patch.object(
            sys,
            "argv",
            ["chessgpt-prepare", "--merge-from-s3", "--bucket", "b", "--output-csv", "out.csv"],
        ):
            with pytest.raises(SystemExit):
                main()


# ---------------------------------------------------------------------------
# _merge_from_s3
# ---------------------------------------------------------------------------


class TestMergeFromS3:
    def test_merge_downloads_and_splits(self, tmp_path: Path) -> None:
        """_merge_from_s3 merges CSVs, shuffles, and splits train/val."""
        import csv

        from chessgpt.cli.prepare import _merge_from_s3

        # Build fake CSV content for two months
        csv_contents = {}
        for month in [1, 2]:
            rows = [
                ["moves", "outcome", "checkmate_move_idx", "ply_count"],
                [f"e4 e5 Nf3 Nc6 Bb5 a6 month{month}", "1-0", "-1", "6"],
                [f"d4 d5 c4 e6 Nc3 Nf6 month{month}", "0-1", "-1", "6"],
            ]
            key = f"prepared/lichess_2013-{month:02d}.csv"
            buf = ""
            for row in rows:
                buf += ",".join(row) + "\n"
            csv_contents[key] = buf

        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {
            "Contents": [{"Key": k} for k in sorted(csv_contents.keys())]
        }

        def fake_download(bucket, key, local_path):
            Path(local_path).write_text(csv_contents[key])

        mock_s3.download_file.side_effect = fake_download

        output_csv = str(tmp_path / "train_large.csv")

        with patch("boto3.client", return_value=mock_s3):
            _merge_from_s3(2013, "test-bucket", output_csv, None, 0.5)

        # Check train file
        train_path = tmp_path / "train_large.csv"
        assert train_path.exists()
        with open(train_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            train_rows = list(reader)
        assert header == ["moves", "outcome", "checkmate_move_idx", "ply_count"]

        # Check val file -- stem replacement: train_large -> val_large
        val_path = tmp_path / "val_large.csv"
        assert val_path.exists()
        with open(val_path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            val_rows = list(reader)

        # Total should be 4 rows (2 per month)
        assert len(train_rows) + len(val_rows) == 4

    def test_val_stem_no_train_in_name(self, tmp_path: Path) -> None:
        """When output filename has no 'train', appends '_val' to stem."""
        from chessgpt.cli.prepare import _merge_from_s3

        csv_content = "moves,outcome,checkmate_move_idx,ply_count\ne4 e5,1-0,-1,2\n"
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {
            "Contents": [{"Key": "prepared/lichess_2013-01.csv"}]
        }
        mock_s3.download_file.side_effect = lambda b, k, p: Path(p).write_text(csv_content)

        output_csv = str(tmp_path / "games.csv")

        with patch("boto3.client", return_value=mock_s3):
            _merge_from_s3(2013, "test-bucket", output_csv, None, 0.0)

        # val file should be games_val.csv
        assert (tmp_path / "games_val.csv").exists()
