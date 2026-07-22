"""Preflight model downloads: missing models are fetched up front, atomically.

The checks under test talk to Hugging Face through ``snapshot_download``, which
is mocked throughout: these tests pin down *when* a download happens, *where*
its bytes land, and what the report says afterwards — not the download itself.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from audiobook import preflight
from audiobook.preflight import OK, WARN, _check_asr_cache, _check_checkpoint


def fake_snapshot(*, files=("model.safetensors",), cache=True):
    """A stand-in for ``snapshot_download`` that materialises a checkpoint.

    Writes the named files into ``local_dir`` the way the real function would,
    including the ``.cache`` resume-metadata directory it leaves behind.
    """

    def download(remote_id, *, local_dir):
        directory = Path(local_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for name in files:
            (directory / name).write_text("weights", encoding="utf-8")
        if cache:
            (directory / ".cache" / "huggingface").mkdir(parents=True)
        return str(directory)

    return download


class CheckpointDownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.local_path = Path(self.tmp.name) / "models" / "checkpoint"

    def test_present_checkpoint_is_ok_without_downloading(self):
        self.local_path.mkdir(parents=True)
        with mock.patch("huggingface_hub.snapshot_download") as download:
            result = _check_checkpoint("TTS checkpoint", self.local_path, "org/model")
        download.assert_not_called()
        self.assertEqual(result.status, OK)
        self.assertEqual(result.detail, str(self.local_path))

    def test_missing_checkpoint_is_downloaded_and_renamed_into_place(self):
        with (
            mock.patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot()),
            mock.patch("builtins.print"),
        ):
            result = _check_checkpoint("TTS checkpoint", self.local_path, "org/model")
        self.assertEqual(result.status, OK)
        self.assertIn("downloaded on this run", result.detail)
        self.assertTrue((self.local_path / "model.safetensors").exists())
        # The staging directory is gone and the resume metadata went with it.
        self.assertFalse(self.local_path.with_name("checkpoint.partial").exists())
        self.assertFalse((self.local_path / ".cache").exists())

    def test_interrupted_download_warns_and_leaves_no_checkpoint(self):
        """A checkpoint must never exist half-written: rename is the last step."""

        with (
            mock.patch(
                "huggingface_hub.snapshot_download",
                side_effect=OSError("connection reset"),
            ),
            mock.patch("builtins.print"),
        ):
            result = _check_checkpoint("TTS checkpoint", self.local_path, "org/model")
        self.assertEqual(result.status, WARN)
        self.assertIn("connection reset", result.detail)
        self.assertIn("first use", result.detail)
        self.assertFalse(self.local_path.exists())


class AsrDownloadTests(unittest.TestCase):
    def test_disabled_transcription_touches_nothing(self):
        with (
            mock.patch.object(preflight, "REFERENCE_TRANSCRIBE", False),
            mock.patch("huggingface_hub.snapshot_download") as download,
        ):
            result = _check_asr_cache()
        download.assert_not_called()
        self.assertEqual(result.status, OK)
        self.assertIn("disabled", result.detail)

    def test_cached_model_is_ok_without_downloading(self):
        with (
            mock.patch.object(preflight, "REFERENCE_TRANSCRIBE", True),
            mock.patch(
                "huggingface_hub.try_to_load_from_cache",
                return_value="/cache/config.json",
            ),
            mock.patch("huggingface_hub.snapshot_download") as download,
        ):
            result = _check_asr_cache()
        download.assert_not_called()
        self.assertEqual(result.status, OK)
        self.assertIn("cached", result.detail)

    def test_missing_model_downloads_only_what_transcription_loads(self):
        """The Whisper repo ships redundant weight formats; fetch one of them."""

        with (
            mock.patch.object(preflight, "REFERENCE_TRANSCRIBE", True),
            mock.patch("huggingface_hub.try_to_load_from_cache", return_value=None),
            mock.patch("huggingface_hub.snapshot_download") as download,
            mock.patch("builtins.print"),
        ):
            result = _check_asr_cache()
        download.assert_called_once_with(
            preflight.ASR_MODEL, allow_patterns=preflight._ASR_FILE_PATTERNS
        )
        patterns = download.call_args.kwargs["allow_patterns"]
        self.assertIn("model.safetensors", patterns)
        self.assertNotIn("*.safetensors", patterns)  # would match the fp32 shards
        self.assertEqual(result.status, OK)
        self.assertIn("downloaded on this run", result.detail)

    def test_failed_download_warns_and_defers_to_first_use(self):
        with (
            mock.patch.object(preflight, "REFERENCE_TRANSCRIBE", True),
            mock.patch("huggingface_hub.try_to_load_from_cache", return_value=None),
            mock.patch(
                "huggingface_hub.snapshot_download",
                side_effect=OSError("offline"),
            ),
            mock.patch("builtins.print"),
        ):
            result = _check_asr_cache()
        self.assertEqual(result.status, WARN)
        self.assertIn("offline", result.detail)
        self.assertIn("first imported recording", result.detail)


if __name__ == "__main__":
    unittest.main()
