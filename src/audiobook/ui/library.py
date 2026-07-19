"""Enumerate and mutate the voices directory.

``resolve_voice`` answers "load this one"; the frontend also needs "what is
there" and "save this change", which is all this module adds.  It stays on the
same on-disk layout the CLI uses, so a voice created here works from the command
line and vice versa — the directory is the database.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..config import (
    DEFAULT_OUTPUT_DIR,
    VOICE_REFERENCE_AUDIO_FILENAME,
    VOICE_REFERENCE_METADATA_FILENAME,
    VOICES_DIR,
)
from ..synthesis.voices import AUDIO_SUFFIXES


@dataclass(frozen=True)
class VoiceEntry:
    """A selectable voice, described without decoding its audio.

    Listing must stay cheap: the picker refreshes on every tab switch, and
    decoding every reference to fill a dropdown would make that crawl.
    """

    spec: str
    label: str
    audio_path: Path
    transcript_path: Path
    ref_text: str | None
    instruct: str | None
    designed: bool

    @property
    def has_transcript(self) -> bool:
        return bool(self.ref_text)


def _designed_voice(voice_dir: Path) -> VoiceEntry | None:
    metadata_path = voice_dir / VOICE_REFERENCE_METADATA_FILENAME
    audio_path = voice_dir / VOICE_REFERENCE_AUDIO_FILENAME
    if not metadata_path.exists() or not audio_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return VoiceEntry(
        spec=voice_dir.name,
        label=f"{voice_dir.name}  (designed)",
        audio_path=audio_path,
        transcript_path=metadata_path,
        ref_text=metadata.get("ref_text") or None,
        instruct=metadata.get("instruct") or None,
        designed=True,
    )


def _recorded_voice(path: Path) -> VoiceEntry:
    transcript_path = path.with_suffix(".txt")
    ref_text = (
        transcript_path.read_text(encoding="utf-8").strip()
        if transcript_path.exists()
        else None
    )
    return VoiceEntry(
        spec=str(path),
        label=f"{path.stem}  (recording)",
        audio_path=path,
        transcript_path=transcript_path,
        ref_text=ref_text or None,
        instruct=None,
        designed=False,
    )


def list_voices(voices_dir: Path = VOICES_DIR) -> list[VoiceEntry]:
    """Every usable voice, designed folders first then loose recordings."""

    if not voices_dir.exists():
        return []

    designed = [
        entry
        for child in sorted(voices_dir.iterdir())
        if child.is_dir() and (entry := _designed_voice(child)) is not None
    ]
    recorded = [
        _recorded_voice(child)
        for child in sorted(voices_dir.iterdir())
        if child.is_file() and child.suffix.lower() in AUDIO_SUFFIXES
    ]
    return designed + recorded


def find_voice(spec: str, voices_dir: Path = VOICES_DIR) -> VoiceEntry | None:
    """Look a voice up by the spec stored in the dropdown."""

    return next((v for v in list_voices(voices_dir) if v.spec == spec), None)


def save_transcript(entry: VoiceEntry, text: str) -> str:
    """Write a corrected transcript back to wherever that voice keeps it.

    An empty transcript is a meaningful choice, not a mistake: it drops the
    voice to timbre-only cloning, which is the right move when the recording
    and its transcript disagree and there is no time to fix the words.
    """

    text = text.strip()
    if entry.designed:
        metadata = json.loads(entry.transcript_path.read_text(encoding="utf-8"))
        metadata["ref_text"] = text
        entry.transcript_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    elif text:
        entry.transcript_path.write_text(text + "\n", encoding="utf-8")
    elif entry.transcript_path.exists():
        entry.transcript_path.unlink()

    mode = "timbre + prosody" if text else "timbre only"
    return f"Saved to {entry.transcript_path} — this voice now clones {mode}."


def rename_voice(entry: VoiceEntry, new_name: str) -> str:
    """Rename a voice on disk and return the spec it now answers to.

    Renaming must move everything that makes the voice findable — the designed
    folder, or the recording plus its transcript sidecar — or the next
    ``list_voices`` would show a half-renamed orphan.
    """

    new_name = new_name.strip().replace("/", "_")
    if not new_name:
        raise ValueError("Give the voice a new name.")

    if entry.designed:
        source_dir = entry.audio_path.parent
        target_dir = source_dir.with_name(new_name)
        if target_dir.exists():
            raise ValueError(f"{target_dir} already exists.")
        source_dir.rename(target_dir)
        metadata_path = target_dir / VOICE_REFERENCE_METADATA_FILENAME
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["slug"] = new_name
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return new_name

    target = entry.audio_path.with_name(new_name + entry.audio_path.suffix)
    if target.exists():
        raise ValueError(f"{target} already exists.")
    entry.audio_path.rename(target)
    if entry.transcript_path.exists():
        entry.transcript_path.rename(target.with_suffix(".txt"))
    return str(target)


def delete_voice(entry: VoiceEntry) -> str:
    """Remove a voice from disk entirely."""

    if entry.designed:
        shutil.rmtree(entry.audio_path.parent)
        return f"Deleted {entry.audio_path.parent}."
    entry.audio_path.unlink()
    if entry.transcript_path.exists():
        entry.transcript_path.unlink()
    return f"Deleted {entry.audio_path}."


def import_recording(
    upload_path: str | Path, name: str = "", voices_dir: Path = VOICES_DIR
) -> Path:
    """Copy an uploaded recording into the voices directory under *name*.

    Gradio hands over a temp file that is cleaned up later, so the audio has to
    be copied somewhere permanent before it can become a voice.
    """

    source = Path(upload_path)
    stem = (name.strip() or source.stem).replace("/", "_")
    voices_dir.mkdir(parents=True, exist_ok=True)
    destination = voices_dir / f"{stem}{source.suffix.lower()}"
    if destination.resolve() != source.resolve():
        shutil.copyfile(source, destination)
    return destination


def list_prepared_scripts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Prepared-book artifacts available to narrate."""

    if not output_dir.exists():
        return []
    return sorted(
        path
        for path in output_dir.rglob("*.json")
        if path.name not in {"chunk_manifest.json"} and _is_prepared_book(path)
    )


def _is_prepared_book(path: Path) -> bool:
    """Cheap structural check so unrelated JSON never reaches the narrator."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            head = handle.read(4096)
    except OSError:
        return False
    return '"chapters"' in head and '"schema_version"' in head


def list_audiobooks(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Finished and preview audiobooks, newest first."""

    if not output_dir.exists():
        return []
    return sorted(
        output_dir.rglob("*.m4b"), key=lambda p: p.stat().st_mtime, reverse=True
    )


__all__ = [
    "VoiceEntry",
    "delete_voice",
    "find_voice",
    "import_recording",
    "list_audiobooks",
    "list_prepared_scripts",
    "list_voices",
    "rename_voice",
    "save_transcript",
]
