"""
Watch the inbox directory for new legal docs and ingest them into the vector DB.

Polls `config.INGEST_INBOX_DIR` every N seconds, ingests any files not already in the
registry (`data/.ingested_files.json`), runs the same pipeline as scripts/ingest_pipeline.py,
and marks them processed. Idempotent — safe to re-run.

Usage:
    python scripts/watch_ingest.py              # watch loop
    python scripts/watch_ingest.py --once       # single pass, exit
    python scripts/watch_ingest.py --inbox /custom/path
"""
import os
import sys
import json
import time
import argparse
import logging

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from scripts.ingest_pipeline import discover_files, process_files, merge_into_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_registry() -> set:
    """Load the set of already-processed file paths from the JSON registry."""
    path = config.INGEST_REGISTRY_PATH
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f).get("processed", []))
    except Exception as e:
        logger.warning(f"Failed to load registry from {path}: {e}")
        return set()


def save_registry(processed: set) -> None:
    """Atomically persist the registry to disk."""
    path = config.INGEST_REGISTRY_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"processed": sorted(processed)}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def scan_once(inbox: str = None) -> dict:
    """Single ingest pass: discover new files, ingest, mark processed. Returns stats dict."""
    inbox = inbox or config.INGEST_INBOX_DIR
    if not os.path.isdir(inbox):
        logger.warning(f"Inbox not found: {inbox} — creating it.")
        os.makedirs(inbox, exist_ok=True)
        return {"new": 0, "skipped": 0, "chunks_added": 0, "vectors_total": 0, "errors": []}

    registry = load_registry()
    all_files = discover_files(inbox)
    new_files = [f for f in all_files if f not in registry]

    if not new_files:
        logger.info(f"No new files in {inbox} (registry has {len(registry)} processed)")
        return {
            "new": 0,
            "skipped": len(all_files),
            "chunks_added": 0,
            "vectors_total": 0,
            "errors": [],
        }

    logger.info(f"📥 {len(new_files)} new file(s) to ingest from {inbox}")
    try:
        documents, chunks = process_files(new_files)
    except Exception as e:
        logger.exception("process_files errored")
        return {"new": 0, "skipped": len(all_files), "chunks_added": 0, "vectors_total": 0, "errors": [str(e)]}

    if not chunks:
        logger.info("No chunks produced (files were empty or unsupported).")
        return {"new": 0, "skipped": len(all_files), "chunks_added": 0, "vectors_total": 0, "errors": []}

    try:
        total_vectors, total_chunks = merge_into_indices(chunks, config.DATA_DIR)
    except Exception as e:
        logger.exception("merge_into_indices errored")
        return {"new": 0, "skipped": len(all_files), "chunks_added": 0, "vectors_total": 0, "errors": [str(e)]}

    registry.update(new_files)
    save_registry(registry)
    logger.info(f"✅ Ingested {len(new_files)} file(s) → +{len(chunks)} chunks; total vectors {total_vectors}")

    return {
        "new": len(new_files),
        "skipped": len(all_files) - len(new_files),
        "chunks_added": len(chunks),
        "vectors_total": total_vectors,
        "errors": [],
    }


def watch_loop(interval_s: int, inbox: str) -> None:
    """Run scan_once in a loop, sleeping `interval_s` seconds between passes."""
    logger.info(f"👀 Watching {inbox} (poll every {interval_s}s). Ctrl-C to exit.")
    while True:
        try:
            scan_once(inbox)
        except KeyboardInterrupt:
            logger.info("Stopping watch loop.")
            return
        except Exception:
            logger.exception("Scan errored — continuing")
        time.sleep(interval_s)


def main():
    parser = argparse.ArgumentParser(description="Watch and ingest new legal documents from an inbox.")
    parser.add_argument("--inbox", default=config.INGEST_INBOX_DIR, help="Directory to watch")
    parser.add_argument("--interval", type=int, default=config.INGEST_WATCH_INTERVAL_S, help="Poll interval (seconds)")
    parser.add_argument("--once", action="store_true", help="Single pass, don't loop")
    args = parser.parse_args()

    print("=" * 60)
    print("📥  كونان — Watch-Folder Ingest")
    print("=" * 60)
    print(f"   Inbox:    {args.inbox}")
    print(f"   Registry: {config.INGEST_REGISTRY_PATH}")
    print(f"   Mode:     {'one-shot' if args.once else f'watch ({args.interval}s)'}")
    print()

    if args.once:
        result = scan_once(args.inbox)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        watch_loop(args.interval, args.inbox)


if __name__ == "__main__":
    main()
