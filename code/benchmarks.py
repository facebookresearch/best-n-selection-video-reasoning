"""Benchmark loaders for the 5 video-QA datasets.

Every loader is a generator of dicts with a common schema::

    {
        "video_id": str,               # stable per-example id (used for cache keys)
        "video_path": str,             # absolute path to a decodable video file
        "question": str,
        "options": List[str],          # e.g. ["A. foo", "B. bar", ...]
        "answer_letter": str,          # ground-truth letter A/B/C/...
        "task_type": str,              # benchmark-specific task label
        "subtitle": Optional[str],     # concatenated SRT text if available
        "meta": dict,                  # everything else worth preserving
                                       # (e.g. Video-MME-v2 group ids for scoring)
    }

Every generator prunes samples whose video file is missing (log + skip) instead
of raising. All I/O is proxy-stripped up front.
"""

from __future__ import annotations

import os

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)

import glob
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------- paths
_HF_CACHE_HUB = Path(os.environ.get("HF_CACHE_HUB",
                                    os.path.expanduser("~/.cache/huggingface/hub")))
# Legacy Video-MME-v1 tree (not required by any benchmark the paper reports).
_LEGACY_VMME = Path(os.environ.get("VIDEO_MME_DIR", "/nonexistent/Video-MME"))


def _resolve_snapshot(dataset_slug: str) -> Path:
    """Return the (single) snapshot directory for an HF dataset in the cache."""
    root = _HF_CACHE_HUB / f"datasets--{dataset_slug}" / "snapshots"
    if not root.exists():
        raise FileNotFoundError(f"HF snapshot root missing: {root}")
    snaps = sorted(p for p in root.iterdir() if p.is_dir())
    if not snaps:
        raise FileNotFoundError(f"no snapshots under {root}")
    if len(snaps) > 1:
        logger.warning(f"multiple snapshots for {dataset_slug}; using latest {snaps[-1].name}")
    return snaps[-1]


def _load_srt(path: str) -> str:
    """Parse a .srt file into plain text (concatenated caption lines)."""
    if not os.path.exists(path):
        return ""
    lines: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.isdigit() or "-->" in s:
                continue
            lines.append(s)
    return " ".join(lines)


# ------------------------------------------------------------- Video-MME-v2
def _ensure_vmme_v2_video(video_id: str, snapshot_dir: Path, cache_root: str) -> Optional[str]:
    """Unzip on demand: videos/{ZZZ}.zip -> video_cache/vmme_v2/{video_id}.mp4

    Video-MME-v2 zip layout: 20 videos per zip, sequentially.
        001.zip contains 001.mp4..020.mp4
        002.zip contains 021.mp4..040.mp4
        ...
        020.zip contains 381.mp4..400.mp4
    Zip index for video_id N: ceil(N / 20) = (N-1)//20 + 1.
    """
    out_dir = Path(cache_root) / "vmme_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Already extracted?
    existing = sorted(out_dir.glob(f"{video_id}.*"))
    if existing:
        return str(existing[0])
    # Locate the zip that contains this video.
    try:
        vid_int = int(video_id)
    except ValueError:
        logger.warning(f"vmme-v2: non-integer video_id {video_id!r}")
        return None
    zip_num = (vid_int - 1) // 20 + 1
    zip_id = str(zip_num).zfill(3)
    zip_path = snapshot_dir / "videos" / f"{zip_id}.zip"
    if not zip_path.exists():
        logger.warning(f"vmme-v2: zip missing for video_id={video_id}: {zip_path}")
        return None
    try:
        with zipfile.ZipFile(zip_path) as zf:
            # Extract ONLY the file matching this video_id — no fallback
            # extractall (which would produce cross-video contamination).
            target_stem = str(vid_int).zfill(3)
            for n in zf.namelist():
                if Path(n).stem == target_stem:
                    zf.extract(n, out_dir)
                    src = out_dir / n
                    tgt = out_dir / f"{video_id}{src.suffix}"
                    if src != tgt:
                        src.rename(tgt)
                    return str(tgt)
            logger.warning(
                f"vmme-v2: video_id={video_id} not found in {zip_path.name} "
                f"(contents: {zf.namelist()[:3]}...)"
            )
    except zipfile.BadZipFile as e:
        logger.warning(f"vmme-v2: bad zip {zip_path}: {e}")
    return None


def load_video_mme_v2(
    cache_root: str,
    subset: str = "all",
    n_samples: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Iterate Video-MME-v2 test set (3200 QAs across 800 videos).

    subset: currently unused (whole set is one test split) but kept for API parity.
    """
    del subset
    snap = _resolve_snapshot("MME-Benchmarks--Video-MME-v2")
    parquet = snap / "test.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"vmme-v2 parquet missing: {parquet}")
    table = pq.read_table(parquet)
    rows = table.to_pylist()
    yielded = 0
    for r in rows:
        if n_samples is not None and yielded >= n_samples:
            return
        vid = str(r["video_id"])
        vpath = _ensure_vmme_v2_video(vid, snap, cache_root)
        if vpath is None:
            continue
        options = _split_options_block(r.get("options", ""))
        yield {
            "video_id": vid,
            "video_path": vpath,
            "question": r["question"],
            "options": options,
            "answer_letter": r["answer"].strip().upper(),
            "task_type": r.get("third_head") or r.get("second_head") or "unknown",
            "subtitle": None,  # v2 does not ship subtitles in the released parquet
            "meta": {
                "question_id": r.get("question_id"),
                "level": r.get("level"),
                "group_type": r.get("group_type"),
                "group_structure": r.get("group_structure"),
                "second_head": r.get("second_head"),
                "third_head": r.get("third_head"),
                "url": r.get("url"),
                "benchmark": "video-mme-v2",
            },
        }
        yielded += 1


def _split_options_block(block: Any) -> List[str]:
    """Video-MME-v2 stores options as a single newline-joined string;
    Video-MME (v1) stores them as a list of strings. Normalize either shape."""
    if isinstance(block, list):
        return [str(o).strip() for o in block if str(o).strip()]
    if isinstance(block, str):
        parts = [p.strip() for p in block.splitlines() if p.strip()]
        return parts
    return []


# ---------------------------------------------------------------- Video-MME
def _ensure_vmme_video(video_key: str, cache_root: str) -> Optional[str]:
    """Video-MME v1 videos are packaged as videos_chunked_XX.zip. Legacy path
    already has them extracted per-videoID. Prefer legacy extracted mp4."""
    legacy = _LEGACY_VMME / "data" / f"{video_key}.mp4"
    if legacy.exists():
        return str(legacy)
    # Fallback: search extracted cache.
    out_dir = Path(cache_root) / "vmme_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    hits = list(out_dir.glob(f"{video_key}.*"))
    if hits:
        return str(hits[0])
    # Try to extract from a chunked zip on demand.
    for zpath in sorted(_LEGACY_VMME.glob("videos_chunked_*.zip")):
        try:
            with zipfile.ZipFile(zpath) as zf:
                for n in zf.namelist():
                    if Path(n).stem == video_key:
                        zf.extract(n, out_dir)
                        src = out_dir / n
                        tgt = out_dir / f"{video_key}{src.suffix}"
                        if src != tgt:
                            src.rename(tgt)
                        return str(tgt)
        except zipfile.BadZipFile:
            continue
    return None


def load_video_mme(
    cache_root: str,
    split: str = "long",
    n_samples: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Iterate Video-MME v1 (900 videos x 3 questions = 2700).
    split: short | medium | long | all
    """
    parquet_candidates = [
        _LEGACY_VMME / "videomme" / "test-00000-of-00001.parquet",
        _LEGACY_VMME / "data" / "test-00000-of-00001.parquet",
    ]
    parquet = next((p for p in parquet_candidates if p.exists()), None)
    if parquet is None:
        raise FileNotFoundError(f"vmme v1 parquet not found in: {parquet_candidates}")
    table = pq.read_table(parquet)
    rows = table.to_pylist()
    split = split.lower()
    yielded = 0
    for r in rows:
        if split != "all" and r.get("duration", "").lower() != split:
            continue
        if n_samples is not None and yielded >= n_samples:
            return
        video_key = r["videoID"]  # YouTube-style id
        vpath = _ensure_vmme_video(video_key, cache_root)
        if vpath is None:
            logger.warning(f"vmme: video missing for videoID={video_key}")
            continue
        subtitle_path = _LEGACY_VMME / "subtitle" / f"{video_key}.srt"
        subtitle_text = _load_srt(str(subtitle_path)) if subtitle_path.exists() else None
        yield {
            "video_id": str(r["video_id"]),
            "video_path": vpath,
            "question": r["question"],
            "options": _split_options_block(r["options"]),
            "answer_letter": r["answer"].strip().upper(),
            "task_type": r.get("task_type", "unknown"),
            "subtitle": subtitle_text,
            "meta": {
                "question_id": r.get("question_id"),
                "duration": r.get("duration"),
                "domain": r.get("domain"),
                "sub_category": r.get("sub_category"),
                "videoID": video_key,
                "url": r.get("url"),
                "benchmark": "video-mme",
            },
        }
        yielded += 1


# ------------------------------------------------------------------- LVBench
def load_lvbench(
    cache_root: str,
    n_samples: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """LVBench (zai-org/LVBench) — 103 hour-long videos with fine-grained QA.

    Meta file: video_info.meta.jsonl (one line per video, embeds all QAs).
    Videos ship separately (may still be downloading); we search common
    locations and skip missing ones.
    """
    try:
        snap = _resolve_snapshot("zai-org--LVBench")
    except FileNotFoundError as e:
        logger.error(f"lvbench: {e}; skipping benchmark")
        return
    meta_path = snap / "video_info.meta.jsonl"
    if not meta_path.exists():
        logger.warning(f"lvbench: meta missing {meta_path}; skipping")
        return
    # Candidate video roots (LVBench ships as .zip that unpacks to `all_videos/`).
    video_roots = [
        snap / "all_videos",
        snap / "videos",
        Path(cache_root) / "lvbench",
    ]
    yielded = 0
    with open(meta_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            key = r["key"]
            vpath: Optional[str] = None
            for root in video_roots:
                for ext in (".mp4", ".mkv", ".webm"):
                    cand = root / f"{key}{ext}"
                    if cand.exists():
                        vpath = str(cand)
                        break
                if vpath is not None:
                    break
            if vpath is None:
                logger.warning(f"lvbench: video missing for key={key}")
                continue
            for qa in r.get("qa", []):
                if n_samples is not None and yielded >= n_samples:
                    return
                q_full = qa["question"]  # already contains options in "(A) ...\n(B) ..." form
                # Split question / options at first "(A)"
                m = re.search(r"\n\s*\(A\)", q_full)
                if m:
                    q_text = q_full[: m.start()].strip()
                    opts_block = q_full[m.start() :]
                    options = re.findall(r"\(([A-Z])\)\s*([^\n(]+)", opts_block)
                    options_list = [f"{L}. {o.strip()}" for L, o in options]
                else:
                    q_text = q_full
                    options_list = []
                yield {
                    "video_id": key,
                    "video_path": vpath,
                    "question": q_text,
                    "options": options_list,
                    "answer_letter": qa["answer"].strip().upper(),
                    "task_type": ",".join(qa.get("question_type", [])) or "unknown",
                    "subtitle": None,
                    "meta": {
                        "uid": qa.get("uid"),
                        "time_reference": qa.get("time_reference"),
                        "video_type": r.get("type"),
                        "video_info": r.get("video_info"),
                        "benchmark": "lvbench",
                    },
                }
                yielded += 1


# --------------------------------------------------------------------- MLVU
_MLVU_TASKS = [
    "1_plotQA",
    "2_needle",
    "3_ego",
    "4_count",
    "5_order",
    "6_anomaly_reco",
    "7_topic_reasoning",
    "8_sub_scene",
    "9_summary",
]


def load_mlvu(
    cache_root: str,
    split: str = "dev",
    n_samples: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """MLVU (MLVU/MVLU) — 9 tasks. The public release ships per-task JSON
    annotations under `json/` (MC tasks) and video files under `video/{task}/`.
    """
    del split  # dev is the standard MC split; kept for API parity
    try:
        snap = _resolve_snapshot("MLVU--MVLU")
    except FileNotFoundError as e:
        logger.error(f"mlvu: {e}; skipping benchmark")
        return
    root = snap / "MLVU"
    json_dir = root / "json"
    video_dir = root / "video"
    if not json_dir.exists():
        logger.warning(f"mlvu: json dir missing at {json_dir}; skipping (dataset still downloading?)")
        return
    yielded = 0
    for task in _MLVU_TASKS:
        task_json = json_dir / f"{task}.json"
        if not task_json.exists():
            logger.warning(f"mlvu: missing {task_json}")
            continue
        with open(task_json) as f:
            rows = json.load(f)
        for r in rows:
            if n_samples is not None and yielded >= n_samples:
                return
            video_name = r.get("video") or r.get("video_path")
            if not video_name:
                continue
            vpath = str(video_dir / task / video_name)
            if not os.path.exists(vpath):
                # Some annotations already include the task prefix.
                alt = str(video_dir / video_name)
                if os.path.exists(alt):
                    vpath = alt
                else:
                    logger.warning(f"mlvu: missing video {vpath}")
                    continue
            options_raw = r.get("candidates") or r.get("options") or []
            if options_raw and isinstance(options_raw[0], str) and not re.match(r"^[A-Z]\.\s", options_raw[0]):
                options = [f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(options_raw)]
            else:
                options = [str(o) for o in options_raw]
            answer = r.get("answer")
            if isinstance(answer, int):
                answer_letter = chr(ord("A") + answer)
            elif isinstance(answer, str):
                # MLVU sometimes stores full option text — resolve back to letter.
                if len(answer) == 1 and answer.isalpha():
                    answer_letter = answer.upper()
                else:
                    match_idx = next(
                        (i for i, o in enumerate(options_raw) if str(o).strip() == answer.strip()),
                        None,
                    )
                    if match_idx is None:
                        logger.warning(f"mlvu: unresolvable answer {answer!r}")
                        continue
                    answer_letter = chr(ord("A") + match_idx)
            else:
                continue
            yield {
                "video_id": f"{task}/{video_name}",
                "video_path": vpath,
                "question": r["question"],
                "options": options,
                "answer_letter": answer_letter,
                "task_type": task,
                "subtitle": None,
                "meta": {
                    "task": task,
                    "duration": r.get("duration"),
                    "benchmark": "mlvu",
                },
            }
            yielded += 1


def load_egoschema(
    cache_root: str,
    n_samples: Optional[int] = None,
    subset: str = "subset",
    include_subtitle: bool = False,
    **_,
) -> Iterator[Dict[str, Any]]:
    """EgoSchema Subset (500 QAs). Parquet cols: question_idx, question, video_idx,
    option (list of 'A. ...'), answer (int 0-4). Videos at benchmarks/egoschema/videos/videos/."""
    del include_subtitle
    import pandas as pd
    root = Path(os.environ["VRITS_DATA"] + "/benchmarks/egoschema")
    pq_path = root / ("Subset" if subset == "subset" else "MC") / "test-00000-of-00001.parquet"
    if not pq_path.exists():
        raise FileNotFoundError(f"egoschema parquet missing: {pq_path}")
    vdir = root / "videos" / "videos"
    df = pd.read_parquet(pq_path)
    yielded = 0
    for _, r in df.iterrows():
        if n_samples is not None and yielded >= n_samples:
            return
        vid = str(r["video_idx"])
        vpath = vdir / f"{vid}.mp4"
        if not vpath.exists():
            continue
        try:
            ans_letter = "ABCDE"[int(r["answer"])]
        except (ValueError, TypeError):
            continue
        yield {
            "video_id": vid,
            "video_path": str(vpath),
            "question": str(r["question"]),
            "options": list(r["option"]),
            "answer_letter": ans_letter,
            "task_type": "egoschema",
            "subtitle": None,
            "meta": {"question_id": str(r["question_idx"])},
        }
        yielded += 1


# --- surviving-suite loaders (ported from the muse-dir canonical benchmarks.py;
#     additive only — leaves the vmme-v2 path byte-identical). Deps (_resolve_snapshot,
#     logger, glob/json/re/os, Path, pq, typing) all already present above.
_TC_OPT_RE = re.compile(r"^\(?([A-H])[\.\)]\s")


def load_tempcompass(cache_root: str, subset: str = "multi-choice", n_samples: Optional[int] = None):
    """TempCompass (fine-grained temporal perception). Default subset = multi-choice (4-way MCQ).
    Videos pre-unzipped to {cache_root}/tempcompass/videos/{video_id}.mp4."""
    try:
        snap = _resolve_snapshot("lmms-lab--TempCompass")
    except FileNotFoundError as e:
        logger.error(f"tempcompass: {e}; skipping benchmark")
        return
    pq_files = sorted(glob.glob(str(snap / subset / "*.parquet")))
    if not pq_files:
        logger.warning(f"tempcompass: no parquet for subset {subset} under {snap}")
        return
    video_dir = Path(cache_root) / "tempcompass" / "videos"
    yielded = 0
    for pqf in pq_files:
        for r in pq.read_table(pqf).to_pylist():
            if n_samples is not None and yielded >= n_samples:
                return
            vid = str(r["video_id"])
            vpath = str(video_dir / f"{vid}.mp4")
            if not os.path.exists(vpath):
                logger.warning(f"tempcompass: missing video {vpath}")
                continue
            lines = [ln.strip() for ln in str(r["question"]).splitlines() if ln.strip()]
            options = [ln for ln in lines if _TC_OPT_RE.match(ln)]
            stem = " ".join(ln for ln in lines if not _TC_OPT_RE.match(ln))
            if len(options) < 2:
                continue
            ans = str(r["answer"]).strip()
            m = _TC_OPT_RE.match(ans)
            if m:
                answer_letter = m.group(1).upper()
            else:
                idx = next((i for i, o in enumerate(options)
                            if o.split(".", 1)[-1].strip() == ans or o.strip() == ans), None)
                if idx is None:
                    continue
                answer_letter = chr(ord("A") + idx)
            yield {
                "video_id": vid, "video_path": vpath, "question": stem, "options": options,
                "answer_letter": answer_letter, "task_type": f"{subset}:{r.get('dim','')}",
                "subtitle": None,
                "meta": {"dim": r.get("dim"), "subset": subset, "benchmark": "tempcompass"},
            }
            yielded += 1


def load_mlvu_test(
    cache_root: str,
    n_samples: Optional[int] = None,
    **_,
) -> Iterator[Dict[str, Any]]:
    """MLVU-Test (MLVU/MLVU_Test) — the held-out TEST split: 502 multiple-choice
    questions over 349 long videos, 9 task types. Public MCQ ground truth ships in
    test-ground-truth/test_mcq_gt.json (answer stored as candidate TEXT -> resolved
    to a letter here). Videos pre-staged to {cache_root}/mlvu_test/video/{video}."""
    try:
        snap = _resolve_snapshot("MLVU--MLVU_Test")
    except FileNotFoundError as e:
        logger.error(f"mlvu_test: {e}; skipping benchmark")
        return
    gt_path = snap / "test-ground-truth" / "test_mcq_gt.json"
    if not gt_path.exists():
        logger.warning(f"mlvu_test: GT missing at {gt_path}; skipping")
        return
    video_dir = Path(cache_root) / "mlvu_test" / "video"
    with open(gt_path) as f:
        rows = json.load(f)
    yielded = 0
    for r in rows:
        if n_samples is not None and yielded >= n_samples:
            return
        video_name = r.get("video")
        if not video_name:
            continue
        vpath = str(video_dir / video_name)
        if not os.path.exists(vpath):
            logger.warning(f"mlvu_test: missing video {vpath}")
            continue
        options_raw = r.get("candidates") or []
        options = [f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(options_raw)]
        answer = str(r.get("answer", "")).strip()
        match_idx = next((i for i, o in enumerate(options_raw) if str(o).strip() == answer), None)
        if match_idx is None:
            logger.warning(f"mlvu_test: unresolvable answer {answer!r}")
            continue
        answer_letter = chr(ord("A") + match_idx)
        yield {
            "video_id": video_name,
            "video_path": vpath,
            "question": r["question"],
            "options": options,
            "answer_letter": answer_letter,
            "task_type": r.get("question_type", ""),
            "subtitle": None,
            "meta": {
                "question_id": r.get("question_id"),
                "task": r.get("question_type"),
                "duration": r.get("duration"),
                "benchmark": "mlvu_test",
            },
        }
        yielded += 1


def load_vrbench(cache_root: str, n_samples: Optional[int] = None, n_shards: int = 1, shard: int = 0):
    """VRBench (OpenGVLab) FULL test set: 8243 MCQ / 960 long narrative videos, 7 reasoning types.
    Videos extracted to datasets_ext/vrbench_videos/v001_360p/{id}.mp4. Options dict A-D, answer letter.
    n_shards/shard split the question stream (global idx mod n_shards == shard) for data-parallel BoN."""
    root = _HF_CACHE_HUB / "datasets--OpenGVLab--VRBench" / "snapshots"
    snaps = sorted(root.iterdir()) if root.exists() else []
    if not snaps:
        logger.error("vrbench: snapshot missing; skipping"); return
    evalf = snaps[-1] / "VRBench_eval.jsonl"
    viddir = Path(os.environ.get("VRBENCH_VIDEO_DIR",
                             os.environ.get("VRITS_DATA", "data")
                             + "/vrbench_videos/v001_360p"))
    gi = 0; yielded = 0
    for line in open(evalf):
        try:
            r = json.loads(line)
        except Exception:
            continue
        vpath = str(viddir / Path(r.get("video_path", "")).name)
        if not os.path.exists(vpath):
            continue
        for qid, qa in (r.get("mcq") or {}).items():
            idx = gi; gi += 1
            if idx % n_shards != shard:
                continue
            if n_samples is not None and yielded >= n_samples:
                return
            opts_d = qa.get("options") or {}
            options = [f"{k}. {opts_d[k]}" for k in sorted(opts_d.keys())]
            ans = str(qa.get("answer", "")).strip().upper()
            answer_letter = ans[0] if ans[:1] in "ABCDEFGH" else ""
            if len(options) < 2 or not answer_letter:
                continue
            yield {
                "video_id": r.get("video_id"), "video_path": vpath,
                "question": qa.get("question", ""), "options": options,
                "answer_letter": answer_letter, "task_type": qa.get("reasoning_type", "vrbench"),
                "subtitle": None,
                "meta": {"question_id": f"{r.get('video_id')}__{qid}",
                         "reasoning_type": qa.get("reasoning_type"), "benchmark": "vrbench"},
            }
            yielded += 1


# ------------------------------------------------------------------- dispatch
def load_video_holmes(cache_root: str, n_samples: Optional[int] = None):
    """Video-Holmes (TencentARC) -- 6-way MCQ (A-F) multi-hop causal reasoning over 270
    suspense short films, 1837 questions, 7 task types (SR/IMC/TCI/TA/MHR/CTI/PAR).
    Approved Video-MME-v2 replacement. Videos: videos.zip -> {cache_root}/video_holmes/
    videos_cropped/{video ID}.mp4. Non-null integer Question ID (no key collapse)."""
    try:
        snap = _resolve_snapshot("TencentARC--Video-Holmes")
    except FileNotFoundError as e:
        logger.error(f"video-holmes: {e}; skipping benchmark")
        return
    jp = snap / "test_Video-Holmes.json"
    if not jp.exists():
        logger.warning(f"video-holmes: test json missing at {jp}; skipping")
        return
    video_dir = Path(cache_root) / "video_holmes" / "videos_cropped"
    with open(jp) as f:
        items = json.load(f)
    yielded = 0
    for it in items:
        if n_samples is not None and yielded >= n_samples:
            return
        vid = str(it["video ID"])
        vpath = str(video_dir / f"{vid}.mp4")
        if not os.path.exists(vpath):
            logger.warning(f"video-holmes: missing video {vpath}")
            continue
        opts = it.get("Options") or {}
        options = [f"{k}. {opts[k]}" for k in sorted(opts) if str(opts.get(k, "")).strip()]
        ans = str(it.get("Answer", "")).strip().upper()[:1]
        if len(options) < 2 or ans not in "ABCDEF":
            continue
        yield {
            "video_id": vid, "video_path": vpath, "question": it.get("Question", ""),
            "options": options, "answer_letter": ans,
            "task_type": it.get("Question Type", "video-holmes"), "subtitle": None,
            "meta": {"question_id": it.get("Question ID"), "benchmark": "video-holmes",
                     "explanation": it.get("Explanation")},
        }
        yielded += 1


BENCHMARK_LOADERS = {
    "video-mme-v2": load_video_mme_v2,
    "video-mme": load_video_mme,
    "lvbench": load_lvbench,
    "mlvu": load_mlvu,
    "mlvu_test": load_mlvu_test,
    "egoschema": load_egoschema,
    "tempcompass": load_tempcompass,
    "vrbench": load_vrbench,
    "video-holmes": load_video_holmes,
}


def load_benchmark(name: str, cache_root: str, **kwargs) -> Iterable[Dict[str, Any]]:
    if name not in BENCHMARK_LOADERS:
        raise ValueError(f"unknown benchmark: {name} (known: {list(BENCHMARK_LOADERS)})")
    return BENCHMARK_LOADERS[name](cache_root=cache_root, **kwargs)
