#!/usr/bin/env python3

"""
search_enamine_similarity_worker.py

Distributed exhaustive Morgan/Tanimoto search of split Enamine
*.cxsmiles.bz2 archives.

Each invocation:

    1. Reads a global manifest of cxsmiles.bz2 files.
    2. Selects one contiguous shard based on task index/task count.
    3. Processes multiple archives concurrently with multiprocessing.
    4. Each archive retains its local top-K.
    5. The parent merges archive results into the exact shard top-K.
    6. Writes one shard_XXXXX.npz file for later global merging.

Designed for use as an LSF job array.

The actual ligand records are NOT retained here; only:

    Tanimoto score
    global archive index
    ligand line number

are kept.

The final merger can later recover SMILES + ligand ID.
"""

import argparse
import bz2
import hashlib
import json
import os
import shutil
import sys
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


# ============================================================
# Worker globals
# ============================================================

QUERY_FP = None
FP_GENERATOR = None

TOP_K = None
BATCH_SIZE = None
HEADER_LINES = None


# ============================================================
# Worker initialization
# ============================================================

def init_worker(
    query_smiles,
    top_k,
    batch_size,
    header_lines,
    radius,
    fp_size,
    use_chirality,
):
    """
    Initialize RDKit objects once per multiprocessing worker.
    """

    global QUERY_FP
    global FP_GENERATOR
    global TOP_K
    global BATCH_SIZE
    global HEADER_LINES

    query_mol = Chem.MolFromSmiles(query_smiles)

    if query_mol is None:
        raise ValueError(
            f"Could not parse query SMILES: {query_smiles}"
        )

    FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=fp_size,
        includeChirality=use_chirality,
    )

    QUERY_FP = FP_GENERATOR.GetFingerprint(query_mol)

    TOP_K = top_k
    BATCH_SIZE = batch_size
    HEADER_LINES = header_lines


# ============================================================
# Utilities
# ============================================================

def manifest_sha256(paths):
    """
    Hash the ordered manifest so all distributed jobs can later
    be verified as belonging to the exact same search.
    """

    h = hashlib.sha256()

    for path in paths:
        h.update(str(path).encode("utf-8"))
        h.update(b"\n")

    return h.hexdigest()


def get_task_slice(n_items, task_index, task_count):
    """
    Divide N archives into task_count contiguous, approximately
    equal-sized groups.

    task_index is 1-based, matching LSF_JOBINDEX.
    """

    if task_index < 1 or task_index > task_count:
        raise ValueError(
            f"task-index must be between 1 and {task_count}, "
            f"got {task_index}"
        )

    zero_index = task_index - 1

    start = (n_items * zero_index) // task_count
    end = (n_items * (zero_index + 1)) // task_count

    return start, end


def reduce_top_k(scores, lines, k):
    """
    Retain the K largest scores without completely sorting
    the array.
    """

    n = scores.size

    if n <= k:
        return scores, lines

    idx = np.argpartition(
        scores,
        n - k,
    )[n - k:]

    return scores[idx], lines[idx]


def update_archive_top_k(
    top_scores,
    top_lines,
    new_scores,
    new_lines,
    k,
):
    """
    Incorporate one batch into an archive's running top-K.
    """

    if new_scores.size == 0:
        return top_scores, top_lines

    # --------------------------------------------------------
    # Initial filling
    # --------------------------------------------------------

    if top_scores.size < k:

        combined_scores = np.concatenate(
            (top_scores, new_scores)
        )

        combined_lines = np.concatenate(
            (top_lines, new_lines)
        )

        return reduce_top_k(
            combined_scores,
            combined_lines,
            k,
        )

    # --------------------------------------------------------
    # Once full, discard candidates definitely below the
    # existing minimum.
    #
    # Keep equality here because tied cutoff values are still
    # legitimate top-K candidates.
    # --------------------------------------------------------

    threshold = top_scores.min()

    mask = new_scores >= threshold

    if not np.any(mask):
        return top_scores, top_lines

    combined_scores = np.concatenate(
        (
            top_scores,
            new_scores[mask],
        )
    )

    combined_lines = np.concatenate(
        (
            top_lines,
            new_lines[mask],
        )
    )

    return reduce_top_k(
        combined_scores,
        combined_lines,
        k,
    )


# ============================================================
# Fingerprint one batch
# ============================================================

def process_batch(
    smiles_batch,
    line_batch,
    top_scores,
    top_lines,
):
    """
    Parse a batch of SMILES, generate Morgan fingerprints,
    and perform a batched Tanimoto calculation.
    """

    fps = []
    valid_lines = []

    for smiles, line_number in zip(
        smiles_batch,
        line_batch,
    ):

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        fp = FP_GENERATOR.GetFingerprint(mol)

        fps.append(fp)
        valid_lines.append(line_number)

    if not fps:
        return (
            top_scores,
            top_lines,
            0,
        )

    similarities = DataStructs.BulkTanimotoSimilarity(
        QUERY_FP,
        fps,
    )

    # Use float64 so ranking is not unnecessarily reduced
    # to float32 precision.
    scores = np.asarray(
        similarities,
        dtype=np.float64,
    )

    # Split archives contain <= 10M ligands, so uint32 is
    # comfortably sufficient.
    lines = np.asarray(
        valid_lines,
        dtype=np.uint32,
    )

    top_scores, top_lines = update_archive_top_k(
        top_scores,
        top_lines,
        scores,
        lines,
        TOP_K,
    )

    return (
        top_scores,
        top_lines,
        len(fps),
    )


# ============================================================
# Search one archive
# ============================================================

def process_archive(task):
    """
    Search one cxsmiles.bz2 archive.

    task:
        archive_path
        archive_global_index
        result_path
    """

    (
        archive_path,
        archive_global_index,
        result_path,
    ) = task

    archive_path = Path(archive_path)
    result_path = Path(result_path)

    start_time = time.time()

    top_scores = np.empty(
        0,
        dtype=np.float64,
    )

    top_lines = np.empty(
        0,
        dtype=np.uint32,
    )

    smiles_batch = []
    line_batch = []

    total_records = 0
    valid_records = 0

    print(
        f"[START] archive_index={archive_global_index} "
        f"{archive_path}",
        flush=True,
    )

    with bz2.open(
        archive_path,
        mode="rt",
        errors="replace",
    ) as handle:

        # ----------------------------------------------------
        # Skip original header
        # ----------------------------------------------------

        for _ in range(HEADER_LINES):
            next(handle, None)

        # ----------------------------------------------------
        # Ligand lines are zero-based AFTER the header.
        # ----------------------------------------------------

        for ligand_line_number, line in enumerate(handle):

            total_records += 1

            stripped = line.strip()

            if not stripped:
                continue

            fields = stripped.split()

            if not fields:
                continue

            # Base SMILES is always the first token.
            #
            # Any CXSMILES information such as:
            #
            #     |&1:11,14|
            #
            # occurs afterward and is not required for
            # fingerprinting.
            smiles = fields[0]

            smiles_batch.append(smiles)
            line_batch.append(ligand_line_number)

            if len(smiles_batch) >= BATCH_SIZE:

                (
                    top_scores,
                    top_lines,
                    n_valid,
                ) = process_batch(
                    smiles_batch,
                    line_batch,
                    top_scores,
                    top_lines,
                )

                valid_records += n_valid

                smiles_batch.clear()
                line_batch.clear()

                if total_records % 1_000_000 < BATCH_SIZE:
                    print(
                        f"[PROGRESS] "
                        f"{archive_path.name}: "
                        f"{total_records:,} records",
                        flush=True,
                    )

        # ----------------------------------------------------
        # Final partial batch
        # ----------------------------------------------------

        if smiles_batch:

            (
                top_scores,
                top_lines,
                n_valid,
            ) = process_batch(
                smiles_batch,
                line_batch,
                top_scores,
                top_lines,
            )

            valid_records += n_valid

    # --------------------------------------------------------
    # Sort archive result descending.
    # --------------------------------------------------------

    order = np.argsort(
        top_scores,
        kind="stable",
    )[::-1]

    top_scores = top_scores[order]
    top_lines = top_lines[order]

    # --------------------------------------------------------
    # Atomic-ish result creation.
    # --------------------------------------------------------

    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = Path(
        str(result_path) + ".tmp.npz"
    )

    np.savez(
        temp_path,
        scores=top_scores,
        lines=top_lines,
    )

    os.replace(
        temp_path,
        result_path,
    )

    elapsed = time.time() - start_time

    rate = (
        total_records / elapsed
        if elapsed > 0
        else 0.0
    )

    print(
        f"[DONE] {archive_path.name}: "
        f"{total_records:,} records; "
        f"{valid_records:,} valid; "
        f"{top_scores.size:,} retained; "
        f"{elapsed:.1f} sec; "
        f"{rate:,.0f} ligands/sec",
        flush=True,
    )

    return {
        "archive": str(archive_path),
        "archive_index": int(archive_global_index),
        "result": str(result_path),
        "records": int(total_records),
        "valid": int(valid_records),
        "retained": int(top_scores.size),
        "elapsed": float(elapsed),
    }


# ============================================================
# Merge archive results into shard top-K
# ============================================================

def merge_archive_results(
    results,
    top_k,
):
    """
    Merge all archives belonging to THIS LSF array element
    into one exact shard top-K.
    """

    shard_scores = np.empty(
        0,
        dtype=np.float64,
    )

    shard_lines = np.empty(
        0,
        dtype=np.uint32,
    )

    shard_archives = np.empty(
        0,
        dtype=np.uint32,
    )

    for counter, result in enumerate(
        results,
        start=1,
    ):

        with np.load(result["result"]) as data:

            scores = np.asarray(
                data["scores"],
                dtype=np.float64,
            )

            lines = np.asarray(
                data["lines"],
                dtype=np.uint32,
            )

        archive_indices = np.full(
            scores.size,
            result["archive_index"],
            dtype=np.uint32,
        )

        # ----------------------------------------------------
        # Filter against current shard threshold where possible.
        # ----------------------------------------------------

        if shard_scores.size >= top_k:

            threshold = shard_scores.min()

            mask = scores >= threshold

            if not np.any(mask):

                print(
                    f"[SHARD MERGE] "
                    f"{counter}/{len(results)} "
                    f"no competitive candidates",
                    flush=True,
                )

                continue

            scores = scores[mask]
            lines = lines[mask]
            archive_indices = archive_indices[mask]

        # ----------------------------------------------------
        # Combine candidates.
        # ----------------------------------------------------

        shard_scores = np.concatenate(
            (
                shard_scores,
                scores,
            )
        )

        shard_lines = np.concatenate(
            (
                shard_lines,
                lines,
            )
        )

        shard_archives = np.concatenate(
            (
                shard_archives,
                archive_indices,
            )
        )

        # ----------------------------------------------------
        # Reduce to top K.
        # ----------------------------------------------------

        if shard_scores.size > top_k:

            n = shard_scores.size

            idx = np.argpartition(
                shard_scores,
                n - top_k,
            )[n - top_k:]

            shard_scores = shard_scores[idx]
            shard_lines = shard_lines[idx]
            shard_archives = shard_archives[idx]

        print(
            f"[SHARD MERGE] "
            f"{counter}/{len(results)} "
            f"retained={shard_scores.size:,}",
            flush=True,
        )

    # --------------------------------------------------------
    # Final descending ordering
    # --------------------------------------------------------

    order = np.argsort(
        shard_scores,
        kind="stable",
    )[::-1]

    return (
        shard_scores[order],
        shard_lines[order],
        shard_archives[order],
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Distributed Enamine Morgan/Tanimoto similarity "
            "search worker."
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help=(
            "Text file containing one absolute "
            "*.cxsmiles.bz2 path per line."
        ),
    )

    parser.add_argument(
        "--query-smiles",
        required=True,
        help="Reference/query SMILES.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=1_000_000,
        help="Number of shard hits to retain.",
    )

    parser.add_argument(
        "--task-count",
        type=int,
        required=True,
        help="Total number of distributed array tasks.",
    )

    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help=(
            "1-based distributed task index. "
            "If omitted, uses LSB_JOBINDEX."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help=(
            "Number of archives processed simultaneously "
            "inside this LSF task."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Molecules fingerprinted per RDKit batch.",
    )

    parser.add_argument(
        "--header-lines",
        type=int,
        default=1,
        help="Header lines in each cxsmiles archive.",
    )

    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius.",
    )

    parser.add_argument(
        "--fp-size",
        type=int,
        default=2048,
        help="Morgan fingerprint bit count.",
    )

    parser.add_argument(
        "--use-chirality",
        action="store_true",
        help="Include chirality in Morgan fingerprint.",
    )

    parser.add_argument(
        "--work-dir",
        required=True,
        help=(
            "Temporary working directory. Node-local scratch "
            "is recommended."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Persistent directory where shard NPZ/JSON "
            "results are written."
        ),
    )

    parser.add_argument(
        "--keep-archive-results",
        action="store_true",
        help=(
            "Do not delete temporary per-archive NPZ files "
            "after successful shard creation."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Resolve task index.
    # --------------------------------------------------------

    if args.task_index is None:

        lsf_index = os.environ.get(
            "LSB_JOBINDEX"
        )

        if lsf_index is None:
            raise RuntimeError(
                "--task-index was not supplied and "
                "LSB_JOBINDEX is not defined."
            )

        task_index = int(lsf_index)

    else:
        task_index = args.task_index

    # --------------------------------------------------------
    # Read manifest.
    # --------------------------------------------------------

    manifest_path = Path(
        args.manifest
    ).resolve()

    with open(manifest_path, "r") as handle:

        archives = [
            line.strip()
            for line in handle
            if line.strip()
        ]

    if not archives:
        raise RuntimeError(
            f"Manifest is empty: {manifest_path}"
        )

    manifest_hash = manifest_sha256(
        archives
    )

    # --------------------------------------------------------
    # Determine this task's slice.
    # --------------------------------------------------------

    start, end = get_task_slice(
        len(archives),
        task_index,
        args.task_count,
    )

    selected = archives[start:end]

    if not selected:
        raise RuntimeError(
            f"Task {task_index}/{args.task_count} "
            f"received no archives."
        )

    print(
        "============================================================",
        flush=True,
    )
    print(
        f"Distributed similarity search task "
        f"{task_index}/{args.task_count}",
        flush=True,
    )
    print(
        f"Manifest archives: {len(archives):,}",
        flush=True,
    )
    print(
        f"This task:        {len(selected):,}",
        flush=True,
    )
    print(
        f"Global indexes:   {start} through {end - 1}",
        flush=True,
    )
    print(
        f"Workers:          {args.workers}",
        flush=True,
    )
    print(
        f"Top K:            {args.top_k:,}",
        flush=True,
    )
    print(
        f"Batch size:       {args.batch_size:,}",
        flush=True,
    )
    print(
        "============================================================",
        flush=True,
    )

    # --------------------------------------------------------
    # Task-specific scratch directory.
    # --------------------------------------------------------

    work_root = (
        Path(args.work_dir).resolve()
        / f"task_{task_index:05d}"
    )

    archive_work = (
        work_root
        / "archive_results"
    )

    archive_work.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dir = Path(
        args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Prepare archive tasks.
    # --------------------------------------------------------

    tasks = []

    for global_index in range(
        start,
        end,
    ):

        archive_path = archives[
            global_index
        ]

        result_path = (
            archive_work
            / f"archive_{global_index:08d}.npz"
        )

        tasks.append(
            (
                archive_path,
                global_index,
                str(result_path),
            )
        )

    # --------------------------------------------------------
    # Parallel archive processing
    # --------------------------------------------------------

    completed = []

    overall_start = time.time()

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(
            args.query_smiles,
            args.top_k,
            args.batch_size,
            args.header_lines,
            args.radius,
            args.fp_size,
            args.use_chirality,
        ),
    ) as executor:

        future_map = {
            executor.submit(
                process_archive,
                task,
            ): task
            for task in tasks
        }

        for future in as_completed(
            future_map
        ):

            task = future_map[future]

            try:
                result = future.result()

            except Exception as exc:

                print(
                    f"[ERROR] Archive failed:\n"
                    f"  {task[0]}\n"
                    f"  {exc}",
                    file=sys.stderr,
                    flush=True,
                )

                raise

            completed.append(result)

    # --------------------------------------------------------
    # Merge task-local results.
    # --------------------------------------------------------

    completed.sort(
        key=lambda x: x[
            "archive_index"
        ]
    )

    print(
        "[INFO] All archives complete. "
        "Building shard top-K...",
        flush=True,
    )

    (
        shard_scores,
        shard_lines,
        shard_archives,
    ) = merge_archive_results(
        completed,
        args.top_k,
    )

    # --------------------------------------------------------
    # Write shard NPZ.
    # --------------------------------------------------------

    shard_path = (
        output_dir
        / f"shard_{task_index:05d}.npz"
    )

    shard_temp = Path(
        str(shard_path) + ".tmp.npz"
    )

    np.savez(
        shard_temp,
        scores=shard_scores,
        lines=shard_lines,
        archives=shard_archives,
    )

    os.replace(
        shard_temp,
        shard_path,
    )

    # --------------------------------------------------------
    # Metadata allows the final merger to catch accidental
    # mixing of different searches.
    # --------------------------------------------------------

    metadata = {
        "task_index": task_index,
        "task_count": args.task_count,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "manifest_archive_count": len(archives),
        "slice_start": start,
        "slice_end": end,
        "archive_count": len(selected),
        "query_smiles": args.query_smiles,
        "top_k": args.top_k,
        "radius": args.radius,
        "fp_size": args.fp_size,
        "use_chirality": args.use_chirality,
        "batch_size": args.batch_size,
        "header_lines": args.header_lines,
        "workers": args.workers,
        "retained": int(
            shard_scores.size
        ),
        "elapsed_seconds": (
            time.time()
            - overall_start
        ),
    }

    metadata_path = (
        output_dir
        / f"shard_{task_index:05d}.json"
    )

    metadata_temp = Path(
        str(metadata_path) + ".tmp"
    )

    with open(
        metadata_temp,
        "w",
    ) as handle:

        json.dump(
            metadata,
            handle,
            indent=2,
        )

    os.replace(
        metadata_temp,
        metadata_path,
    )

    # --------------------------------------------------------
    # Cleanup scratch archive results.
    # --------------------------------------------------------

    if not args.keep_archive_results:

        shutil.rmtree(
            archive_work,
            ignore_errors=True,
        )

    elapsed = (
        time.time()
        - overall_start
    )

    print(
        "============================================================",
        flush=True,
    )
    print(
        f"[SUCCESS] Task "
        f"{task_index}/{args.task_count} complete",
        flush=True,
    )
    print(
        f"Shard result: {shard_path}",
        flush=True,
    )
    print(
        f"Retained:     {shard_scores.size:,}",
        flush=True,
    )
    print(
        f"Elapsed:      {elapsed / 3600:.2f} hours",
        flush=True,
    )
    print(
        "============================================================",
        flush=True,
    )


if __name__ == "__main__":
    main()
