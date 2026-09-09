#!/usr/bin/env python3

import argparse
import bz2
import json
import os
import sys
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
SMILES_COLUMN = None


def init_worker(
    query_smiles,
    top_k,
    batch_size,
    header_lines,
    smiles_column,
    radius,
    fp_size,
):
    """
    Initialize objects once in each worker process.
    """

    global QUERY_FP
    global FP_GENERATOR
    global TOP_K
    global BATCH_SIZE
    global HEADER_LINES
    global SMILES_COLUMN

    mol = Chem.MolFromSmiles(query_smiles)

    if mol is None:
        raise ValueError(f"Could not parse query SMILES: {query_smiles}")

    FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=fp_size,
    )

    QUERY_FP = FP_GENERATOR.GetFingerprint(mol)

    TOP_K = top_k
    BATCH_SIZE = batch_size
    HEADER_LINES = header_lines
    SMILES_COLUMN = smiles_column


# ============================================================
# Utility functions
# ============================================================

def extract_smiles(line):
    """
    Extract SMILES from one CXSMILES line.

    Default behavior assumes whitespace-separated columns.

    SMILES_COLUMN is zero-based.
    """

    fields = line.rstrip("\n").split()

    if len(fields) <= SMILES_COLUMN:
        return None

    return fields[SMILES_COLUMN]


def reduce_to_top_k(scores, lines, k):
    """
    Reduce arrays to the largest k scores.

    Uses np.argpartition rather than sorting the entire array.
    """

    n = scores.size

    if n <= k:
        return scores, lines

    idx = np.argpartition(scores, n - k)[n - k:]

    return scores[idx], lines[idx]


def update_top_k(
    top_scores,
    top_lines,
    new_scores,
    new_lines,
    k,
):
    """
    Merge a batch of candidate similarities with the current
    top-K arrays.
    """

    if new_scores.size == 0:
        return top_scores, top_lines

    # --------------------------------------------------------
    # Initial filling
    # --------------------------------------------------------

    if top_scores.size < k:

        scores = np.concatenate((top_scores, new_scores))
        lines = np.concatenate((top_lines, new_lines))

        return reduce_to_top_k(scores, lines, k)

    # --------------------------------------------------------
    # Once K entries exist, discard anything that cannot beat
    # the current minimum.
    # --------------------------------------------------------

    threshold = top_scores.min()

    mask = new_scores > threshold

    if not np.any(mask):
        return top_scores, top_lines

    filtered_scores = new_scores[mask]
    filtered_lines = new_lines[mask]

    scores = np.concatenate((top_scores, filtered_scores))
    lines = np.concatenate((top_lines, filtered_lines))

    return reduce_to_top_k(scores, lines, k)


# ============================================================
# Batch processing
# ============================================================

def process_batch(
    smiles_batch,
    line_batch,
    top_scores,
    top_lines,
):
    """
    Fingerprint a batch and calculate Tanimoto similarities.
    """

    fps = []
    valid_lines = []

    for smiles, line_number in zip(smiles_batch, line_batch):

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        fp = FP_GENERATOR.GetFingerprint(mol)

        fps.append(fp)
        valid_lines.append(line_number)

    if not fps:
        return top_scores, top_lines, 0

    similarities = DataStructs.BulkTanimotoSimilarity(
        QUERY_FP,
        fps,
    )

    scores = np.asarray(similarities, dtype=np.float32)

    lines = np.asarray(
        valid_lines,
        dtype=np.uint64,
    )

    top_scores, top_lines = update_top_k(
        top_scores,
        top_lines,
        scores,
        lines,
        TOP_K,
    )

    return top_scores, top_lines, len(fps)


# ============================================================
# Archive worker
# ============================================================

def process_archive(task):
    """
    Process one *.cxsmiles.bz2 archive.

    Returns path to a local NPZ file containing this archive's
    top-K results.
    """

    archive_path, result_path = task

    archive_path = Path(archive_path)
    result_path = Path(result_path)

    top_scores = np.empty(0, dtype=np.float32)
    top_lines = np.empty(0, dtype=np.uint64)

    smiles_batch = []
    line_batch = []

    total_records = 0
    valid_records = 0

    print(
        f"[START] {archive_path}",
        flush=True,
    )

    with bz2.open(
        archive_path,
        mode="rt",
        errors="replace",
    ) as handle:

        # Skip header
        for _ in range(HEADER_LINES):
            next(handle, None)

        for ligand_line_number, line in enumerate(handle):

            total_records += 1

            smiles = extract_smiles(line)

            if smiles is None:
                continue

            smiles_batch.append(smiles)
            line_batch.append(ligand_line_number)

            if len(smiles_batch) >= BATCH_SIZE:

                top_scores, top_lines, n_valid = process_batch(
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
                        f"[PROGRESS] {archive_path.name}: "
                        f"{total_records:,} records",
                        flush=True,
                    )

        # Final partial batch
        if smiles_batch:

            top_scores, top_lines, n_valid = process_batch(
                smiles_batch,
                line_batch,
                top_scores,
                top_lines,
            )

            valid_records += n_valid

    # --------------------------------------------------------
    # Sort this archive's hits descending before saving
    # --------------------------------------------------------

    order = np.argsort(top_scores)[::-1]

    top_scores = top_scores[order]
    top_lines = top_lines[order]

    np.savez(
        result_path,
        scores=top_scores,
        lines=top_lines,
    )

    print(
        f"[DONE] {archive_path.name}: "
        f"{total_records:,} records, "
        f"{valid_records:,} valid molecules, "
        f"{top_scores.size:,} retained",
        flush=True,
    )

    return {
        "archive": str(archive_path),
        "result": str(result_path),
        "records": total_records,
        "valid": valid_records,
        "retained": int(top_scores.size),
    }


# ============================================================
# Global merge
# ============================================================

def merge_global_results(
    results,
    top_k,
):
    """
    Merge per-archive top-K results into an exact global top-K.

    Returns:
        global_scores
        global_lines
        global_files

    global_files stores an integer index corresponding to the
    'results' list.
    """

    global_scores = np.empty(
        0,
        dtype=np.float32,
    )

    global_lines = np.empty(
        0,
        dtype=np.uint64,
    )

    global_files = np.empty(
        0,
        dtype=np.uint32,
    )

    for file_index, result in enumerate(results):

        data = np.load(
            result["result"],
            mmap_mode="r",
        )

        scores = np.asarray(
            data["scores"],
            dtype=np.float32,
        )

        lines = np.asarray(
            data["lines"],
            dtype=np.uint64,
        )

        if global_scores.size >= top_k:

            threshold = global_scores.min()

            mask = scores > threshold

            if not np.any(mask):
                continue

            scores = scores[mask]
            lines = lines[mask]

        files = np.full(
            scores.size,
            file_index,
            dtype=np.uint32,
        )

        global_scores = np.concatenate(
            (global_scores, scores)
        )

        global_lines = np.concatenate(
            (global_lines, lines)
        )

        global_files = np.concatenate(
            (global_files, files)
        )

        if global_scores.size > top_k:

            n = global_scores.size

            idx = np.argpartition(
                global_scores,
                n - top_k,
            )[n - top_k:]

            global_scores = global_scores[idx]
            global_lines = global_lines[idx]
            global_files = global_files[idx]

        print(
            f"[MERGE] {file_index + 1}/{len(results)} "
            f"global candidates={global_scores.size:,}",
            flush=True,
        )

    # Final descending sort
    order = np.argsort(global_scores)[::-1]

    return (
        global_scores[order],
        global_lines[order],
        global_files[order],
    )


# ============================================================
# Write final results
# ============================================================

def write_results(
    output_file,
    scores,
    lines,
    files,
    results,
):
    """
    Write global hits.

    Ligand line numbers are zero-based AFTER the header.
    """

    with open(output_file, "w") as handle:

        handle.write(
            "rank\t"
            "tanimoto\t"
            "source_file\t"
            "ligand_line\n"
        )

        for rank, (score, line_number, file_index) in enumerate(
            zip(scores, lines, files),
            start=1,
        ):

            archive = results[int(file_index)]["archive"]

            handle.write(
                f"{rank}\t"
                f"{float(score):.6f}\t"
                f"{archive}\t"
                f"{int(line_number)}\n"
            )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Exhaustive Morgan/Tanimoto similarity search "
            "across split Enamine CXSMILES bz2 archives."
        )
    )

    parser.add_argument(
        "root",
        help=(
            "Root directory containing *.cxsmiles.bz2 "
            "archives. Search is recursive."
        ),
    )

    parser.add_argument(
        "query_smiles",
        help="Query SMILES",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=1_000_000,
        help="Number of global hits to retain",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of archives processed concurrently",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Molecules fingerprinted per batch",
    )

    parser.add_argument(
        "--header-lines",
        type=int,
        default=1,
        help="Number of header lines in each file",
    )

    parser.add_argument(
        "--smiles-column",
        type=int,
        default=0,
        help="Zero-based SMILES column",
    )

    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius",
    )

    parser.add_argument(
        "--fp-size",
        type=int,
        default=2048,
        help="Fingerprint size",
    )

    parser.add_argument(
        "--work-dir",
        default="similarity_work",
        help="Temporary per-archive result directory",
    )

    parser.add_argument(
        "--output",
        default="top_similarity_hits.tsv",
        help="Final result TSV",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Discover archives
    # --------------------------------------------------------

    archives = sorted(
        root.rglob("*.cxsmiles.bz2")
    )

    if not archives:
        print(
            f"No *.cxsmiles.bz2 files found beneath {root}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"[INFO] Found {len(archives):,} archives",
        flush=True,
    )

    print(
        f"[INFO] Query: {args.query_smiles}",
        flush=True,
    )

    print(
        f"[INFO] Requested top K: {args.top_k:,}",
        flush=True,
    )

    print(
        f"[INFO] Workers: {args.workers}",
        flush=True,
    )

    # --------------------------------------------------------
    # Generate tasks
    # --------------------------------------------------------

    tasks = []

    for i, archive in enumerate(archives):

        result_path = work_dir / f"archive_{i:08d}.npz"

        tasks.append(
            (
                str(archive),
                str(result_path),
            )
        )

    # --------------------------------------------------------
    # Search archives in parallel
    # --------------------------------------------------------

    completed = []

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(
            args.query_smiles,
            args.top_k,
            args.batch_size,
            args.header_lines,
            args.smiles_column,
            args.radius,
            args.fp_size,
        ),
    ) as executor:

        future_map = {
            executor.submit(
                process_archive,
                task,
            ): task
            for task in tasks
        }

        for future in as_completed(future_map):

            task = future_map[future]

            try:
                result = future.result()

            except Exception as exc:

                print(
                    f"[ERROR] Failed archive: "
                    f"{task[0]}\n{exc}",
                    file=sys.stderr,
                    flush=True,
                )

                raise

            completed.append(result)

    # --------------------------------------------------------
    # Restore archive order
    # --------------------------------------------------------

    completed.sort(
        key=lambda x: x["archive"]
    )

    # --------------------------------------------------------
    # Global merge
    # --------------------------------------------------------

    print(
        "[INFO] Beginning global top-K merge",
        flush=True,
    )

    scores, lines, files = merge_global_results(
        completed,
        args.top_k,
    )

    # --------------------------------------------------------
    # Write final locator table
    # --------------------------------------------------------

    write_results(
        args.output,
        scores,
        lines,
        files,
        completed,
    )

    print(
        f"[SUCCESS] Wrote {scores.size:,} hits to "
        f"{args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
