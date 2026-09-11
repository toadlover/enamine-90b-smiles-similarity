#!/usr/bin/env python3

"""
merge_enamine_similarity_results.py

Combine distributed Enamine similarity-search shard results
into an exact global top-K.

Outputs:

1. Locator TSV:
       rank
       tanimoto
       source_file
       ligand_line

2. Pipeline-compatible CSV:
       SMILES
       ligand_name
       source_bz2
       tanimoto

The CSV is ordered from highest to lowest Tanimoto similarity.

CXSMILES format handling
------------------------

The 90B input format is:

smiles id MW HAC sLogP HBA HBD RotBonds FSP3 TPSA Type InChIKey

Some records have CXSMILES annotation between SMILES and ID:

SMILES |&1:11,14| id MW ...

There are always 10 metadata fields AFTER id:

    MW
    HAC
    sLogP
    HBA
    HBD
    RotBonds
    FSP3
    TPSA
    Type
    InChIKey

Therefore:

    fields[-11]

is the ligand ID regardless of the number of CXSMILES annotation
tokens appearing before it.
"""

import argparse
import bz2
import csv
import heapq
import json
import os
import shutil
import sys

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# ============================================================
# Manifest
# ============================================================

def read_manifest(path):

    with open(path, "r") as handle:

        return [
            line.strip()
            for line in handle
            if line.strip()
        ]


# ============================================================
# Global top-K merge
# ============================================================

def merge_shards(
    shard_paths,
    top_k,
):
    """
    Merge every distributed shard into one exact global top-K.
    """

    global_scores = np.empty(
        0,
        dtype=np.float64,
    )

    global_lines = np.empty(
        0,
        dtype=np.uint32,
    )

    global_archives = np.empty(
        0,
        dtype=np.uint32,
    )

    for counter, shard_path in enumerate(
        shard_paths,
        start=1,
    ):

        with np.load(
            shard_path
        ) as data:

            scores = np.asarray(
                data["scores"],
                dtype=np.float64,
            )

            lines = np.asarray(
                data["lines"],
                dtype=np.uint32,
            )

            archives = np.asarray(
                data["archives"],
                dtype=np.uint32,
            )

        # ----------------------------------------------------
        # Ignore shard candidates definitely below current
        # global threshold.
        # ----------------------------------------------------

        if global_scores.size >= top_k:

            threshold = (
                global_scores.min()
            )

            mask = (
                scores >= threshold
            )

            if not np.any(mask):

                print(
                    f"[GLOBAL MERGE] "
                    f"{counter}/{len(shard_paths)} "
                    f"no competitive candidates",
                    flush=True,
                )

                continue

            scores = scores[mask]
            lines = lines[mask]
            archives = archives[mask]

        global_scores = np.concatenate(
            (
                global_scores,
                scores,
            )
        )

        global_lines = np.concatenate(
            (
                global_lines,
                lines,
            )
        )

        global_archives = np.concatenate(
            (
                global_archives,
                archives,
            )
        )

        if global_scores.size > top_k:

            n = global_scores.size

            idx = np.argpartition(
                global_scores,
                n - top_k,
            )[n - top_k:]

            global_scores = (
                global_scores[idx]
            )

            global_lines = (
                global_lines[idx]
            )

            global_archives = (
                global_archives[idx]
            )

        print(
            f"[GLOBAL MERGE] "
            f"{counter}/{len(shard_paths)} "
            f"retained={global_scores.size:,}",
            flush=True,
        )

    # --------------------------------------------------------
    # Sort final global result descending.
    # --------------------------------------------------------

    order = np.argsort(
        global_scores,
        kind="stable",
    )[::-1]

    return (
        global_scores[order],
        global_lines[order],
        global_archives[order],
    )


# ============================================================
# Compact locator output
# ============================================================

def write_locator_tsv(
    output_file,
    scores,
    lines,
    archives,
    manifest,
):
    """
    Write compact top-K locator output.
    """

    with open(
        output_file,
        "w",
    ) as handle:

        handle.write(
            "rank\t"
            "tanimoto\t"
            "source_file\t"
            "ligand_line\n"
        )

        for rank, (
            score,
            line_number,
            archive_index,
        ) in enumerate(
            zip(
                scores,
                lines,
                archives,
            ),
            start=1,
        ):

            source = manifest[
                int(archive_index)
            ]

            handle.write(
                f"{rank}\t"
                f"{float(score):.10f}\t"
                f"{source}\t"
                f"{int(line_number)}\n"
            )


# ============================================================
# CXSMILES record parser
# ============================================================

def parse_enamine_record(line):
    """
    Parse one 90B Enamine cxsmiles record.

    The base SMILES is always first.

    The ligand ID is determined FROM THE RIGHT because
    CXSMILES annotations may occur between SMILES and ID.

    Expected trailing layout:

        id
        MW
        HAC
        sLogP
        HBA
        HBD
        RotBonds
        FSP3
        TPSA
        Type
        InChIKey

    Therefore id = fields[-11].
    """

    fields = line.strip().split()

    if len(fields) < 12:

        raise ValueError(
            "Record contains fewer than 12 "
            "whitespace-separated fields."
        )

    smiles = fields[0]

    ligand_id = fields[-11]

    return smiles, ligand_id


# ============================================================
# Recovery-worker function
# ============================================================

def recover_archive(task):
    """
    Re-open one selected bz2 archive and recover the SMILES/ID
    for all global top-K records originating from it.

    Writes a temporary CSV sorted by global rank.

    task:
        archive_path
        selections
        header_lines
        temp_output

    selections:
        list of:
            (rank, ligand_line, score)
    """

    (
        archive_path,
        selections,
        header_lines,
        temp_output,
    ) = task

    archive_path = str(
        archive_path
    )

    temp_output = Path(
        temp_output
    )

    # --------------------------------------------------------
    # line -> one or more (rank, score)
    # --------------------------------------------------------

    needed = defaultdict(list)

    for (
        rank,
        ligand_line,
        score,
    ) in selections:

        needed[
            int(ligand_line)
        ].append(
            (
                int(rank),
                float(score),
            )
        )

    max_needed_line = max(
        needed
    )

    recovered = []

    found_line_count = 0

    print(
        f"[RECOVER START] "
        f"{Path(archive_path).name}: "
        f"{len(needed):,} unique selected lines",
        flush=True,
    )

    with bz2.open(
        archive_path,
        mode="rt",
        errors="replace",
    ) as handle:

        # ----------------------------------------------------
        # Skip header.
        # ----------------------------------------------------

        for _ in range(
            header_lines
        ):
            next(handle, None)

        # ----------------------------------------------------
        # Search requested records.
        # ----------------------------------------------------

        for ligand_line, line in enumerate(
            handle
        ):

            if ligand_line > max_needed_line:
                break

            if ligand_line not in needed:
                continue

            try:

                (
                    smiles,
                    ligand_id,
                ) = parse_enamine_record(
                    line
                )

            except Exception as exc:

                raise RuntimeError(
                    f"Failed parsing "
                    f"{archive_path} "
                    f"ligand line "
                    f"{ligand_line}: "
                    f"{exc}\n"
                    f"Record:\n{line}"
                )

            for rank, score in needed[
                ligand_line
            ]:

                recovered.append(
                    (
                        rank,
                        smiles,
                        ligand_id,
                        archive_path,
                        score,
                    )
                )

            found_line_count += 1

    if found_line_count != len(
        needed
    ):

        raise RuntimeError(
            f"Recovery failure for "
            f"{archive_path}: expected "
            f"{len(needed):,} unique "
            f"lines but recovered "
            f"{found_line_count:,}."
        )

    # --------------------------------------------------------
    # Keep each temporary file rank-sorted so the parent can
    # perform a memory-light k-way merge.
    # --------------------------------------------------------

    recovered.sort(
        key=lambda x: x[0]
    )

    temp_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        temp_output,
        "w",
        newline="",
    ) as handle:

        writer = csv.writer(
            handle
        )

        for row in recovered:
            writer.writerow(row)

    print(
        f"[RECOVER DONE] "
        f"{Path(archive_path).name}: "
        f"{len(recovered):,} records",
        flush=True,
    )

    return str(temp_output)


# ============================================================
# K-way merge recovered rank files
# ============================================================

def merge_recovered_csvs(
    temp_files,
    output_csv,
    include_header=False,
):
    """
    Merge per-archive recovery CSVs by global rank.

    Temporary row layout:

        rank,smiles,id,source,score

    Final requested layout:

        smiles,id,source,score
    """

    handles = []
    readers = []

    heap = []

    try:

        # ----------------------------------------------------
        # Initialize one row from each file.
        # ----------------------------------------------------

        for file_index, path in enumerate(
            temp_files
        ):

            handle = open(
                path,
                "r",
                newline="",
            )

            reader = csv.reader(
                handle
            )

            handles.append(handle)
            readers.append(reader)

            try:
                row = next(reader)

            except StopIteration:
                continue

            rank = int(row[0])

            heapq.heappush(
                heap,
                (
                    rank,
                    file_index,
                    row,
                )
            )

        # ----------------------------------------------------
        # Global rank merge.
        # ----------------------------------------------------

        with open(
            output_csv,
            "w",
            newline="",
        ) as output_handle:

            writer = csv.writer(
                output_handle
            )

            if include_header:

                writer.writerow(
                    [
                        "smiles",
                        "ligand_name",
                        "source_bz2",
                        "tanimoto",
                    ]
                )

            while heap:

                (
                    rank,
                    file_index,
                    row,
                ) = heapq.heappop(
                    heap
                )

                # Temporary:
                # rank,smiles,id,source,score
                #
                # Requested:
                # smiles,id,source,score

                writer.writerow(
                    [
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                    ]
                )

                try:

                    next_row = next(
                        readers[
                            file_index
                        ]
                    )

                except StopIteration:
                    continue

                next_rank = int(
                    next_row[0]
                )

                heapq.heappush(
                    heap,
                    (
                        next_rank,
                        file_index,
                        next_row,
                    )
                )

    finally:

        for handle in handles:
            handle.close()


# ============================================================
# Metadata verification
# ============================================================

def load_and_verify_metadata(
    result_dir,
    task_count,
):
    """
    Verify that:
        * every expected task exists
        * all tasks used the same query
        * all tasks used the same fingerprint settings
        * all tasks used the same manifest
        * all tasks used the same K
    """

    metadata = []

    for task_index in range(
        1,
        task_count + 1,
    ):

        metadata_path = (
            result_dir
            / f"shard_{task_index:05d}.json"
        )

        shard_path = (
            result_dir
            / f"shard_{task_index:05d}.npz"
        )

        if not metadata_path.exists():

            raise FileNotFoundError(
                f"Missing metadata: "
                f"{metadata_path}"
            )

        if not shard_path.exists():

            raise FileNotFoundError(
                f"Missing shard: "
                f"{shard_path}"
            )

        with open(
            metadata_path,
            "r",
        ) as handle:

            info = json.load(
                handle
            )

        metadata.append(
            info
        )

    reference = metadata[0]

    keys_to_match = [
        "task_count",
        "manifest_sha256",
        "manifest_archive_count",
        "query_smiles",
        "top_k",
        "min_tanimoto",
        "radius",
        "fp_size",
        "use_chirality",
        "header_lines",
    ]

    for info in metadata[1:]:

        for key in keys_to_match:

            if info[key] != reference[key]:

                raise RuntimeError(
                    f"Shard metadata mismatch "
                    f"for '{key}':\n"
                    f"reference={reference[key]}\n"
                    f"other={info[key]}"
                )

    return reference


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Merge distributed Enamine similarity "
            "search shards."
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help=(
            "Same ordered cxsmiles manifest used "
            "by worker jobs."
        ),
    )

    parser.add_argument(
        "--result-dir",
        required=True,
        help=(
            "Directory containing shard_XXXXX.npz "
            "and shard_XXXXX.json files."
        ),
    )

    parser.add_argument(
        "--task-count",
        type=int,
        required=True,
        help="Total number of distributed tasks.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Final top K. Defaults to worker top-K. "
            "Must not exceed worker top-K."
        ),
    )

    parser.add_argument(
        "--locator-output",
        default="top_similarity_hits.tsv",
        help="Compact locator TSV output.",
    )

    parser.add_argument(
        "--pipeline-output",
        default="top_similarity_hits_pipeline.csv",
        help=(
            "Pipeline CSV output: "
            "SMILES,ID,source_bz2,Tanimoto"
        ),
    )

    parser.add_argument(
        "--recovery-work-dir",
        required=True,
        help=(
            "Temporary directory for recovered "
            "per-archive CSV fragments."
        ),
    )

    parser.add_argument(
        "--recovery-workers",
        type=int,
        default=8,
        help=(
            "Number of bz2 archives decompressed "
            "simultaneously during SMILES/ID recovery."
        ),
    )

    parser.add_argument(
        "--csv-header",
        action="store_true",
        help=(
            "Include column names in pipeline CSV. "
            "Default is headerless for compatibility "
            "with your existing downstream scripts."
        ),
    )

    parser.add_argument(
        "--keep-recovery-files",
        action="store_true",
        help="Keep temporary recovered rank CSV files.",
    )

    args = parser.parse_args()

    result_dir = Path(
        args.result_dir
    ).resolve()

    recovery_work_dir = Path(
        args.recovery_work_dir
    ).resolve()

    recovery_work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = read_manifest(
        args.manifest
    )

    # --------------------------------------------------------
    # Verify distributed search.
    # --------------------------------------------------------

    metadata = load_and_verify_metadata(
        result_dir,
        args.task_count,
    )

    worker_top_k = int(
        metadata["top_k"]
    )

    if args.top_k is None:

        final_top_k = (
            worker_top_k
        )

    else:

        final_top_k = (
            args.top_k
        )

        if final_top_k > worker_top_k:

            raise ValueError(
                f"Requested final top-K "
                f"({final_top_k:,}) exceeds "
                f"the top-K retained by each "
                f"worker shard "
                f"({worker_top_k:,})."
            )

    if len(manifest) != int(
        metadata[
            "manifest_archive_count"
        ]
    ):

        raise RuntimeError(
            "Manifest archive count does not "
            "match shard metadata."
        )

    print(
        "============================================================",
        flush=True,
    )
    print(
        "Merging distributed Enamine similarity search",
        flush=True,
    )
    print(
        f"Query:             "
        f"{metadata['query_smiles']}",
        flush=True,
    )
    print(
        f"Shards:            "
        f"{args.task_count}",
        flush=True,
    )
    print(
        f"Archives:          "
        f"{len(manifest):,}",
        flush=True,
    )
    print(
        f"Worker top-K:      "
        f"{worker_top_k:,}",
        flush=True,
    )
    print(
        f"Final top-K:       "
        f"{final_top_k:,}",
        flush=True,
    )
    print(
        f"Fingerprint:       "
        f"Morgan radius={metadata['radius']} "
        f"bits={metadata['fp_size']} "
        f"chirality={metadata['use_chirality']}",
        flush=True,
    )
    print(
        "============================================================",
        flush=True,
    )

    # --------------------------------------------------------
    # Identify shard NPZs.
    # --------------------------------------------------------

    shard_paths = [
        result_dir
        / f"shard_{i:05d}.npz"
        for i in range(
            1,
            args.task_count + 1,
        )
    ]

    # --------------------------------------------------------
    # Exact global merge.
    # --------------------------------------------------------

    (
        scores,
        lines,
        archives,
    ) = merge_shards(
        shard_paths,
        final_top_k,
    )

    print(
        f"[SUCCESS] Global top-K contains "
        f"{scores.size:,} ligands.",
        flush=True,
    )

    # --------------------------------------------------------
    # Compact locator TSV.
    # --------------------------------------------------------

    write_locator_tsv(
        args.locator_output,
        scores,
        lines,
        archives,
        manifest,
    )

    print(
        f"[SUCCESS] Locator TSV: "
        f"{args.locator_output}",
        flush=True,
    )

    # --------------------------------------------------------
    # Group global selected results by archive.
    #
    # Rank is zero-based internally here.
    # --------------------------------------------------------

    selections_by_archive = (
        defaultdict(list)
    )

    for rank, (
        score,
        line_number,
        archive_index,
    ) in enumerate(
        zip(
            scores,
            lines,
            archives,
        )
    ):

        selections_by_archive[
            int(archive_index)
        ].append(
            (
                rank,
                int(line_number),
                float(score),
            )
        )

    print(
        f"[INFO] Final top-K is represented "
        f"in {len(selections_by_archive):,} "
        f"source archives.",
        flush=True,
    )

    # --------------------------------------------------------
    # Prepare recovery tasks.
    # --------------------------------------------------------

    recovery_tasks = []

    for archive_index, selections in (
        selections_by_archive.items()
    ):

        archive_path = manifest[
            archive_index
        ]

        temp_output = (
            recovery_work_dir
            / f"archive_{archive_index:08d}.csv"
        )

        recovery_tasks.append(
            (
                archive_path,
                selections,
                int(
                    metadata[
                        "header_lines"
                    ]
                ),
                str(temp_output),
            )
        )

    # --------------------------------------------------------
    # Parallel SMILES/ID recovery.
    # --------------------------------------------------------

    recovered_files = []

    with ProcessPoolExecutor(
        max_workers=args.recovery_workers
    ) as executor:

        future_map = {
            executor.submit(
                recover_archive,
                task,
            ): task
            for task in recovery_tasks
        }

        for future in as_completed(
            future_map
        ):

            task = future_map[
                future
            ]

            try:

                recovered_file = (
                    future.result()
                )

            except Exception as exc:

                print(
                    f"[ERROR] Recovery failed "
                    f"for {task[0]}:\n"
                    f"{exc}",
                    file=sys.stderr,
                    flush=True,
                )

                raise

            recovered_files.append(
                recovered_file
            )

    # --------------------------------------------------------
    # K-way rank merge into requested pipeline CSV.
    # --------------------------------------------------------

    merge_recovered_csvs(
        recovered_files,
        args.pipeline_output,
        include_header=args.csv_header,
    )

    print(
        f"[SUCCESS] Pipeline CSV: "
        f"{args.pipeline_output}",
        flush=True,
    )

    # --------------------------------------------------------
    # Cleanup.
    # --------------------------------------------------------

    if not args.keep_recovery_files:

        shutil.rmtree(
            recovery_work_dir,
            ignore_errors=True,
        )

    print(
        "============================================================",
        flush=True,
    )
    print(
        "ALL MERGING AND RECOVERY COMPLETE",
        flush=True,
    )
    print(
        f"Locator output:  "
        f"{args.locator_output}",
        flush=True,
    )
    print(
        f"Pipeline output: "
        f"{args.pipeline_output}",
        flush=True,
    )
    print(
        "============================================================",
        flush=True,
    )


if __name__ == "__main__":
    main()
