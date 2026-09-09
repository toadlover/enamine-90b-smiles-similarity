#!/usr/bin/env python3

import argparse
import bz2
import os
import shutil
import sys
from pathlib import Path


DEFAULT_CHUNK_SIZE = 10_000_000


def format_int(n):
    return f"{n:,}"


def split_bz2_file(
    input_file,
    chunk_size=DEFAULT_CHUNK_SIZE,
    header_lines=1,
    delete_original=False,
    overwrite_inprogress=False,
):
    input_file = Path(input_file).resolve()

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    if not input_file.name.endswith(".cxsmiles.bz2"):
        raise ValueError(
            f"Expected a *.cxsmiles.bz2 file, got: {input_file.name}"
        )

    # Example:
    # ABCD.cxsmiles.bz2 -> ABCD
    base_name = input_file.name[:-len(".cxsmiles.bz2")]

    final_dir = input_file.parent / f"{base_name}_chunks"
    staging_dir = input_file.parent / f"{base_name}_chunks.inprogress"

    # If final directory exists, assume this file was already processed.
    if final_dir.exists():
        raise RuntimeError(
            f"Final output directory already exists:\n"
            f"  {final_dir}\n"
            f"Refusing to overwrite it."
        )

    # Clean up an abandoned staging directory only if explicitly allowed.
    if staging_dir.exists():
        if overwrite_inprogress:
            print(
                f"[INFO] Removing previous incomplete directory: {staging_dir}",
                flush=True,
            )
            shutil.rmtree(staging_dir)
        else:
            raise RuntimeError(
                f"Incomplete staging directory already exists:\n"
                f"  {staging_dir}\n"
                f"Remove it manually or rerun with --overwrite-inprogress."
            )

    staging_dir.mkdir(parents=False)

    total_ligands = 0
    output_ligands = 0
    chunk_index = 0
    current_chunk_count = 0
    output_handle = None
    output_path = None

    headers = []

    def open_output(index):
        output_name = f"{base_name}.part{index:05d}.cxsmiles.bz2"
        path = staging_dir / output_name

        print(f"[INFO] Opening output chunk: {path.name}", flush=True)

        handle = bz2.open(path, mode="wt")

        for header in headers:
            handle.write(header)

        return handle, path

    try:
        # Text mode is appropriate for line-oriented CXSMILES files.
        with bz2.open(input_file, mode="rt") as infile:

            # Read the header(s) once.
            for _ in range(header_lines):
                line = infile.readline()

                if line == "":
                    raise RuntimeError(
                        f"Input ended before all {header_lines} header lines "
                        f"could be read."
                    )

                headers.append(line)

            # Stream ligand records.
            for line in infile:

                # Open first chunk, or a new chunk after reaching chunk_size.
                if output_handle is None:
                    output_handle, output_path = open_output(chunk_index)
                    current_chunk_count = 0

                output_handle.write(line)

                total_ligands += 1
                output_ligands += 1
                current_chunk_count += 1

                if current_chunk_count == chunk_size:
                    output_handle.close()
                    output_handle = None

                    print(
                        f"[INFO] Finished {output_path.name}: "
                        f"{format_int(current_chunk_count)} ligands",
                        flush=True,
                    )

                    chunk_index += 1
                    current_chunk_count = 0

                if total_ligands % 1_000_000 == 0:
                    print(
                        f"[PROGRESS] {format_int(total_ligands)} ligands processed",
                        flush=True,
                    )

        # Close final partial chunk.
        if output_handle is not None:
            output_handle.close()
            output_handle = None

            print(
                f"[INFO] Finished {output_path.name}: "
                f"{format_int(current_chunk_count)} ligands",
                flush=True,
            )

            chunk_index += 1

        # Special case: header-only input.
        if total_ligands == 0:
            raise RuntimeError(
                f"No ligand records found in {input_file}. "
                f"Original file will not be deleted."
            )

        print(
            f"[INFO] Input ligand count:  {format_int(total_ligands)}",
            flush=True,
        )
        print(
            f"[INFO] Output ligand count: {format_int(output_ligands)}",
            flush=True,
        )
        print(
            f"[INFO] Number of chunks:    {chunk_index}",
            flush=True,
        )

        # Internal consistency check.
        if total_ligands != output_ligands:
            raise RuntimeError(
                f"Count mismatch: input={total_ligands}, "
                f"output={output_ligands}"
            )

        #
        # Verification pass
        #
        # Re-open every generated bz2 and verify that:
        #   1. it decompresses successfully
        #   2. it contains the expected number of ligand lines
        #   3. its header matches the original header
        #
        print("[INFO] Starting verification pass...", flush=True)

        verified_ligands = 0
        generated_files = sorted(staging_dir.glob("*.cxsmiles.bz2"))

        if len(generated_files) != chunk_index:
            raise RuntimeError(
                f"Expected {chunk_index} output files but found "
                f"{len(generated_files)}."
            )

        for i, chunk_file in enumerate(generated_files):
            ligand_count = 0

            with bz2.open(chunk_file, mode="rt") as handle:

                # Check headers.
                for header_number, expected_header in enumerate(headers):
                    actual_header = handle.readline()

                    if actual_header != expected_header:
                        raise RuntimeError(
                            f"Header mismatch in {chunk_file}, "
                            f"header line {header_number + 1}"
                        )

                # Count ligand lines.
                for _ in handle:
                    ligand_count += 1

            # Every chunk except possibly the last must have exactly chunk_size.
            if i < len(generated_files) - 1:
                if ligand_count != chunk_size:
                    raise RuntimeError(
                        f"{chunk_file.name} contains "
                        f"{format_int(ligand_count)} ligands; expected "
                        f"{format_int(chunk_size)}."
                    )
            else:
                if ligand_count > chunk_size:
                    raise RuntimeError(
                        f"Final chunk {chunk_file.name} contains "
                        f"{format_int(ligand_count)} ligands, exceeding "
                        f"{format_int(chunk_size)}."
                    )

            verified_ligands += ligand_count

            print(
                f"[VERIFY] {chunk_file.name}: "
                f"{format_int(ligand_count)} ligands OK",
                flush=True,
            )

        if verified_ligands != total_ligands:
            raise RuntimeError(
                f"Verification failed: original contained "
                f"{format_int(total_ligands)} ligands, but output files "
                f"contain {format_int(verified_ligands)}."
            )

        print(
            f"[SUCCESS] Verified all {format_int(verified_ligands)} ligand records.",
            flush=True,
        )

        #
        # Atomically-ish mark processing as complete by renaming the directory.
        #
        staging_dir.rename(final_dir)

        print(f"[SUCCESS] Final output directory: {final_dir}", flush=True)

        #
        # Delete original only after successful validation + directory rename.
        #
        if delete_original:
            input_file.unlink()
            print(
                f"[SUCCESS] Deleted original file: {input_file}",
                flush=True,
            )
        else:
            print(
                f"[INFO] Original retained: {input_file}",
                flush=True,
            )

        return total_ligands, chunk_index

    except Exception:
        # Close output if an exception occurred while writing.
        if output_handle is not None:
            try:
                output_handle.close()
            except Exception:
                pass

        print(
            f"[ERROR] Processing failed for {input_file}.",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[ERROR] Original file has NOT been deleted.",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[ERROR] Partial output, if any, is in: {staging_dir}",
            file=sys.stderr,
            flush=True,
        )
        raise


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split an Enamine .cxsmiles.bz2 file into compressed chunks "
            "containing at most N ligand records, while preserving headers."
        )
    )

    parser.add_argument(
        "input_file",
        help="Input *.cxsmiles.bz2 file",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Maximum ligand records per output file "
            f"(default: {DEFAULT_CHUNK_SIZE:,})"
        ),
    )

    parser.add_argument(
        "--header-lines",
        type=int,
        default=1,
        help="Number of header lines to copy to every output file (default: 1)",
    )

    parser.add_argument(
        "--delete-original",
        action="store_true",
        help="Delete input file after successful verification",
    )

    parser.add_argument(
        "--overwrite-inprogress",
        action="store_true",
        help="Delete an existing *.inprogress directory before starting",
    )

    args = parser.parse_args()

    split_bz2_file(
        input_file=args.input_file,
        chunk_size=args.chunk_size,
        header_lines=args.header_lines,
        delete_original=args.delete_original,
        overwrite_inprogress=args.overwrite_inprogress,
    )


if __name__ == "__main__":
    main()
