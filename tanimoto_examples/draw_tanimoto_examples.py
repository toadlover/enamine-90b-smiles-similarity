#!/usr/bin/env python3

import argparse
import random

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# Settings
# ============================================================

QUERY_NAME = "PV-000518115177"
QUERY_SMILES = "CN1C(=O)C=CC=C1C(=O)NC2CCCCN(C2)C(=O)CC3(C)CCCCC3"

# Similarity brackets: lower bound inclusive, upper bound exclusive
BRACKETS = [
    (0.90, 1.00),
    (0.80, 0.90),
    (0.70, 0.80),
    (0.60, 0.70),
]

N_PER_BRACKET = 5

# Set this to any integer to get reproducible "random" selections.
RANDOM_SEED = 42


# ============================================================
# Helper functions
# ============================================================

def smiles_to_image(smiles, size=(300, 220)):
    """Generate a 2D RDKit depiction of a SMILES string."""

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        img = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "Invalid SMILES", fill="black")
        return img

    return Draw.MolToImage(
        mol,
        size=size,
        kekulize=True
    )


def load_font(size=18, bold=False):
    """
    Try some common fonts; fall back to PIL default if unavailable.
    """

    candidates = []

    if bold:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ])
    else:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ])

    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            pass

    return ImageFont.load_default()


def wrap_text(text, font, max_width, draw):
    """Wrap text according to pixel width."""

    text = str(text)
    lines = []
    current = ""

    for char in text:
        test = current + char

        bbox = draw.textbbox((0, 0), test, font=font)
        width = bbox[2] - bbox[0]

        if width <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = char

    if current:
        lines.append(current)

    return lines


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Select representative molecules from Tanimoto similarity ranges and draw them."
    )

    parser.add_argument(
        "csv_file",
        help="Input CSV file"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="tanimoto_representative_molecules.png",
        help="Output PNG filename"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducible sampling"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    # Input file has no header:
    # SMILES, ligand_name, source_file, tanimoto_score

    df = pd.read_csv(
        args.csv_file,
        header=None,
        names=[
            "smiles",
            "ligand_name",
            "source_file",
            "tanimoto"
        ]
    )

    df["tanimoto"] = pd.to_numeric(
        df["tanimoto"],
        errors="coerce"
    )

    df = df.dropna(subset=["tanimoto", "smiles"])

    # Explicitly remove exact 1.0 hits
    df = df[df["tanimoto"] < 1.0].copy()

    # --------------------------------------------------------
    # Randomly select molecules
    # --------------------------------------------------------

    rng = random.Random(args.seed)

    selected_groups = []

    for lower, upper in BRACKETS:

        subset = df[
            (df["tanimoto"] >= lower) &
            (df["tanimoto"] < upper)
        ].copy()

        print(
            f"{lower:.2f}-{upper:.2f}: "
            f"{len(subset):,} molecules available"
        )

        if len(subset) == 0:
            print("  WARNING: no molecules in this bracket")
            continue

        n_select = min(N_PER_BRACKET, len(subset))

        random_state = rng.randint(0, 2**32 - 1)

        selected = subset.sample(
            n=n_select,
            random_state=random_state
        )

        selected_groups.append(
            (
                f"{lower:.2f} – <{upper:.2f}",
                selected
            )
        )

    # --------------------------------------------------------
    # Image layout
    # --------------------------------------------------------

    molecule_width = 300
    molecule_height = 220

    left_margin = 35
    right_margin = 35

    label_width = 165

    gap = 15

    columns = N_PER_BRACKET

    canvas_width = (
        left_margin
        + label_width
        + columns * molecule_width
        + (columns - 1) * gap
        + right_margin
    )

    query_section_height = 350
    row_height = 360
    title_height = 65

    canvas_height = (
        title_height
        + query_section_height
        + len(selected_groups) * row_height
        + 50
    )

    canvas = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    title_font = load_font(26, bold=True)
    section_font = load_font(21, bold=True)
    normal_font = load_font(16)
    small_font = load_font(14)
    score_font = load_font(17, bold=True)

    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    draw.text(
        (left_margin, 20),
        "Representative molecules across Tanimoto similarity ranges",
        font=title_font,
        fill="black"
    )

    current_y = title_height

    # --------------------------------------------------------
    # Query ligand
    # --------------------------------------------------------

    draw.text(
        (left_margin, current_y + 10),
        "Original ligand",
        font=section_font,
        fill="black"
    )

    query_img = smiles_to_image(
        QUERY_SMILES,
        size=(molecule_width, molecule_height)
    )

    query_x = left_margin + label_width

    canvas.paste(
        query_img,
        (query_x, current_y + 45)
    )

    text_x = query_x + molecule_width + 25
    text_y = current_y + 75

    draw.text(
        (text_x, text_y),
        QUERY_NAME,
        font=score_font,
        fill="black"
    )

    draw.text(
        (text_x, text_y + 35),
        "SMILES:",
        font=normal_font,
        fill="black"
    )

    smiles_lines = wrap_text(
        QUERY_SMILES,
        small_font,
        canvas_width - text_x - right_margin,
        draw
    )

    for i, line in enumerate(smiles_lines):
        draw.text(
            (text_x, text_y + 65 + i * 20),
            line,
            font=small_font,
            fill="black"
        )

    current_y += query_section_height

    # Divider
    draw.line(
        (
            left_margin,
            current_y,
            canvas_width - right_margin,
            current_y
        ),
        fill="black",
        width=2
    )

    # --------------------------------------------------------
    # Similarity rows
    # --------------------------------------------------------

    for bracket_label, selected in selected_groups:

        row_top = current_y

        # Bracket name
        draw.text(
            (left_margin, row_top + 130),
            bracket_label,
            font=section_font,
            fill="black"
        )

        # Each selected molecule
        for col_index, (_, row) in enumerate(selected.iterrows()):

            x = (
                left_margin
                + label_width
                + col_index * (molecule_width + gap)
            )

            # Molecule drawing
            mol_img = smiles_to_image(
                row["smiles"],
                size=(molecule_width, molecule_height)
            )

            canvas.paste(
                mol_img,
                (x, row_top + 15)
            )

            # Ligand name
            ligand_name = str(row["ligand_name"])

            draw.text(
                (x + 5, row_top + 240),
                ligand_name,
                font=small_font,
                fill="black"
            )

            # Score
            score_text = f"Tanimoto = {row['tanimoto']:.3f}"

            draw.text(
                (x + 5, row_top + 265),
                score_text,
                font=score_font,
                fill="black"
            )

            # SMILES
            smiles_lines = wrap_text(
                row["smiles"],
                small_font,
                molecule_width - 10,
                draw
            )

            # Only use enough lines to stay within the row
            for line_index, line in enumerate(smiles_lines[:3]):
                draw.text(
                    (
                        x + 5,
                        row_top + 295 + line_index * 18
                    ),
                    line,
                    font=small_font,
                    fill="black"
                )

        current_y += row_height

        # Horizontal divider
        draw.line(
            (
                left_margin,
                current_y,
                canvas_width - right_margin,
                current_y
            ),
            fill="gray",
            width=1
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    canvas.save(
        args.output,
        dpi=(300, 300)
    )

    print()
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
