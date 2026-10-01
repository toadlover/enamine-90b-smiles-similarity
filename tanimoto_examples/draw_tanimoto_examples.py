#!/usr/bin/env python3

import argparse
import random
from io import BytesIO

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D


# ============================================================
# User settings
# ============================================================

QUERY_NAME = "PV-000518115177"
QUERY_SMILES = "CN1C(=O)C=CC=C1C(=O)NC2CCCCN(C2)C(=O)CC3(C)CCCCC3"

# lower inclusive, upper exclusive
BRACKETS = [
    (0.90, 1.00),
    (0.80, 0.90),
    (0.70, 0.80),
    (0.60, 0.70),
]

N_PER_BRACKET = 5
RANDOM_SEED = 42

# layout / drawing controls
CELL_W = 320
MOL_W = 320
MOL_H = 210
ROW_LABEL_W = 155
GAP_X = 14
LEFT_MARGIN = 30
RIGHT_MARGIN = 30
TOP_MARGIN = 20

QUERY_SECTION_H = 320
ROW_H = 355
TITLE_H = 45


# ============================================================
# Fonts
# ============================================================

def load_font(size=16, bold=False):
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

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass

    return ImageFont.load_default()


TITLE_FONT = load_font(24, bold=True)
SECTION_FONT = load_font(20, bold=True)
LABEL_FONT = load_font(17, bold=True)
TEXT_FONT = load_font(14, bold=False)
SMALL_FONT = load_font(13, bold=False)


# ============================================================
# Text wrapping helpers
# ============================================================

def text_width(draw, text, font):
    bbox = draw.textbbox((0, 0), str(text), font=font)
    return bbox[2] - bbox[0]


def wrap_text_pixels(draw, text, font, max_width):
    """
    Character-level wrapping so it also works for long IDs/SMILES
    without spaces.
    """
    text = str(text)
    lines = []
    current = ""

    for ch in text:
        trial = current + ch
        if text_width(draw, trial, font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = ch

    if current:
        lines.append(current)

    return lines


def clamp_lines(draw, lines, font, max_width, max_lines):
    """
    Limit line count and add ... to the last line if truncated.
    """
    if len(lines) <= max_lines:
        return lines

    clipped = lines[:max_lines]
    last = clipped[-1]

    while last and text_width(draw, last + "...", font) > max_width:
        last = last[:-1]

    clipped[-1] = last + "..."
    return clipped


def draw_wrapped_text(draw, x, y, text, font, max_width, max_lines, fill="black", line_spacing=2):
    lines = wrap_text_pixels(draw, text, font, max_width)
    lines = clamp_lines(draw, lines, font, max_width, max_lines)

    bbox = draw.textbbox((0, 0), "Ag", font=font)
    line_h = bbox[3] - bbox[1]

    for i, line in enumerate(lines):
        draw.text((x, y + i * (line_h + line_spacing)), line, font=font, fill=fill)

    total_h = len(lines) * line_h + max(0, len(lines) - 1) * line_spacing
    return total_h


# ============================================================
# Molecule depiction helpers
# ============================================================

def prepare_reference_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Could not parse query SMILES")
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol)
    return mol


def align_mol_to_reference(mol, ref_mol):
    """
    Try to orient mol similarly to ref_mol using MCS.
    Falls back to ordinary 2D coordinates if matching fails.
    """
    mol = Chem.Mol(mol)
    rdDepictor.SetPreferCoordGen(True)

    try:
        # Start with ordinary 2D coords
        rdDepictor.Compute2DCoords(mol)

        # Find maximum common substructure
        mcs = rdFMCS.FindMCS(
            [ref_mol, mol],
            timeout=5,
            ringMatchesRingOnly=True,
            completeRingsOnly=False,
            matchValences=False,
        )

        if not mcs.smartsString:
            return mol

        patt = Chem.MolFromSmarts(mcs.smartsString)
        if patt is None:
            return mol

        ref_match = ref_mol.GetSubstructMatch(patt)
        mol_match = mol.GetSubstructMatch(patt)

        # Need at least a few atoms to make orientation meaningful
        if len(ref_match) < 3 or len(mol_match) < 3:
            return mol

        atom_map = list(zip(mol_match, ref_match))

        rdDepictor.GenerateDepictionMatching2DStructure(
            mol,
            ref_mol,
            atomMap=atom_map
        )

        return mol

    except Exception:
        # Fall back silently
        try:
            rdDepictor.Compute2DCoords(mol)
        except Exception:
            pass
        return mol


def render_molecule_image(smiles, ref_mol=None, size=(MOL_W, MOL_H), align=True):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        img = Image.new("RGB", size, "white")
        d = ImageDraw.Draw(img)
        d.text((10, 10), "Invalid SMILES", font=TEXT_FONT, fill="black")
        return img

    if align and ref_mol is not None:
        mol = align_mol_to_reference(mol, ref_mol)
    else:
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()

    # More padding so longer molecules feel “zoomed out”
    opts.padding = 0.15

    # Optional cosmetic tweaks
    opts.clearBackground = False
    opts.addStereoAnnotation = False

    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()

    png = drawer.GetDrawingText()
    return Image.open(BytesIO(png)).convert("RGB")


# ============================================================
# Data handling
# ============================================================

def load_input_csv(csv_file):
    """
    Assumes headerless CSV with:
    smiles, ligand_name, source_file, tanimoto
    """
    df = pd.read_csv(
        csv_file,
        header=None,
        names=["smiles", "ligand_name", "source_file", "tanimoto"]
    )

    df["tanimoto"] = pd.to_numeric(df["tanimoto"], errors="coerce")
    df = df.dropna(subset=["smiles", "tanimoto"]).copy()

    # Ignore exact 1.0 hits
    df = df[df["tanimoto"] < 1.0].copy()

    return df


def sample_by_brackets(df, brackets, n_per_bracket=5, seed=42):
    rng = random.Random(seed)
    selected_groups = []

    for low, high in brackets:
        subset = df[(df["tanimoto"] >= low) & (df["tanimoto"] < high)].copy()

        print(f"{low:.2f}-{high:.2f}: {len(subset):,} available")

        if len(subset) == 0:
            continue

        n_pick = min(n_per_bracket, len(subset))
        picked = subset.sample(n=n_pick, random_state=rng.randint(0, 2**32 - 1))

        selected_groups.append((f"{low:.2f} – <{high:.2f}", picked.reset_index(drop=True)))

    return selected_groups


# ============================================================
# Figure drawing
# ============================================================

def draw_query_section(canvas, draw, ref_mol, total_width):
    draw.text((LEFT_MARGIN, TITLE_H + 10), "Original ligand", font=SECTION_FONT, fill="black")

    qx = LEFT_MARGIN + ROW_LABEL_W
    qy = TITLE_H + 45

    qimg = render_molecule_image(QUERY_SMILES, ref_mol=ref_mol, align=False)
    canvas.paste(qimg, (qx, qy))

    info_x = qx + MOL_W + 25
    info_y = qy + 15

    draw.text((info_x, info_y), QUERY_NAME, font=LABEL_FONT, fill="black")
    draw.text((info_x, info_y + 30), "SMILES:", font=TEXT_FONT, fill="black")

    draw_wrapped_text(
        draw,
        info_x,
        info_y + 55,
        QUERY_SMILES,
        font=SMALL_FONT,
        max_width=total_width - info_x - RIGHT_MARGIN,
        max_lines=6,
        fill="black"
    )


def draw_cell(canvas, draw, x, y, row):
    img = render_molecule_image(row["smiles"], ref_mol=REF_MOL, align=True)
    canvas.paste(img, (x, y))

    text_x = x + 5

    # Ligand name (wrapped)
    name_y = y + MOL_H + 6
    name_h = draw_wrapped_text(
        draw,
        text_x,
        name_y,
        str(row["ligand_name"]),
        font=SMALL_FONT,
        max_width=CELL_W - 10,
        max_lines=3,
        fill="black"
    )

    # Score
    score_y = name_y + name_h + 6
    draw.text(
        (text_x, score_y),
        f"Tanimoto = {row['tanimoto']:.3f}",
        font=LABEL_FONT,
        fill="black"
    )

    # SMILES
    smiles_y = score_y + 24
    draw_wrapped_text(
        draw,
        text_x,
        smiles_y,
        str(row["smiles"]),
        font=SMALL_FONT,
        max_width=CELL_W - 10,
        max_lines=4,
        fill="black"
    )


def make_figure(selected_groups, output_png):
    n_cols = N_PER_BRACKET

    total_width = (
        LEFT_MARGIN
        + ROW_LABEL_W
        + n_cols * CELL_W
        + (n_cols - 1) * GAP_X
        + RIGHT_MARGIN
    )

    total_height = (
        TOP_MARGIN
        + TITLE_H
        + QUERY_SECTION_H
        + len(selected_groups) * ROW_H
        + 40
    )

    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)

    # title
    draw.text(
        (LEFT_MARGIN, TOP_MARGIN),
        "Representative molecules across Tanimoto similarity ranges",
        font=TITLE_FONT,
        fill="black"
    )

    # query section
    draw_query_section(canvas, draw, REF_MOL, total_width)

    # divider below query section
    current_y = TITLE_H + QUERY_SECTION_H
    draw.line(
        (LEFT_MARGIN, current_y, total_width - RIGHT_MARGIN, current_y),
        fill="black",
        width=2
    )

    # similarity rows
    for bracket_label, group_df in selected_groups:
        row_top = current_y + 12

        # left-side bracket label
        draw.text(
            (LEFT_MARGIN, row_top + 100),
            bracket_label,
            font=SECTION_FONT,
            fill="black"
        )

        # draw each cell
        for col_idx, (_, row) in enumerate(group_df.iterrows()):
            x = LEFT_MARGIN + ROW_LABEL_W + col_idx * (CELL_W + GAP_X)
            draw_cell(canvas, draw, x, row_top, row)

        current_y += ROW_H

        draw.line(
            (LEFT_MARGIN, current_y, total_width - RIGHT_MARGIN, current_y),
            fill="gray",
            width=1
        )

    canvas.save(output_png, dpi=(300, 300))
    print(f"\nSaved figure to: {output_png}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Draw representative molecules by Tanimoto similarity bracket."
    )
    parser.add_argument("csv_file", help="Input CSV file")
    parser.add_argument(
        "-o", "--output",
        default="tanimoto_representative_molecules.png",
        help="Output PNG file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducible sampling"
    )
    args = parser.parse_args()

    df = load_input_csv(args.csv_file)
    selected_groups = sample_by_brackets(
        df,
        BRACKETS,
        n_per_bracket=N_PER_BRACKET,
        seed=args.seed
    )
    make_figure(selected_groups, args.output)


if __name__ == "__main__":
    REF_MOL = prepare_reference_mol(QUERY_SMILES)
    main()
