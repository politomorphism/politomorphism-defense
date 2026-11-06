#!/usr/bin/env python3
"""
Inspect SemEval-2020-style technique annotations.

Expected annotation format, without header:
article_id<TAB>label<TAB>start<TAB>end

Offsets are interpreted as Python character offsets:
start inclusive, end exclusive.

This script performs data inspection only.
It does not validate label mappings, construct negatives,
call a model, or compute classifier performance.

Workflow position: this is step 22 (data inspection). It runs BEFORE
21_semeval_eval.py: inspect the real files here, confirm spans reconstruct and
read the label list, then fill 21's config (mapping with justification records,
unit, negatives rule) and run 21 --dry-run. Only after both pass does any
evaluation loop run.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


ARTICLE_PATTERN = re.compile(r"article(\d+)\.txt", re.IGNORECASE)


def read_raw_text(path):
    # Preserve line endings. Do not strip or normalize article text.
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_id(value):
    value = value.strip()

    if value.isdigit():
        return str(int(value))

    match = re.fullmatch(
        r"article(\d+)(?:\.txt)?",
        value,
        flags=re.IGNORECASE,
    )
    if match:
        return str(int(match.group(1)))

    raise ValueError(f"Invalid article ID: {value!r}")


def save_json(path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect raw articles and span annotations."
    )
    parser.add_argument(
        "--articles",
        required=True,
        type=Path,
        help="Directory containing article<ID>.txt files.",
    )
    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="One annotation file or a directory of annotation files.",
    )
    parser.add_argument(
        "--labels-glob",
        default="*.labels",
        help="Pattern used if --labels is a directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("audit_semeval"),
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=200,
        help="Characters displayed before and after each span.",
    )
    parser.add_argument(
        "--dataset-split",
        required=True,
        choices=["train", "dev", "test"],
        help="Split declared according to the dataset documentation.",
    )
    parser.add_argument(
        "--split-source",
        required=True,
        help="Documented source supporting the split assignment: README "
             "section, official archive name, or documentation URL.",
    )
    parser.add_argument(
        "--verify-sample",
        type=int,
        default=0,
        help="Print this many randomly-sampled spans (text + context) for MANUAL "
             "confirmation that the fragment matches the annotation; range checks "
             "alone do not prove semantic correspondence.",
    )
    parser.add_argument(
        "--peek",
        action="store_true",
        help="Print the first raw lines and the column-count histogram of each "
             "annotation file, then exit WITHOUT parsing. Use this first to "
             "confirm the real corpus uses the article_id/label/start/end 4-column "
             "format before trusting the parse.",
    )
    args = parser.parse_args()

    if not args.split_source.strip():
        parser.error("--split-source must contain a provenance reference.")
    # Note: this is a DOCUMENTED declaration, not an automatic verification of
    # split membership.

    if args.context_chars < 0:
        parser.error("--context-chars must be non-negative.")

    if not args.articles.is_dir():
        parser.error("--articles must point to an existing directory.")

    if args.labels.is_file():
        label_files = [args.labels]
    elif args.labels.is_dir():
        label_files = sorted(
            path
            for path in args.labels.rglob(args.labels_glob)
            if path.is_file()
        )
    else:
        parser.error("--labels does not exist.")

    if not label_files:
        parser.error("No annotation files found. Check --labels-glob.")

    # Improvement 2: confirm the real TSV format before trusting the parse.
    if args.peek:
        for path in label_files:
            print(f"\n=== {path} ===")
            counts = Counter()
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for i, row in enumerate(csv.reader(handle, delimiter="\t")):
                    if row and any(c.strip() for c in row):
                        counts[len(row)] += 1
                    if i < 5:
                        print(f"  line {i+1} ({len(row)} cols): {row}")
            print(f"  column-count histogram (non-empty rows): {dict(counts)}")
        print("\npeek only: no parsing done. Confirm the 4-column "
              "article_id/label/start/end format before a full run.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)

    errors = []
    articles = {}
    manifest = []

    # Read articles without altering the text.
    for path in sorted(args.articles.rglob("*")):
        if not path.is_file():
            continue

        match = ARTICLE_PATTERN.fullmatch(path.name)
        if not match:
            continue

        article_id = str(int(match.group(1)))

        if article_id in articles:
            errors.append(
                f"Duplicate article ID {article_id}: "
                f"{articles[article_id]['path']} and {path}"
            )
            continue

        try:
            text = read_raw_text(path)
        except (UnicodeError, OSError) as exc:
            errors.append(f"Cannot read article {path}: {exc}")
            continue

        articles[article_id] = {
            "path": str(path),
            "text": text,
        }
        manifest.append({
            "kind": "article",
            "path": str(path),
            "sha256": sha256(path),
        })

    if not articles:
        errors.append("No readable article<ID>.txt files found.")

    # Aggregate labels only when article ID and boundaries match exactly.
    span_labels = {}
    span_sources = {}
    span_source_files = {}
    column_counts = Counter()
    seen_annotations = set()
    duplicate_annotations = 0

    for path in label_files:
        manifest.append({
            "kind": "annotations",
            "path": str(path),
            "sha256": sha256(path),
        })

        try:
            with path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.reader(handle, delimiter="\t"))
        except (UnicodeError, OSError) as exc:
            errors.append(f"Cannot read annotations {path}: {exc}")
            continue

        for line_number, fields in enumerate(rows, start=1):
            if not fields or all(not field.strip() for field in fields):
                continue

            location = f"{path}:{line_number}"
            column_counts[len(fields)] += 1

            if len(fields) != 4:
                errors.append(
                    f"{location}: expected 4 TSV columns, "
                    f"found {len(fields)}."
                )
                continue

            try:
                article_id = normalize_id(fields[0])
                label = fields[1].strip()
                start = int(fields[2])
                end = int(fields[3])

                if not label:
                    raise ValueError("Empty label.")

                if article_id not in articles:
                    raise ValueError(
                        f"Article {article_id} is missing."
                    )

                text = articles[article_id]["text"]

                if not (0 <= start < end <= len(text)):
                    raise ValueError(
                        f"Invalid offsets [{start}, {end}) "
                        f"for text length {len(text)}."
                    )

                if not text[start:end].strip():
                    raise ValueError("Annotated span is whitespace only.")

            except ValueError as exc:
                errors.append(f"{location}: {exc}")
                continue

            annotation = (article_id, start, end, label)

            if annotation in seen_annotations:
                duplicate_annotations += 1
            seen_annotations.add(annotation)

            key = (article_id, start, end)
            span_labels.setdefault(key, set()).add(label)
            span_sources.setdefault(key, []).append(location)
            span_source_files.setdefault(key, set()).add(str(path))

    # Record provenance on every manifest entry (declared, not auto-verified).
    for entry in manifest:
        entry["dataset_split"] = args.dataset_split
        entry["split_source"] = args.split_source
        entry["split_assignment_status"] = "declared_not_independently_verified"

    # Detect exact byte-identical articles, including different article IDs.
    paths_by_hash = {}
    for entry in manifest:
        if entry["kind"] != "article":
            continue
        paths_by_hash.setdefault(entry["sha256"], []).append(entry["path"])

    exact_duplicate_groups = [
        {"sha256": digest, "paths": paths, "file_count": len(paths)}
        for digest, paths in sorted(paths_by_hash.items())
        if len(paths) > 1
    ]
    duplicate_audit = {
        "scope": "article files included in this run",
        "comparison": "SHA-256 of original file bytes",
        "exact_duplicate_groups": exact_duplicate_groups,
        "extra_identical_files": sum(
            g["file_count"] - 1 for g in exact_duplicate_groups
        ),
        "files_removed": False,
        "limitations": [
            "Different bytes do not prove different substantive content.",
            "Near-duplicates are not detected.",
            "Cross-split duplication requires comparison with other manifests.",
            "Duplicate groups require a documented decision before evaluation.",
        ],
    }

    save_json(args.out / "manifest.json", manifest)
    save_json(args.out / "duplicates.json", duplicate_audit)

    # Avoid leaving old successful outputs after a failed rerun.
    for name in ("spans.csv", "report.json"):
        old_file = args.out / name
        if old_file.exists():
            old_file.unlink()

    if errors:
        save_json(args.out / "errors.json", errors)
        print(f"STOP: {len(errors)} error(s).")
        print(f"Details: {args.out / 'errors.json'}")
        return 1

    if not span_labels:
        save_json(
            args.out / "errors.json",
            ["No valid annotated spans found."],
        )
        print("STOP: no valid annotated spans found.")
        return 1

    old_errors = args.out / "errors.json"
    if old_errors.exists():
        old_errors.unlink()

    label_counts = Counter()
    articles_per_label = {}
    output_rows = []

    for (article_id, start, end), labels in sorted(span_labels.items()):
        text = articles[article_id]["text"]
        context_start = max(0, start - args.context_chars)
        context_end = min(len(text), end + args.context_chars)

        for label in labels:
            label_counts[label] += 1
            articles_per_label.setdefault(label, set()).add(article_id)

        output_rows.append({
            "article_id": article_id,
            "article_path": articles[article_id]["path"],
            "start": start,
            "end": end,
            "labels_json": json.dumps(sorted(labels), ensure_ascii=False),
            "span_text": text[start:end],
            "context_before": text[context_start:start],
            "context_after": text[end:context_end],
            "annotation_sources_json": json.dumps(
                span_sources[(article_id, start, end)],
                ensure_ascii=False,
            ),
        })

    fieldnames = [
        "article_id",
        "article_path",
        "start",
        "end",
        "labels_json",
        "span_text",
        "context_before",
        "context_after",
        "annotation_sources_json",
    ]

    with (args.out / "spans.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    multi_same = sum(
        1 for key, labels in span_labels.items()
        if len(labels) > 1 and len(span_source_files.get(key, set())) == 1
    )
    multi_cross = sum(
        1 for key, labels in span_labels.items()
        if len(labels) > 1 and len(span_source_files.get(key, set())) > 1
    )

    report = {
        "status": "technical_inspection_passed",
        "meaning": (
            "Files parsed and offsets passed range checks. "
            "Manual inspection is still needed to confirm text alignment."
        ),
        "dataset_split": args.dataset_split,
        "split_source": args.split_source,
        "split_assignment_status": "declared_not_independently_verified",
        "tsv_column_counts": dict(column_counts),
        "articles_loaded": len(articles),
        "articles_with_annotations": len({
            key[0] for key in span_labels
        }),
        "exact_duplicate_groups": len(exact_duplicate_groups),
        "extra_identical_files": duplicate_audit["extra_identical_files"],
        "duplicate_review_required": bool(exact_duplicate_groups),
        "unique_spans": len(span_labels),
        "unique_span_label_annotations": len(seen_annotations),
        "duplicate_annotation_rows": duplicate_annotations,
        "spans_with_multiple_labels": sum(
            len(labels) > 1 for labels in span_labels.values()
        ),
        "multi_label_same_file": multi_same,
        "multi_label_cross_file": multi_cross,
        "manual_span_review_completed": False,
        "labels": [
            {
                "label": label,
                "annotated_spans": count,
                "articles": len(articles_per_label[label]),
            }
            for label, count in sorted(label_counts.items())
        ],
        "mapping_verified": False,
        "negatives_constructed": False,
        "classifier_evaluated": False,
        "cautions": [
            "Do not infer target absence from another label alone.",
            "Do not treat unannotated text as automatically negative.",
            "Context columns are inspection aids, not a frozen model input.",
            "Use one documented dataset split per audit.",
            "Range checks do not prove the fragment matches the annotation; "
            "verify a sample manually (--verify-sample).",
        ],
    }
    if exact_duplicate_groups:
        report["cautions"].append(
            f"{len(exact_duplicate_groups)} group(s) of byte-identical article "
            f"files (see duplicates.json); documented decision required before "
            f"evaluation. The script flags, it does not remove them.")
    if multi_cross:
        report["cautions"].append(
            f"{multi_cross} multi-label span(s) draw labels from DIFFERENT files; "
            f"confirm these are genuine multi-labels, not split/corpus duplication.")

    save_json(args.out / "report.json", report)

    # Improvement 1: manual sample verification of fragment<->annotation match.
    if args.verify_sample > 0:
        import random
        keys = list(span_labels)
        random.Random(0).shuffle(keys)
        print(f"\n--- MANUAL VERIFICATION SAMPLE ({min(args.verify_sample, len(keys))}) ---")
        for (aid, s, e) in keys[:args.verify_sample]:
            txt = articles[aid]["text"]
            print(f"[article {aid} {s}:{e}] labels={sorted(span_labels[(aid,s,e)])}")
            print(f"    ...{txt[max(0,s-40):s]}[[{txt[s:e]}]]{txt[e:e+40]}...")
        print("--- confirm each bracketed span is the intended fragment ---")

    print("Technical inspection completed.")
    print(f"Articles loaded: {len(articles)}  split={args.dataset_split} "
          f"(declared; source: {args.split_source})")
    print(f"Unique spans: {len(span_labels)}  tsv_columns={dict(column_counts)}")
    if exact_duplicate_groups:
        print(f"WARNING: {len(exact_duplicate_groups)} byte-identical article "
              f"group(s); see {args.out / 'duplicates.json'}")
    print("\nExact labels found:")

    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    print(f"\nInspect: {args.out / 'spans.csv'}")
    print(f"Summary: {args.out / 'report.json'}")
    print("No classifier was run. No negatives were created.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
