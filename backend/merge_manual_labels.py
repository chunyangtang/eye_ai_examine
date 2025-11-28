import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object at {path}, got {type(data).__name__}")
    return data


def _resolve_img_path(img_path: str, base_dir: Path) -> Path:
    src = Path(img_path)
    if not src.is_absolute():
        src = (base_dir / img_path).resolve()
    return src


def _copy_images(images: List[Dict[str, Any]], base_dir: Path, dest_root: Path, patient_id: str) -> List[Dict[str, Any]]:
    copied_images: List[Dict[str, Any]] = []
    dest_dir = dest_root / str(patient_id)
    dest_dir.mkdir(parents=True, exist_ok=True)

    for img in images or []:
        img_path = img.get("img_path")
        if not img_path:
            continue
        src = _resolve_img_path(img_path, base_dir)
        if not src.exists():
            print(f"[WARN] Image not found for patient {patient_id}: {src}")
            continue

        dest = dest_dir / src.name
        shutil.copy2(src, dest)

        img_copy = dict(img)
        img_copy["copied_img_path"] = str(dest)
        try:
            img_copy["copied_img_rel"] = str(dest.relative_to(dest_root.parent))
        except ValueError:
            # Fallback to basename if we cannot compute the relative path
            img_copy["copied_img_rel"] = dest.name
        copied_images.append(img_copy)
    return copied_images


def _is_patient_payload(val: Any) -> bool:
    return isinstance(val, dict) and (
        "manual_diagnosis" in val or "custom_diseases" in val or "diagnosis_notes" in val
    )


def _extract_manual_entries(manual_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Supports two shapes:
    1) {patient_id: {...manual...}}
    2) {date: {patient_id: {...manual...}}}
    3) {patient_id: {date: {...manual...}}}
    """
    entries: List[Dict[str, Any]] = []

    # Shape 1: values look like patient payloads
    if manual_data and all(_is_patient_payload(v) for v in manual_data.values()):
        for pid, payload in manual_data.items():
            entries.append(
                {
                    "patient_id": str(pid),
                    "manual": payload,
                    "date": payload.get("exam_date") or payload.get("date"),
                }
            )
        return entries

    # Shape 3: patient_id -> date -> payload
    if manual_data and all(isinstance(v, dict) for v in manual_data.values()):
        nested_entries: List[Dict[str, Any]] = []
        for pid, date_map in manual_data.items():
            if not isinstance(date_map, dict):
                continue
            for date_key, payload in date_map.items():
                if not _is_patient_payload(payload):
                    continue
                nested_entries.append(
                    {
                        "patient_id": str(pid),
                        "manual": payload,
                        "date": payload.get("exam_date") or payload.get("date") or str(date_key),
                    }
                )
        if nested_entries:
            return nested_entries

    # Shape 2: date -> patient map
    for date_key, patient_map in manual_data.items():
        if not isinstance(patient_map, dict):
            continue
        for pid, payload in patient_map.items():
            if not _is_patient_payload(payload):
                continue
            entries.append(
                {
                    "patient_id": str(pid),
                    "manual": payload,
                    "date": payload.get("exam_date") or payload.get("date") or str(date_key),
                }
            )
    if not entries:
        raise ValueError("examine_results.json format not recognized")
    return entries


def _parse_root_spec(spec: str) -> Tuple[str, Path]:
    """
    Accepts either 'name=path' or just 'path'. Returns (name, Path(path)).
    Name defaults to the directory name of the path.
    """
    if "=" in spec:
        name, path_str = spec.split("=", 1)
    else:
        path_str = spec
        name = Path(path_str).name or "root"
    return name, Path(path_str)


def _load_inference_sets(inference_roots: List[Tuple[str, Path]]) -> List[Tuple[str, str, Path, Dict[str, Any]]]:
    """
    Returns list of tuples: (root_name, date_str, inference_json_path, loaded_dict)
    """
    result: List[Tuple[str, str, Path, Dict[str, Any]]] = []
    for root_name, root in inference_roots:
        inference_paths = sorted(root.glob("**/inference_results.json"))
        for path in inference_paths:
            date_str = path.parent.name
            try:
                data = _load_json(path)
                result.append((root_name, date_str, path, data))
            except Exception as exc:
                print(f"[WARN] Skip {path}: {exc}")
    return result


def _find_inference_for_patient(
    patient_id: str,
    preferred_date: Optional[str],
    inference_sets: List[Tuple[str, str, Path, Dict[str, Any]]],
) -> List[Tuple[str, str, Path, Dict[str, Any]]]:
    """
    Find all inference sets that contain the patient.
    If preferred_date is provided, only return matches for that date.
    Otherwise return all matches (sorted by date desc), logging if multiple dates are present.
    """
    if preferred_date:
        matches = [
            (root_name, date_str, path, data)
            for root_name, date_str, path, data in inference_sets
            if date_str == preferred_date and patient_id in data
        ]
        return sorted(matches, key=lambda t: t[1], reverse=True)

    matches = [(root_name, date_str, path, data) for root_name, date_str, path, data in inference_sets if patient_id in data]
    matches.sort(key=lambda t: t[1], reverse=True)
    if len(matches) > 1:
        alt_dates = ", ".join(sorted({m[1] for m in matches}))
        print(f"[WARN] Patient {patient_id} found in multiple dates: {alt_dates}. Keeping all.")
    return matches


def merge_results(
    manual_path: Path,
    inference_roots: List[str],
    output_path: Path,
    dest_images_dir: Path,
    dry_run: bool = False,
) -> Dict[str, Dict[str, Any]]:
    # Normalize roots to (name, Path) tuples
    parsed_roots: List[Tuple[str, Path]] = [_parse_root_spec(r) for r in inference_roots]

    manual_data = _load_json(manual_path)
    manual_entries = _extract_manual_entries(manual_data)
    inference_sets = _load_inference_sets(parsed_roots)

    if not inference_sets:
        roots_str = ", ".join(str(r) for _, r in parsed_roots)
        raise ValueError(f"No inference_results.json found under roots: {roots_str}")

    combined: Dict[str, Dict[str, Any]] = {}

    for entry in manual_entries:
        patient_id = entry["patient_id"]
        preferred_date = entry.get("date")

        matches = _find_inference_for_patient(patient_id, preferred_date, inference_sets)
        if not matches:
            print(f"[WARN] No inference results found for patient {patient_id} (preferred date: {preferred_date})")
            continue

        # Use the first match (latest or the specified date) for copying images
        primary_root, primary_date, primary_path, primary_data = matches[0]
        primary_entry = primary_data.get(str(patient_id))
        if primary_entry is None:
            print(f"[WARN] Patient {patient_id} missing in {primary_path}")
            continue

        inference_results_by_source: Dict[str, Dict[str, Any]] = {}
        for root_name, date_str, inf_path, inf_data in matches:
            entry_data = inf_data.get(str(patient_id))
            if not isinstance(entry_data, dict):
                continue
            entry_copy = dict(entry_data)
            entry_copy["exam_date"] = date_str
            inference_results_by_source[root_name] = entry_copy

        if dry_run:
            copied_images = primary_entry.get("images", [])
        else:
            dest_images_dir.mkdir(parents=True, exist_ok=True)
            copied_images = _copy_images(
                primary_entry.get("images", []),
                base_dir=primary_path.parent,
                dest_root=dest_images_dir,
                patient_id=str(patient_id),
            )
        primary_copy = dict(primary_entry)
        primary_copy["images"] = copied_images
        primary_copy["exam_date"] = primary_date

        combined[str(patient_id)] = {
            "patient_id": str(patient_id),
            "exam_date": primary_date,
            "inference_results": primary_copy,  # backward-compatible primary
            "inference_results_by_source": inference_results_by_source,
            "manual_diagnosis": entry["manual"].get("manual_diagnosis"),
            "custom_diseases": entry["manual"].get("custom_diseases"),
            "diagnosis_notes": entry["manual"].get("diagnosis_notes"),
        }

    if dry_run:
        return combined

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote {len(combined)} patient(s) to {output_path}")

    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge inference results with manual labels and copy labeled images."
    )
    parser.add_argument(
        "--manual",
        default="Z:\\eye_ai_data\\examine_data\\examine_results.json",
        type=Path,
        help="Path to examine_results.json containing manual labels.",
    )
    parser.add_argument(
        "--inference-root",
        dest="inference_roots",
        action="append",
        type=Path,
        default=["baseline=Z:\\eye_ai_data\\inference_results", "focal=Z:\\eye_ai_data\\inference_results_FOCAL"],
        help="Root directory to search for inference_results.json (one per date). Can be provided multiple times.",
    )
    parser.add_argument(
        "--output",
        default="Z:\\eye_ai_data\\combined_results.json",
        type=Path,
        help="Path to write the merged JSON file.",
    )
    parser.add_argument(
        "--dest-images",
        default="Z:\\eye_ai_data\\manual_labelled_images",
        type=Path,
        help="Directory to copy images for manually labeled cases.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and match without copying or writing files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    roots = args.inference_roots if args.inference_roots else [Path("data")]
    combined = merge_results(
        manual_path=args.manual,
        inference_roots=roots,
        output_path=args.output,
        dest_images_dir=args.dest_images,
        dry_run=args.dry_run,
    )
    print(f"Merged {len(combined)} patient(s) across {len(roots)} root(s).")
    if args.dry_run:
        print("[DRY-RUN] Patients:", list(combined.keys()))
        return
    print(f"Combined JSON written to: {args.output}")
    print(f"Copied images under: {args.dest_images}")


if __name__ == "__main__":
    main()
