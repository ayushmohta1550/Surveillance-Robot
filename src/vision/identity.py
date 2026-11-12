import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.transforms import functional as TF
from ultralytics import YOLO


BODY_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

FACE_NAMES = [
    "face_left_eye",
    "face_right_eye",
    "face_nose",
    "face_left_mouth",
    "face_right_mouth",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SquarePad:
    def __init__(self, fill=255):
        self.fill = fill

    def __call__(self, img):
        width, height = img.size
        side = max(width, height)
        pad_left = (side - width) // 2
        pad_top = (side - height) // 2
        pad_right = side - width - pad_left
        pad_bottom = side - height - pad_top
        return TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)


def prepare_local_caches(repo_root):
    cache_root = repo_root / ".cache"
    home_root = cache_root / "home"
    yolo_cache = home_root / ".cache"
    yolo_cache.mkdir(parents=True, exist_ok=True)

    os.environ["USERPROFILE"] = str(home_root)
    os.environ["HOME"] = str(home_root)
    os.environ["YOLO_CONFIG_DIR"] = str(cache_root / "ultralytics")
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "huggingface" / "transformers")
    os.environ["TORCH_HOME"] = str(cache_root / "torch")

    for key in ["YOLO_CONFIG_DIR", "HF_HOME", "TRANSFORMERS_CACHE", "TORCH_HOME"]:
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

    src_yolo = repo_root / "tasks" / "WebBody" / "c_detect" / "yolov8n-pose.pt"
    dst_yolo = yolo_cache / "yolov8n-pose.pt"
    if not dst_yolo.exists():
        shutil.copy2(src_yolo, dst_yolo)


def patch_torch_load_for_cpu():
    if torch.cuda.is_available():
        return

    original_torch_load = torch.load

    def cpu_torch_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("map_location", torch.device("cpu"))
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = cpu_torch_load


def make_aligner(repo_root):
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "tasks" / "sapiensID"))
    patch_torch_load_for_cpu()

    from src.aligners import get_aligner

    cfg = OmegaConf.create(
        {
            "name": "keypoint_predictor",
            "start_from": "",
            "freeze": True,
            "rgb_mean": [0.5, 0.5, 0.5],
            "rgb_std": [0.5, 0.5, 0.5],
            "use_yolo": True,
            "use_openpose": False,
            "face_detector": "dfa_mobilenetv4",
            "faceness_threshold": 0.5,
            "use_aligned_face_template": False,
            "use_cache": True,
        }
    )
    aligner = get_aligner(cfg)
    aligner.eval()
    return aligner


def resolve_checkpoint_dir(repo_root, ckpt_dir):
    if not ckpt_dir:
        ckpt_dir = repo_root / "checkpoints" / "sapiensid_wb4m"
    else:
        ckpt_dir = Path(ckpt_dir)
        if not ckpt_dir.is_absolute():
            ckpt_dir = repo_root / ckpt_dir
    return ckpt_dir


def check_checkpoint(ckpt_dir):
    model_yaml = ckpt_dir / "model.yaml"
    model_pth = ckpt_dir / "model.pth"
    missing = [str(path) for path in [model_yaml, model_pth] if not path.exists()]
    return missing


def get_runtime_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_sapiens_model(repo_root, ckpt_dir, device):
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "tasks" / "sapiensID"))
    patch_torch_load_for_cpu()

    from src.models import get_model
    from general_utils.config_utils import load_config

    model_config = load_config(str(ckpt_dir / "model.yaml"))
    if not hasattr(model_config, "yaml_path"):
        model_config.yaml_path = str(repo_root / "tasks" / "sapiensID" / "src" / "models" / "sapiens_id" / "configs" / "base.yaml")

    model = get_model(model_config)
    model.load_state_dict_from_path(str(ckpt_dir / "model.pth"))
    model = model.to(device)
    model.eval()
    return model


def make_default_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            SquarePadTensor(fill=1),
            transforms.Resize((384, 384), antialias=None),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


class SquarePadTensor:
    def __init__(self, fill=1):
        self.fill = fill

    def __call__(self, img):
        _, height, width = img.shape
        side = max(width, height)
        pad_left = (side - width) // 2
        pad_top = (side - height) // 2
        pad_right = side - width - pad_left
        pad_bottom = side - height - pad_top
        return TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)


def tensor_to_pil(x):
    x = x.squeeze(0).detach().cpu()
    x = (x * 0.5 + 0.5).clamp(0, 1)
    return TF.to_pil_image(x)


def keypoints_to_records(keypoints):
    names = BODY_NAMES + FACE_NAMES
    records = []
    for idx, (name, xy) in enumerate(zip(names, keypoints.tolist())):
        x, y = xy
        valid = not (x == -1 and y == -1)
        records.append(
            {
                "index": idx,
                "name": name,
                "x_normalized": round(float(x), 6),
                "y_normalized": round(float(y), 6),
                "valid": bool(valid),
            }
        )
    return records


def draw_keypoints(image, records):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for record in records:
        if not record["valid"]:
            continue
        x = record["x_normalized"] * width
        y = record["y_normalized"] * height
        radius = 4 if record["index"] < 17 else 5
        color = "lime" if record["index"] < 17 else "red"
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="black")
        draw.text((x + 6, y - 6), str(record["index"]), fill=color)
    return image


def save_embedding_output(model, image_tensor, keypoints, foreground_masks, output_dir, device):
    image_tensor = image_tensor.to(device)
    keypoints = keypoints.to(device)
    foreground_masks = foreground_masks.to(device)
    with torch.no_grad():
        embedding = model(image_tensor, foreground_masks=foreground_masks, ldmks=keypoints)

    embedding = embedding.detach().cpu()
    embedding_path = output_dir / "sapiens_embedding.pt"
    embedding_json_path = output_dir / "sapiens_embedding_summary.json"
    torch.save(embedding, embedding_path)

    embedding_norm = torch.norm(embedding, dim=1)
    preview = embedding[0, : min(16, embedding.shape[1])].tolist()
    summary = {
        "embedding_shape": list(embedding.shape),
        "embedding_norm": round(float(embedding_norm[0]), 6),
        "embedding_preview_first_16_values": [round(float(v), 6) for v in preview],
        "embedding_path": str(embedding_path),
    }
    embedding_json_path.write_text(json.dumps(summary, indent=2))
    return summary, embedding


def save_aligner_output(aligner, image_path, output_dir, label, transform, device, sapiens_model=None):
    pil = Image.open(image_path).convert("RGB")
    x = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        keypoints, foreground_masks = aligner(x)

    records = keypoints_to_records(keypoints[0].cpu())
    valid_count = sum(1 for record in records if record["valid"])

    output_dir.mkdir(parents=True, exist_ok=True)
    keypoints_path = output_dir / "keypoints.json"
    vis_path = output_dir / "aligner_keypoints.png"
    mask_path = output_dir / "foreground_mask.png"

    draw_keypoints(tensor_to_pil(x).copy(), records).save(vis_path)
    mask = foreground_masks[0].detach().cpu().to(torch.uint8) * 255
    TF.to_pil_image(mask).save(mask_path)

    data = {
        "label": label,
        "input_image": str(image_path),
        "input_tensor_shape": list(x.shape),
        "keypoints_shape": list(keypoints.shape),
        "foreground_mask_shape": list(foreground_masks.shape),
        "valid_keypoints": valid_count,
        "keypoints_path": str(keypoints_path),
        "visualization_path": str(vis_path),
        "foreground_mask_path": str(mask_path),
        "keypoints": records,
    }

    if sapiens_model is not None:
        embedding_summary, embedding = save_embedding_output(
            sapiens_model, x, keypoints, foreground_masks, output_dir, device
        )
        data["step_4_sapiens_embedding"] = embedding_summary
        data["_embedding_tensor"] = embedding

    json_data = {k: v for k, v in data.items() if k != "_embedding_tensor"}
    keypoints_path.write_text(json.dumps(json_data, indent=2))
    return data


def pad_bbox_xyxyn(bbox, padding_ratio):
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    pad_x = padding_ratio * width
    pad_y = padding_ratio * height
    return torch.tensor(
        [
            max(0.0, float(xmin - pad_x)),
            max(0.0, float(ymin - pad_y)),
            min(1.0, float(xmax + pad_x)),
            min(1.0, float(ymax + pad_y)),
        ]
    )


def crop_persons_webbody_style(image_path, yolo_path, output_dir, confidence_threshold, padding_ratio, max_detections):
    model = YOLO(str(yolo_path))
    # Run confidence filtering and non-maximum suppression inside YOLO. The
    # 0.70 IoU keeps distinct, heavily-overlapping people in group scenes while
    # suppressing duplicate boxes for the same person.
    result = model.predict(
        str(image_path),
        conf=confidence_threshold,
        iou=0.70,
        imgsz=640,
        max_det=max_detections,
        verbose=False,
    )[0]
    image = Image.open(image_path).convert("RGB")

    crop_dir = output_dir / "person_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    detections = []
    for det_idx, (bbox, conf) in enumerate(zip(result.boxes.xyxyn, result.boxes.conf)):
        if float(conf) < confidence_threshold:
            continue
        if len(detections) >= max_detections:
            break

        padded_bbox = pad_bbox_xyxyn(bbox.cpu(), padding_ratio)
        x1, y1, x2, y2 = padded_bbox.tolist()
        abs_box = (
            int(x1 * image.width),
            int(y1 * image.height),
            int(x2 * image.width),
            int(y2 * image.height),
        )
        crop_path = crop_dir / f"person_{det_idx}.jpg"
        image.crop(abs_box).save(crop_path)
        keypoint_confidences = []
        if result.keypoints is not None and result.keypoints.conf is not None:
            keypoint_confidences = [
                round(float(value), 6)
                for value in result.keypoints.conf[det_idx].cpu().tolist()
            ]
        # COCO pose order: shoulders 5-6, elbows/wrists 7-10,
        # hips/knees/ankles 11-16. Visibility gates prevent OpenPAR from
        # describing body regions that are outside the frame.
        visible = lambda indices: bool(keypoint_confidences) and max(
            keypoint_confidences[index] for index in indices
        ) >= 0.30
        detections.append(
            {
                "det_idx": det_idx,
                "confidence": round(float(conf), 6),
                "bbox_xyxyn": [round(float(v), 6) for v in bbox.cpu().tolist()],
                "padded_bbox_xyxyn": [round(float(v), 6) for v in padded_bbox.tolist()],
                "crop_path": str(crop_path),
                "pose_keypoint_confidences": keypoint_confidences,
                "visibility": {
                    "upper_body": visible([5, 6]),
                    "lower_body": visible([11, 12, 13, 14, 15, 16]),
                    "carried_items": visible([7, 8, 9, 10]),
                },
            }
        )

    return detections


def run_direct_mode(aligner, image_path, output_dir, transform, device, sapiens_model=None):
    return [
        save_aligner_output(
            aligner,
            image_path,
            output_dir / "direct_aligner",
            "direct_aligner",
            transform=transform,
            device=device,
            sapiens_model=sapiens_model,
        )
    ]


def run_webbody_mode(
    repo_root,
    aligner,
    image_path,
    output_dir,
    confidence_threshold,
    padding_ratio,
    max_detections,
    transform,
    device,
    sapiens_model=None,
):
    yolo_path = repo_root / "tasks" / "WebBody" / "c_detect" / "yolov8n-pose.pt"
    detections = crop_persons_webbody_style(
        image_path=image_path,
        yolo_path=yolo_path,
        output_dir=output_dir,
        confidence_threshold=confidence_threshold,
        padding_ratio=padding_ratio,
        max_detections=max_detections,
    )

    outputs = []
    for idx, detection in enumerate(detections):
        person_dir = output_dir / "aligned_person_outputs" / f"person_{idx}"
        aligned = save_aligner_output(
            aligner,
            Path(detection["crop_path"]),
            person_dir,
            f"person_{idx}",
            transform=transform,
            device=device,
            sapiens_model=sapiens_model,
        )
        outputs.append({**detection, **aligned})
    return outputs


def clean_output_dir(allowed_root, output_dir):
    output_dir = output_dir.resolve()
    allowed_root = allowed_root.resolve()
    if output_dir == allowed_root or allowed_root not in output_dir.parents:
        raise RuntimeError(f"Refusing to clean output outside workspace: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def compare_embeddings(outputs, output_dir):
    items = []
    for item in outputs:
        embedding = item.get("_embedding_tensor")
        if embedding is None:
            continue
        items.append(
            {
                "label": item["label"],
                "crop_path": item["input_image"],
                "embedding_path": item["step_4_sapiens_embedding"]["embedding_path"],
                "embedding": embedding.squeeze(0).float(),
            }
        )

    comparisons = []
    if len(items) >= 2:
        stacked = torch.stack([item["embedding"] for item in items], dim=0)
        normalized = torch.nn.functional.normalize(stacked, p=2, dim=1)
        similarity_matrix = normalized @ normalized.T

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                cosine_similarity = float(similarity_matrix[i, j])
                comparisons.append(
                    {
                        "person_a": items[i]["label"],
                        "person_b": items[j]["label"],
                        "cosine_similarity": round(cosine_similarity, 6),
                        "distance": round(1.0 - cosine_similarity, 6),
                        "note": "Higher similarity means more likely same identity.",
                    }
                )
    else:
        similarity_matrix = torch.empty((len(items), len(items)))

    report = {
        "method": "cosine similarity on L2-normalized SapiensID embeddings",
        "num_embeddings": len(items),
        "embedding_dim": int(items[0]["embedding"].numel()) if items else 0,
        "labels": [item["label"] for item in items],
        "similarity_matrix": [
            [round(float(v), 6) for v in row]
            for row in similarity_matrix.tolist()
        ],
        "comparisons": comparisons,
    }

    compare_path = output_dir / "embedding_comparisons.json"
    compare_path.write_text(json.dumps(report, indent=2))

    for item in outputs:
        item.pop("_embedding_tensor", None)

    return report, compare_path


def safe_name(name):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned.strip("_") or "identity"


def list_database_images(database_dir):
    if not database_dir.exists():
        return []
    return sorted(
        [
            path
            for path in database_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )


def list_input_images(input_path):
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMAGE_EXTENSIONS else []
    if not input_path.exists():
        return []
    return sorted(
        [
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: str(path).lower(),
    )


def default_db_index_path(repo_root, ckpt_dir):
    ckpt_name = ckpt_dir.name if ckpt_dir else "unknown_ckpt"
    return repo_root / ".cache" / "database_indexes" / f"{safe_name(ckpt_name)}_database_index.pt"


def embed_database_image(
    repo_root,
    aligner,
    sapiens_model,
    transform,
    device,
    image_path,
    artifact_root,
    confidence_threshold,
    padding_ratio,
):
    label = image_path.stem
    identity_dir = artifact_root / safe_name(label)
    crop_root = identity_dir / "webbody"
    detections = crop_persons_webbody_style(
        image_path=image_path,
        yolo_path=repo_root / "tasks" / "WebBody" / "c_detect" / "yolov8n-pose.pt",
        output_dir=crop_root,
        confidence_threshold=confidence_threshold,
        padding_ratio=padding_ratio,
        max_detections=1,
    )

    used_image_path = Path(detections[0]["crop_path"]) if detections else image_path
    aligned = save_aligner_output(
        aligner,
        used_image_path,
        identity_dir / "aligned_output",
        label,
        transform=transform,
        device=device,
        sapiens_model=sapiens_model,
    )
    return {
        "label": label,
        "source_image": str(image_path),
        "used_image": str(used_image_path),
        "detection": detections[0] if detections else None,
        "embedding_path": aligned["step_4_sapiens_embedding"]["embedding_path"],
        "embedding": aligned["_embedding_tensor"].squeeze(0).float(),
    }


def load_or_build_database_index(
    repo_root,
    aligner,
    sapiens_model,
    transform,
    device,
    database_dir,
    db_index_path,
    output_dir,
    refresh_db,
    confidence_threshold,
    padding_ratio,
):
    db_index_path.parent.mkdir(parents=True, exist_ok=True)
    if db_index_path.exists() and not refresh_db:
        index = torch.load(db_index_path, map_location="cpu")
        index["status"] = "loaded"
        return index

    database_images = list_database_images(database_dir)
    artifact_root = output_dir / "database_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    entries = []
    for image_path in database_images:
        entries.append(
            embed_database_image(
                repo_root=repo_root,
                aligner=aligner,
                sapiens_model=sapiens_model,
                transform=transform,
                device=device,
                image_path=image_path,
                artifact_root=artifact_root,
                confidence_threshold=confidence_threshold,
                padding_ratio=padding_ratio,
            )
        )

    embeddings = (
        torch.stack([entry.pop("embedding") for entry in entries], dim=0)
        if entries
        else torch.empty((0, 0))
    )
    index = {
        "status": "built",
        "database_dir": str(database_dir),
        "db_index_path": str(db_index_path),
        "num_database_images": len(database_images),
        "num_embeddings": len(entries),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.numel() > 0 else 0,
        "entries": entries,
        "embeddings": embeddings,
    }
    torch.save(index, db_index_path)
    return index


def match_outputs_to_database(query_outputs, database_index, output_dir, top_k, match_threshold):
    db_embeddings = database_index.get("embeddings", torch.empty((0, 0))).float()
    query_items = []
    for item in query_outputs:
        embedding = item.get("_embedding_tensor")
        if embedding is None:
            continue
        query_items.append(
            {
                "label": item["label"],
                "input_image": item["input_image"],
                "embedding_path": item["step_4_sapiens_embedding"]["embedding_path"],
                "embedding": embedding.squeeze(0).float(),
            }
        )

    if not query_items or db_embeddings.numel() == 0:
        report = {
            "status": "skipped",
            "reason": "Need at least one input embedding and one database embedding.",
            "num_queries": len(query_items),
            "num_database_embeddings": int(db_embeddings.shape[0]) if db_embeddings.ndim == 2 else 0,
            "matches": [],
        }
    else:
        query_embeddings = torch.stack([item["embedding"] for item in query_items], dim=0)
        query_norm = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        db_norm = torch.nn.functional.normalize(db_embeddings, p=2, dim=1)
        scores = query_norm @ db_norm.T
        k = min(top_k, db_norm.shape[0])
        top_scores, top_indices = torch.topk(scores, k=k, dim=1)

        matches = []
        for query_idx, query in enumerate(query_items):
            top_matches = []
            for score, db_idx in zip(top_scores[query_idx].tolist(), top_indices[query_idx].tolist()):
                db_entry = database_index["entries"][db_idx]
                top_matches.append(
                    {
                        "label": db_entry["label"],
                        "score": round(float(score), 6),
                        "source_image": db_entry["source_image"],
                        "used_image": db_entry["used_image"],
                        "embedding_path": db_entry["embedding_path"],
                    }
                )
            best_match = top_matches[0] if top_matches else None
            is_known = bool(
                best_match and best_match["score"] > match_threshold
            )
            recognized_identity = best_match["label"] if is_known else None
            matches.append(
                {
                    "query_label": query["label"],
                    "query_image": query["input_image"],
                    "query_embedding_path": query["embedding_path"],
                    "best_match": best_match,
                    "recognition_status": "KNOWN" if is_known else "UNKNOWN",
                    "recognized_identity": recognized_identity,
                    "decision": recognized_identity if is_known else "UNKNOWN",
                    "match_threshold": match_threshold,
                    "top_matches": top_matches,
                }
            )

        report = {
            "status": "saved",
            "method": "cosine similarity on L2-normalized SapiensID embeddings",
            "decision_rule": "KNOWN when best similarity > match_threshold; otherwise UNKNOWN",
            "top_k": k,
            "match_threshold": match_threshold,
            "num_queries": len(query_items),
            "num_database_embeddings": int(db_embeddings.shape[0]),
            "matches": matches,
        }

    match_path = output_dir / "database_match_results.json"
    match_path.write_text(json.dumps(report, indent=2))
    return report, match_path


def process_image_once(
    repo_root,
    aligner,
    sapiens_model,
    transform,
    device,
    image_path,
    output_dir,
    args,
    database_index=None,
):
    image_summary = {
        "input_image": str(image_path),
        "outputs": {},
    }

    if args.mode in ["direct", "both"]:
        image_summary["outputs"]["direct"] = run_direct_mode(
            aligner,
            image_path,
            output_dir,
            transform=transform,
            device=device,
            sapiens_model=sapiens_model,
        )

    if args.mode in ["webbody", "both"]:
        image_summary["outputs"]["webbody"] = run_webbody_mode(
            repo_root=repo_root,
            aligner=aligner,
            image_path=image_path,
            output_dir=output_dir / "webbody",
            confidence_threshold=args.confidence_threshold,
            padding_ratio=args.padding_ratio,
            max_detections=args.max_detections,
            transform=transform,
            device=device,
            sapiens_model=sapiens_model,
        )

    if sapiens_model is None:
        image_summary["step_5_identity_embeddings"] = {"status": "skipped"}
        image_summary["step_6_embedding_comparison"] = {"status": "skipped"}
        image_summary["step_7_database_match"] = {"status": "skipped"}
        return image_summary

    outputs_for_comparison = []
    for output_items in image_summary["outputs"].values():
        outputs_for_comparison.extend(output_items)

    if args.skip_db_match or database_index is None:
        image_summary["step_7_database_match"] = {
            "status": "skipped_by_user" if args.skip_db_match else "skipped",
        }
    else:
        match_report, match_path = match_outputs_to_database(
            outputs_for_comparison,
            database_index,
            output_dir,
            top_k=args.top_k,
            match_threshold=args.match_threshold,
        )
        image_summary["step_7_database_match"] = {
            "status": match_report["status"],
            "match_threshold": match_report.get("match_threshold"),
            "match_path": str(match_path),
            "matches": match_report["matches"],
        }

    comparison_report, comparison_path = compare_embeddings(outputs_for_comparison, output_dir)
    image_summary["step_5_identity_embeddings"] = {
        "status": "saved",
        "count": comparison_report["num_embeddings"],
        "embedding_dim": comparison_report["embedding_dim"],
    }
    image_summary["step_6_embedding_comparison"] = {
        "status": "saved",
        "comparison_path": str(comparison_path),
        "comparisons": comparison_report["comparisons"],
    }
    return image_summary


def run_batch_mode(
    repo_root,
    aligner,
    sapiens_model,
    transform,
    device,
    image_dir,
    output_dir,
    args,
    database_index=None,
    expected_label="Johnny Depp",
):
    image_files = list_input_images(image_dir)
    batch_root = output_dir / "batch"
    batch_root.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []
    for image_idx, image_file in enumerate(image_files, start=1):
        per_image_output = batch_root / f"{image_idx:03d}_{safe_name(image_file.stem)}"
        per_image_output.mkdir(parents=True, exist_ok=True)
        processing_error = None
        try:
            image_summary = process_image_once(
                repo_root=repo_root,
                aligner=aligner,
                sapiens_model=sapiens_model,
                transform=transform,
                device=device,
                image_path=image_file,
                output_dir=per_image_output,
                args=args,
                database_index=database_index,
            )
        except (IndexError, RuntimeError, ValueError, OSError) as exc:
            # A blurred, occluded, or very small crop can make the upstream
            # aligner return no usable keypoints. Record it as Unresolved and
            # continue the batch instead of losing all completed work.
            processing_error = f"{type(exc).__name__}: {exc}"
            image_summary = {
                "input_image": str(image_file),
                "status": "unresolved_processing_error",
                "error": processing_error,
                "outputs": {},
                "step_7_database_match": {"status": "unresolved", "matches": []},
            }
        image_summary_path = per_image_output / "summary.json"
        image_summary_path.write_text(json.dumps(image_summary, indent=2))

        matches = image_summary.get("step_7_database_match", {}).get("matches", [])
        image_failures = []
        if processing_error:
            image_failures.append(
                {
                    "query_label": None,
                    "decision": "UNRESOLVED",
                    "recognition_status": "UNRESOLVED",
                    "recognized_identity": None,
                    "best_candidate": None,
                    "score": None,
                    "error": processing_error,
                }
            )
        elif not matches:
            image_failures.append(
                {
                    "query_label": None,
                    "decision": "NO_MATCH",
                    "recognition_status": "UNKNOWN",
                    "recognized_identity": None,
                    "best_candidate": None,
                    "score": None,
                }
            )
        for match in matches:
            best_match = match.get("best_match") or {}
            recognized_identity = match.get("recognized_identity")
            if recognized_identity != expected_label:
                image_failures.append(
                    {
                        "query_label": match.get("query_label"),
                        "decision": match.get("decision"),
                        "recognition_status": match.get("recognition_status"),
                        "recognized_identity": recognized_identity,
                        "best_candidate": best_match.get("label"),
                        "score": best_match.get("score"),
                    }
                )

        image_result = {
            "input_image": str(image_file),
            "summary_path": str(image_summary_path),
            "num_detected_people": len(matches),
            "passed": not image_failures,
            "failures": image_failures,
        }
        results.append(image_result)
        failures.extend([{**failure, "input_image": str(image_file)} for failure in image_failures])
        print(f"[{image_idx}/{len(image_files)}] {image_file.name}: {'PASS' if image_result['passed'] else 'FAIL'}")

    report = {
        "status": "saved",
        "input_dir": str(image_dir),
        "expected_label": expected_label,
        "match_threshold": args.match_threshold,
        "top_k": args.top_k,
        "num_images": len(image_files),
        "passed_images": sum(1 for result in results if result["passed"]),
        "failed_images": sum(1 for result in results if not result["passed"]),
        "failures": failures,
        "results": results,
    }
    report_path = output_dir / "batch_match_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report, report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="input/input image.jpg")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument(
        "--mode",
        choices=["webbody", "direct", "both"],
        default="webbody",
        help="webbody crops each detected person first; direct runs aligner on the original image.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.30,
        help="YOLO person confidence threshold. 0.30 works better for crowded scenes.",
    )
    parser.add_argument("--padding-ratio", type=float, default=0.05)
    parser.add_argument("--max-detections", type=int, default=6)
    parser.add_argument(
        "--ckpt-dir",
        default="",
        help="Folder containing SapiensID model.yaml and model.pth. Default: checkpoints/sapiensid_wb4m",
    )
    parser.add_argument("--skip-model", action="store_true", help="Run only detection/cropping/aligner and skip SapiensID embedding.")
    parser.add_argument("--no-clean-output", action="store_true", help="Do not delete the output folder before running.")
    parser.add_argument("--database-dir", default="Database", help="Folder of flat identity images, e.g. Database/ayush.jpg.")
    parser.add_argument("--db-index-path", default="", help="Saved database embedding index. Default is .cache/database_indexes/<checkpoint>_database_index.pt.")
    parser.add_argument("--refresh-db", action="store_true", help="Rebuild database embeddings even if a saved index already exists.")
    parser.add_argument("--skip-db-match", action="store_true", help="Do not match input embeddings against the database.")
    parser.add_argument("--top-k", type=int, default=1, help="Number of database matches to return per detected input person.")
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.33,
        help="A best cosine similarity strictly above this value is KNOWN; otherwise it is UNKNOWN.",
    )
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    repo_root = workspace_root / "sapiensid"
    prepare_local_caches(repo_root)

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = workspace_root / image_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = workspace_root / output_dir
    if args.no_clean_output:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        clean_output_dir(workspace_root, output_dir)

    device = get_runtime_device()
    aligner = make_aligner(repo_root)
    aligner = aligner.to(device)
    ckpt_dir = resolve_checkpoint_dir(repo_root, args.ckpt_dir)
    database_dir = Path(args.database_dir)
    if not database_dir.is_absolute():
        database_dir = workspace_root / database_dir
    db_index_path = Path(args.db_index_path) if args.db_index_path else default_db_index_path(repo_root, ckpt_dir)
    if not db_index_path.is_absolute():
        db_index_path = repo_root / db_index_path

    checkpoint_missing = check_checkpoint(ckpt_dir)
    sapiens_model = None
    transform = make_default_transform()
    step_4_status = "skipped_by_user"
    if not args.skip_model:
        if checkpoint_missing:
            step_4_status = "skipped_missing_checkpoint"
        else:
            sapiens_model = make_sapiens_model(repo_root, ckpt_dir, device)
            transform = sapiens_model.make_test_transform()
            step_4_status = "enabled"

    if image_path.is_dir():
        database_index = None
        database_index_status = "skipped"
        if sapiens_model is not None and not args.skip_db_match:
            database_index = load_or_build_database_index(
                repo_root=repo_root,
                aligner=aligner,
                sapiens_model=sapiens_model,
                transform=transform,
                device=device,
                database_dir=database_dir,
                db_index_path=db_index_path,
                output_dir=output_dir,
                refresh_db=args.refresh_db,
                confidence_threshold=args.confidence_threshold,
                padding_ratio=args.padding_ratio,
            )
            database_index_status = database_index["status"]

        batch_report, batch_report_path = run_batch_mode(
            repo_root=repo_root,
            aligner=aligner,
            sapiens_model=sapiens_model,
            transform=transform,
            device=device,
            image_dir=image_path,
            output_dir=output_dir,
            args=args,
            database_index=database_index,
            expected_label=image_path.name,
        )
        summary = {
            "input_folder": str(image_path),
            "mode": args.mode,
            "torch_cuda_available": torch.cuda.is_available(),
            "runtime_device": str(device),
            "step_4_sapiens_model": {
                "status": step_4_status,
                "ckpt_dir": str(ckpt_dir),
                "missing_files": checkpoint_missing,
            },
            "database": {
                "database_dir": str(database_dir),
                "db_index_path": str(db_index_path),
                "refresh_db": args.refresh_db,
                "database_index_status": database_index_status,
                "database_embeddings": database_index["num_embeddings"] if database_index is not None else 0,
                "match_threshold": args.match_threshold,
            },
            "batch_report_path": str(batch_report_path),
            "batch": batch_report,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        print(f"Input folder: {image_path}")
        print(f"Saved output folder: {output_dir}")
        print(f"Saved batch report: {batch_report_path}")
        print(f"Expected label: {batch_report['expected_label']}")
        print(f"Match threshold: {batch_report['match_threshold']}")
        print(f"Images passed: {batch_report['passed_images']}/{batch_report['num_images']}")
        print(f"Images failed: {batch_report['failed_images']}/{batch_report['num_images']}")
        for failure in batch_report["failures"]:
            print(
                f"FAIL {failure['input_image']}: decision={failure['decision']} "
                f"best={failure['best_candidate']} score={failure['score']}"
            )
        return

    summary = {
        "input_image": str(image_path),
        "mode": args.mode,
        "torch_cuda_available": torch.cuda.is_available(),
        "runtime_device": str(device),
        "step_4_sapiens_model": {
            "status": step_4_status,
            "ckpt_dir": str(ckpt_dir),
            "missing_files": checkpoint_missing,
        },
        "database": {
            "database_dir": str(database_dir),
            "db_index_path": str(db_index_path),
            "refresh_db": args.refresh_db,
            "match_threshold": args.match_threshold,
        },
        "outputs": {},
    }

    if args.mode in ["direct", "both"]:
        summary["outputs"]["direct"] = run_direct_mode(
            aligner,
            image_path,
            output_dir,
            transform=transform,
            device=device,
            sapiens_model=sapiens_model,
        )

    if args.mode in ["webbody", "both"]:
        summary["outputs"]["webbody"] = run_webbody_mode(
            repo_root=repo_root,
            aligner=aligner,
            image_path=image_path,
            output_dir=output_dir / "webbody",
            confidence_threshold=args.confidence_threshold,
            padding_ratio=args.padding_ratio,
            max_detections=args.max_detections,
            transform=transform,
            device=device,
            sapiens_model=sapiens_model,
        )

    if sapiens_model is not None:
        outputs_for_comparison = []
        for output_items in summary["outputs"].values():
            outputs_for_comparison.extend(output_items)

        if args.skip_db_match:
            summary["step_7_database_match"] = {
                "status": "skipped_by_user",
            }
        else:
            database_index = load_or_build_database_index(
                repo_root=repo_root,
                aligner=aligner,
                sapiens_model=sapiens_model,
                transform=transform,
                device=device,
                database_dir=database_dir,
                db_index_path=db_index_path,
                output_dir=output_dir,
                refresh_db=args.refresh_db,
                confidence_threshold=args.confidence_threshold,
                padding_ratio=args.padding_ratio,
            )
            match_report, match_path = match_outputs_to_database(
                outputs_for_comparison,
                database_index,
                output_dir,
                top_k=args.top_k,
                match_threshold=args.match_threshold,
            )
            summary["step_7_database_match"] = {
                "status": match_report["status"],
                "database_index_status": database_index["status"],
                "database_embeddings": database_index["num_embeddings"],
                "match_threshold": match_report.get("match_threshold"),
                "match_path": str(match_path),
                "matches": match_report["matches"],
            }

        comparison_report, comparison_path = compare_embeddings(outputs_for_comparison, output_dir)
        summary["step_5_identity_embeddings"] = {
            "status": "saved",
            "count": comparison_report["num_embeddings"],
            "embedding_dim": comparison_report["embedding_dim"],
        }
        summary["step_6_embedding_comparison"] = {
            "status": "saved",
            "comparison_path": str(comparison_path),
            "comparisons": comparison_report["comparisons"],
        }
    else:
        summary["step_5_identity_embeddings"] = {
            "status": "skipped",
            "reason": step_4_status,
        }
        summary["step_6_embedding_comparison"] = {
            "status": "skipped",
            "reason": "No SapiensID embeddings were generated.",
        }
        summary["step_7_database_match"] = {
            "status": "skipped",
            "reason": "No SapiensID embeddings were generated.",
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Input image: {image_path}")
    print(f"Mode: {args.mode}")
    print(f"Saved output folder: {output_dir}")
    print(f"Saved summary: {summary_path}")
    print(f"Step 4 SapiensID model: {step_4_status}")
    print(f"Step 5 identity embeddings: {summary['step_5_identity_embeddings']['status']}")
    print(f"Step 6 embedding comparison: {summary['step_6_embedding_comparison']['status']}")
    print(f"Step 7 database match: {summary['step_7_database_match']['status']}")
    if summary["step_7_database_match"]["status"] == "saved":
        for match in summary["step_7_database_match"]["matches"]:
            best_match = match["best_match"]
            if best_match:
                print(
                    f"{match['query_label']} status: {match['recognition_status']} "
                    f"identity: {match['recognized_identity']} "
                    f"(best candidate: {best_match['label']} score={best_match['score']}, "
                    f"threshold={match['match_threshold']})"
                )
    if summary["step_6_embedding_comparison"]["status"] == "saved":
        for comparison in summary["step_6_embedding_comparison"]["comparisons"]:
            print(
                f"{comparison['person_a']} vs {comparison['person_b']}: "
                f"cosine_similarity={comparison['cosine_similarity']}"
            )
    if checkpoint_missing:
        print("Missing checkpoint files:")
        for missing in checkpoint_missing:
            print(f"  {missing}")
    if "webbody" in summary["outputs"]:
        print(f"WebBody-style persons processed: {len(summary['outputs']['webbody'])}")
    if "direct" in summary["outputs"]:
        print("Direct aligner processed original image once")


if __name__ == "__main__":
    main()
