from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loaders import discover_lmvd_training_sample_ids, load_lmvd_labels
from src.data.trimodal_dataset import LMVDTrimodalDataset, trimodal_text_dim
from src.evaluation.metrics import results_to_dataframe
from src.models.fusion import build_fusion_model
from src.models.image_cnn import ImageEncoder, build_image_model
from src.models.text_dl import build_text_dl_model
from src.models.trimodal import build_trimodal_model
from src.models.video_lstm import VideoAudioLSTM, build_video_model
from src.training.dl_utils import class_weights_from_loader, evaluate_model, forward_image, forward_text, forward_trimodal, forward_video, train_epoch
from src.utils import ensure_dir, load_config, set_seed


def train_with_best_checkpoint(model, train_loader, val_loader, forward_fn, epochs, lr, device, selection_metric="f1_macro"):
    import copy

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weight = class_weights_from_loader(train_loader).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    for _ in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, device, forward_fn)
        metrics = evaluate_model(model, val_loader, device, forward_fn)
        score = metrics.get(selection_metric, metrics.get("accuracy", 0.0))
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


class BimodalFusionNet(nn.Module):

    def __init__(
        self,
        modalities: tuple[str, str],
        text_dim: int,
        image_architecture: str,
        embed_dim: int,
        fusion_method: str,
        video_dim: int = 171,
        audio_dim: int = 128,
        video_hidden: int = 128,
        video_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.modalities = modalities
        encoders = {}
        if "text" in modalities:
            encoders["text"] = nn.Sequential(
                nn.Linear(text_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            )
        if "image" in modalities:
            encoders["image"] = ImageEncoder(image_architecture, embed_dim=embed_dim)
        if "video" in modalities:
            backbone = VideoAudioLSTM(
                video_dim=video_dim, audio_dim=audio_dim, hidden_dim=video_hidden,
                num_layers=video_layers, num_classes=num_classes, dropout=dropout,
            )
            proj = nn.Sequential(
                nn.Linear(backbone.video_encoder.out_dim + backbone.audio_encoder.out_dim, embed_dim),
                nn.ReLU(),
            )
            encoders["video"] = nn.ModuleDict({"backbone": backbone, "proj": proj})
        self.encoders = nn.ModuleDict(encoders)
        self.fusion = build_fusion_model(fusion_method, [embed_dim] * len(modalities), hidden_dim=embed_dim * 2, num_classes=num_classes)

    def _encode(self, modality: str, batch: dict, device: str) -> torch.Tensor:
        if modality == "text":
            return self.encoders["text"](batch["text"].to(device))
        if modality == "image":
            return self.encoders["image"](batch["image"].to(device))
        video_enc = self.encoders["video"]
        feat = video_enc["backbone"].encode(batch["video"].to(device), batch["audio"].to(device))
        return video_enc["proj"](feat)

    def forward_batch(self, batch: dict, device: str) -> torch.Tensor:
        embeddings = [self._encode(m, batch, device) for m in self.modalities]
        return self.fusion(*embeddings)


def _bimodal_forward_fn(model: BimodalFusionNet, batch: dict, device: str) -> torch.Tensor:
    return model.forward_batch(batch, device)


def _stratified_head(df, n):
    per_class = max(1, n // 2)
    return df.groupby("label", group_keys=False).apply(lambda g: g.head(per_class)).head(n)


def _split_loaders(labels_df, data_dir, image_root, image_transform, batch_size, max_train, max_val, max_test):
    from scripts.extract_lmvd_modalities import extract_and_cache_samples

    loaders = {}
    for split_name, cap in (("train", max_train), ("val", max_val), ("test", max_test)):
        subset = _stratified_head(labels_df[labels_df["split"] == split_name], cap)
        print(f"  [data] extracting/caching {len(subset)} {split_name} samples from the LMVD zip ...", flush=True)
        cached_ids = extract_and_cache_samples(subset["sample_id"].tolist(), data_dir)
        subset = subset[subset["sample_id"].isin(cached_ids)]
        print(f"  [data] {split_name}: {len(subset)} samples ready "
              f"(label balance: {subset['label'].value_counts().to_dict()})")
        ds = LMVDTrimodalDataset(
            subset["sample_id"].tolist(), subset["label"].tolist(), data_dir,
            image_root=image_root, image_transform=image_transform, cache_only=True,
        )
        loaders[split_name] = DataLoader(ds, batch_size=batch_size, shuffle=(split_name == "train"))
    return loaders


def _run_group(name, architectures, build_fn, forward_fn, loaders, epochs, lr, device, results):
    for arch in architectures:
        print(f"  [{name}] training {arch} ...", flush=True)
        model = build_fn(arch)
        model = train_with_best_checkpoint(
            model, loaders["train"], loaders["val"], forward_fn,
            epochs=epochs, lr=lr, device=device, selection_metric="f1_macro",
        )
        test_metrics = evaluate_model(model, loaders["test"], device, forward_fn)
        results.append({"modality": name, "model": arch, "task": "classification", "test": test_metrics})
        print(f"  [{name}] {arch}: {test_metrics}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trimodal (text+image+video) LMVD experiments.")
    parser.add_argument("--fast", action="store_true", help="Fewer epochs / smaller sample caps for a quick smoke run.")
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--skip-bimodal", action="store_true")
    parser.add_argument("--skip-trimodal", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    config = load_config(ROOT / "config.yaml")
    data_dir = ROOT / "lmvd-analysis" / "data"
    results_dir = ensure_dir(ROOT / "outputs" / "results")

    zip_path = data_dir / "LMVD_Feature.zip"
    has_zip = zip_path.exists() and zip_path.stat().st_size > 15_000_000_000
    has_cache = bool(discover_lmvd_training_sample_ids(data_dir, min_cached=config["video_models"].get("min_cached_npy", 100)))
    if not (has_zip or has_cache):
        print("[skip] No LMVD_Feature.zip and no cached features found - trimodal experiments need "
              "lmvd-analysis/download_lmvd_features.py to have been run first (~16GB download). "
              "trimodal_results.csv was not written.")
        return

    labels_df = load_lmvd_labels(data_dir)
    print(f"[info] {len(labels_df)} labeled LMVD samples available across splits: "
          f"{labels_df['split'].value_counts().to_dict()} (features extracted on demand from the zip)")

    tri_cfg = config["trimodal"]
    vid_cfg = config["video_models"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = 2 if args.fast else tri_cfg["epochs"]
    max_train = min(150, tri_cfg["max_train_samples"]) if args.fast else tri_cfg["max_train_samples"]
    max_val = min(40, tri_cfg["max_val_samples"]) if args.fast else tri_cfg["max_val_samples"]
    max_test = max_val

    image_size = config["image_models"]["image_size"]
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    lmvd_image_root = ROOT / config["paths"].get("lmvd_image_dir", "datasets/lmvd_extracted/images")

    loaders = _split_loaders(
        labels_df, data_dir, lmvd_image_root, image_transform,
        batch_size=tri_cfg["batch_size"], max_train=max_train, max_val=max_val, max_test=max_test,
    )

    results: list[dict] = []
    text_dim = trimodal_text_dim()

    if not args.skip_text:
        _run_group(
            "text_dl", config["text_dl_models"],
            lambda arch: build_text_dl_model(arch, input_dim=text_dim),
            forward_text, loaders, epochs, tri_cfg["learning_rate"], device, results,
        )

    if not args.skip_image:
        _run_group(
            "image_dl", config["image_models"]["architectures"],
            lambda arch: build_image_model(arch, num_classes=2, pretrained=(arch != "cnn")),
            forward_image, loaders, epochs, tri_cfg["learning_rate"], device, results,
        )

    if not args.skip_video:
        _run_group(
            "video_dl", vid_cfg["architectures"],
            lambda arch: build_video_model(
                arch, video_dim=vid_cfg["video_feature_dim"], audio_dim=vid_cfg["audio_feature_dim"],
                hidden_dim=vid_cfg["lstm_hidden"], num_layers=vid_cfg["lstm_layers"], dropout=vid_cfg["dropout"],
            ),
            forward_video, loaders, epochs, tri_cfg["learning_rate"], device, results,
        )

    if not args.skip_bimodal:
        # One fusion method per pair (attention - the strongest of the three in
        # the trimodal results) to keep the pairwise sweep affordable; all three
        # methods are still exercised individually via the trimodal fusion below.
        bimodal_method = "attention"
        for pair in (("text", "video"), ("image", "video"), ("text", "image")):
            model_name = "+".join(pair)
            print(f"  [bimodal] training {model_name} ({bimodal_method}) ...", flush=True)
            model = BimodalFusionNet(
                pair, text_dim, tri_cfg["image_architecture"], tri_cfg["embed_dim"], bimodal_method,
                video_dim=vid_cfg["video_feature_dim"], audio_dim=vid_cfg["audio_feature_dim"],
                video_hidden=vid_cfg["lstm_hidden"], video_layers=vid_cfg["lstm_layers"],
            )
            model = train_with_best_checkpoint(
                model, loaders["train"], loaders["val"], _bimodal_forward_fn,
                epochs=epochs, lr=tri_cfg["learning_rate"], device=device, selection_metric="f1_macro",
            )
            test_metrics = evaluate_model(model, loaders["test"], device, _bimodal_forward_fn)
            results.append({"modality": "bimodal", "model": model_name, "task": "classification", "test": test_metrics})
            print(f"  [bimodal] {model_name}: {test_metrics}")

    if not args.skip_trimodal:
        for method in tri_cfg["fusion_methods"]:
            model_name = f"text+image+video_{method}"
            print(f"  [trimodal] training {model_name} ...", flush=True)
            model = build_trimodal_model(
                method, text_dim, image_architecture=tri_cfg["image_architecture"],
                video_dim=vid_cfg["video_feature_dim"], audio_dim=vid_cfg["audio_feature_dim"],
                video_hidden=vid_cfg["lstm_hidden"], video_layers=vid_cfg["lstm_layers"],
                embed_dim=tri_cfg["embed_dim"],
            )
            model = train_with_best_checkpoint(
                model, loaders["train"], loaders["val"], forward_trimodal,
                epochs=epochs, lr=tri_cfg["learning_rate"], device=device, selection_metric="f1_macro",
            )
            test_metrics = evaluate_model(model, loaders["test"], device, forward_trimodal)
            results.append({"modality": "trimodal", "model": model_name, "task": "classification", "test": test_metrics})
            print(f"  [trimodal] {model_name}: {test_metrics}")

    if not results:
        print("[skip] Nothing ran (all groups skipped) - trimodal_results.csv not written.")
        return

    df = results_to_dataframe(results)
    df.to_csv(results_dir / "trimodal_results.csv", index=False)
    with (results_dir / "trimodal_full_results.json").open("w", encoding="utf-8") as handle:
        json.dump({"results": results}, handle, indent=2, default=str)
    print(f"[ok] Wrote {len(df)} trimodal result rows -> {results_dir / 'trimodal_results.csv'}")


if __name__ == "__main__":
    main()
