"""Prompt material bundle for one candidate setting — all derived from train.

Filled by materialize (loop). The candidate's axes decide which fields are
active; inactive axes are empty containers (= exactly the incumbent, a safe point).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..core.prompts import PromptBox


@dataclass(frozen=True)
class Materials:
    exemplars: dict[str, list[PromptBox]] = field(default_factory=dict)
    visual_negatives: dict[str, list[PromptBox]] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)          # final merged map (manual entries win)
    text_negatives: dict[str, list[str]] = field(default_factory=dict)

    def fingerprint(self) -> dict:
        """Content fingerprint built from (image filename, bbox, text), not paths,
        so test_materialize_purity can compare across different out_dirs."""
        def box_key(b: PromptBox):
            return [Path(b.image).name, [round(v, 3) for v in b.bbox_xyxy]]
        return {
            "exemplars": {n: [box_key(b) for b in bs]
                          for n, bs in sorted(self.exemplars.items())},
            "visual_negatives": {n: [box_key(b) for b in bs]
                          for n, bs in sorted(self.visual_negatives.items())},
            "aliases": dict(sorted(self.aliases.items())),
            "text_negatives": {n: sorted(v)
                               for n, v in sorted(self.text_negatives.items())},
        }

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        def dump(boxes_by_name):
            return {n: [{"image": b.image, "bbox_xyxy": b.bbox_xyxy} for b in bs]
                    for n, bs in boxes_by_name.items()}
        (out_dir / "exemplars.json").write_text(
            json.dumps(dump(self.exemplars), indent=2, ensure_ascii=False))
        (out_dir / "visual_negatives.json").write_text(
            json.dumps(dump(self.visual_negatives), indent=2, ensure_ascii=False))
        (out_dir / "text_aliases.json").write_text(
            json.dumps(self.aliases, indent=2, ensure_ascii=False))
        (out_dir / "text_negatives.json").write_text(
            json.dumps(self.text_negatives, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, out_dir: Path) -> "Materials":
        def boxes(path):
            data = json.loads(path.read_text())
            return {n: [PromptBox(class_name=n, image=e["image"],
                                  bbox_xyxy=e["bbox_xyxy"]) for e in es]
                    for n, es in data.items()}
        out_dir = Path(out_dir)
        return cls(
            exemplars=boxes(out_dir / "exemplars.json"),
            visual_negatives=boxes(out_dir / "visual_negatives.json"),
            aliases=json.loads((out_dir / "text_aliases.json").read_text()),
            text_negatives=json.loads((out_dir / "text_negatives.json").read_text()),
        )
