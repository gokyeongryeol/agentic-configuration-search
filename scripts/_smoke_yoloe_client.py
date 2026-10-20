"""GPU smoke: YoloeClient runs real inference per the gateway contract (a few soccer images)."""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_configuration_search.clients.yoloe import YoloeClient
from agentic_configuration_search.config import config_from_dict
from agentic_configuration_search.core.data import load_dataset
from agentic_configuration_search.core.prompts import PromptBox
from agentic_configuration_search.infer import gateway
from agentic_configuration_search.infer.queries import build_config_queries, text_query_bundle

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:4"
cfg = config_from_dict({"dataset": "demo_data/soccer-players-5fuqs",
                        "detector": {"device": DEV},
                        "query": {"box_threshold": 0.001,
                                  "threshold_floor": 0.001}})
ds = load_dataset("demo_data/soccer-players-5fuqs", seed=0, fit_train_ratio=0.8)
onto = ds.ontology
paths = [ds.fit_dev.image_path(e) for e in ds.fit_dev.images][:4]
client = YoloeClient(model_path="weights_yoloe/yoloe-26x-seg.pt", device=DEV)


class Ctx:
    def __init__(self, cfg):
        self.cfg = cfg


# ---- text pass (with negatives, default suppression) ----
tq, torder, alias = text_query_bundle(cfg.query, onto,
                                      {"player": "person"},
                                      {"player": ["shadow on grass"]})
config = {"config_id": 0, "negative": False, "neg_eta": None, "tiling": None}
t0 = time.time()
dets = gateway.infer_config(Ctx(cfg), client, config, tq, onto, paths,
                            alias_to_name=alias, return_embeddings=True)
print(f"text pass {time.time()-t0:.1f}s:",
      [len(d) for d in dets], "emb sample:",
      dets[0][0].get("embedding") if dets[0] else None)
per = {}
for d in dets[0]:
    per[d["category_id"]] = per.get(d["category_id"], 0) + 1
print("  img0 by class:", per, "top:",
      max((d["score"] for d in dets[0]), default=0))

# ---- semantic pass (exemplars + visual negative, a neg_eta sweep value) ----
train = ds.train
ex, neg = {}, {}
for e in train.images[:3]:
    for a in e.annotations:
        name = onto.id_to_name[a.category_id]
        ex.setdefault(name, [])
        if len(ex[name]) < 2:
            ex[name].append(PromptBox(name, str(train.image_path(e)),
                                      list(a.bbox_xyxy)))
first = train.images[0]
neg = {"player": [PromptBox("player", str(train.image_path(first)),
                            [0, 0, 60, 40])]}
ex = {k: v for k, v in ex.items() if v}
print("exemplar classes:", {k: len(v) for k, v in ex.items()})
queries, order, excluded = build_config_queries(
    cfg.query, {"config_id": 1, "negative": True, "neg_eta": 0.5,
                "tiling": None}, ex, neg)
t0 = time.time()
dets = gateway.infer_config(Ctx(cfg), client,
                            {"config_id": 1, "negative": True, "neg_eta": 0.5,
                             "tiling": None},
                            queries, onto, paths)
print(f"semantic pass {time.time()-t0:.1f}s:", [len(d) for d in dets])
for d in dets[0][:5]:
    print("  ", onto.id_to_name[d["category_id"]], round(d["score"], 4),
          [round(v, 1) for v in d["box"]])

# ---- tiling path (PIL crop input) ----
tile_cfg = {"config_id": 2, "negative": False, "neg_eta": None,
            "tiling": {"grid": 3, "overlap": 0.5, "merge": "nmm"}}
q2, o2, _ = build_config_queries(cfg.query, tile_cfg, ex, None)
t0 = time.time()
dets = gateway.infer_config(Ctx(cfg), client, tile_cfg, q2, onto, paths[:2])
print(f"tiled pass {time.time()-t0:.1f}s:", [len(d) for d in dets])
print("OK")
