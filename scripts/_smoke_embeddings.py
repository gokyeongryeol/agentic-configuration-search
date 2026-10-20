"""GPU smoke: embedding capture on real detections (dim, alignment) and vocab reuse on repeat calls."""
import sys, glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_configuration_search.clients.yoloe import YoloeClient
from agentic_configuration_search.core.prompts import build_text_queries

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:5"
imgs = sorted(glob.glob("demo_data/soccer-players-5fuqs/train/*.jpg"))[:4]
client = YoloeClient(model_path="weights_yoloe/yoloe-26x-seg.pt", device=DEV)
queries, _o, _a = build_text_queries(
    ["person", "sports ball"], {}, box_threshold=0.001,
    multimodal_threshold=0.001, negatives={"person": ["shadow on grass"]})

preds = client.predict_images(imgs, queries, neg_eta=0.5, apply_topk=300,
                              iou_threshold=0.5, return_embeddings=True)
dims = {len(e) for p in preds for e in p["embeddings"]}
print("dets:", [len(p["boxes"]) for p in preds], "emb dims:", dims)
assert dims == {512}, dims
e0 = preds[0]["embeddings"][0]
print("emb sample:", [round(v, 3) for v in e0[:6]], "norm-ish:",
      round(sum(v * v for v in e0) ** 0.5, 2))

# repeat the same query — embeddings must also come out on the vocab-cache reuse path
preds2 = client.predict_images(imgs[:1], queries, neg_eta=0.5,
                               return_embeddings=True)
assert preds2[0]["embeddings"] and len(preds2[0]["embeddings"][0]) == 512
print("second call ok:", len(preds2[0]["boxes"]))
print("EMBEDDINGS OK")
