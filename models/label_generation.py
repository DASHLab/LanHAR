import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoTokenizer
from model import *

if "TEXT_PROTOS" not in globals():
    TEXT_PROTOS = {
        "walk": [
            "walking","walk","on foot","casual stroll","brisk walk",
            "steady steps","walking outdoors","commuting on foot",
            "normal walking pace","relaxed walking",
            "periodic vertical oscillation around 1 to 2 Hz",
            "repetitive up and down acceleration like footsteps",
            "moderate impact at each landing",
            "consistent step cycle with low variance",
            "small gyroscope sway from arm and body motion",
            "stable rhythmic pattern over time",
            "energy concentrated in low frequencies near step rate",
            "vertical axis dominates the periodic motion",
            "symmetric gait with regular intervals",
            "medium jerk during heel strike",
            "mild horizontal sway superimposed on vertical pattern",
            "smooth stride transitions without abrupt spikes",
        ],
        "still": [
            "sitting", "sit", "remain seated", "stay seated", "seated still",
            "stationary sitting", "idle while seated", "quiet sitting",
            "sitting and resting", "seated posture",
            "standing still", "stand upright", "remain standing", "stay standing",
            "motionless while standing", "standing without moving", "upright posture without motion",
            "acceleration is nearly constant close to gravity",
            "orientation remains stable over time",
            "very low motion energy with only small random noise",
            "gyroscope readings are almost flat",
            "no noticeable periodic fluctuation",
            "absence of step-like impacts",
            "minimal jerk over the entire duration",
            "power spectrum lacks dominant peaks",
            "flat and low-variance signal in all axes",
            "near-static state with negligible movement",
            ],

        "stairsup": [
            "walk upstairs","climb stairs","ascending stairs","going up the stairs","climb up steps",
            "stair ascent","stepping upward","up the staircase","moving upward on stairs","climbing a staircase",
            "walk-like rhythm with stronger upward thrust on each step",
            "vertical acceleration spikes during lifting the body",
            "slower rhythm than flat walking with heavier steps",
            "positive pitch rotation as the body leans upward",
            "pronounced vertical impulse followed by softer landing",
            "step cycle shows increased vertical amplitude",
            "moderate jerk with ascent-driven pulses",
            "energy centered near 1 to 2 Hz with stronger vertical component",
            "slight asymmetry between lift and landing phases",
            "gyroscope shows forward pitch bursts",
            "periodic pattern with elevated vertical RMS",
            "clearer up-thrust signature than on level walking",
        ],
        "stairsdown": [
            "walk downstairs","descend stairs","going down the stairs","moving downward on stairs","stair descent",
            "down the staircase","stepping downward","descending steps","walking down steps","going downstairs",
            "sharp downward acceleration followed by strong landing impact",
            "higher jerk than walking due to abrupt foot strikes",
            "negative pitch rotation as the body leans forward",
            "rhythm close to walking but more uneven",
            "richer high-frequency components from impacts",
            "vertical axis shows pronounced landing spikes",
            "step timing slightly faster than stair ascent",
            "asymmetric cycle dominated by descent impacts",
            "gyroscope exhibits stronger negative pitch swings",
            "elevated RMS with spiky transients",
            "energy spreads to higher frequencies during landings",
            "distinct impact-driven peaks across consecutive steps",
        ],
    }

CLASSES = ["walk", "still", "stairsup", "stairsdown"]

def wrap_template(cls, s):
    return f"Activity={cls.upper()}. Sensor pattern: {s}"

def encode_texts_raw(model, tokenizer, texts, batch_size=16):
    embs = []
    max_len = getattr(model, "max_len", 512)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,       
                max_length=max_len,    
                return_tensors="pt"
            )
            e = model._bert_embed_long(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device)
            )                                 # (B,H)
            z = F.normalize(e, dim=-1)        
            embs.append(z.detach().cpu().float())
    return torch.cat(embs, dim=0).numpy()   

def build_class_centers(model, tokenizer, TEXT_PROTOS, topk=12, temperature=0.07):
    all_snips, all_cls = [], []
    for c in CLASSES:
        for s in TEXT_PROTOS[c]:
            all_snips.append(wrap_template(c, s))
            all_cls.append(c)
    X = encode_texts_raw(model, tokenizer, all_snips)  # (N,H)
    all_cls = np.array(all_cls)

    centers0 = {c: (X[all_cls==c].mean(0)) for c in CLASSES}
    for c in centers0:
        centers0[c] = centers0[c] / (np.linalg.norm(centers0[c]) + 1e-9)

    Cmat = np.stack([centers0[c] for c in CLASSES], axis=1) 
    sims = X @ Cmat                                          

    own_idx = np.array([CLASSES.index(c) for c in all_cls]) 
    own_sim = sims[np.arange(len(X)), own_idx]
    other_sim = sims.copy()
    other_sim[np.arange(len(X)), own_idx] = -1e9         
    max_other = other_sim.max(axis=1)
    margin = own_sim - max_other

    selected = np.ones(len(X), dtype=bool)
    if topk is not None:
        selected[:] = False
        for c in CLASSES:
            idx = np.where(all_cls==c)[0]
            top = idx[np.argsort(-margin[idx])[:topk]]
            selected[top] = True

    weights = np.zeros(len(X), dtype=np.float32)
    for c in CLASSES:
        idx = np.where((all_cls==c) & selected)[0]
        scores = margin[idx]/temperature
        w = np.exp(scores - scores.max())
        w = w / (w.sum() + 1e-9)
        weights[idx] = w

    centers = {}
    for c in CLASSES:
        idx = np.where((all_cls==c) & selected)[0]
        if len(idx)==0:
            idx = np.where(all_cls==c)[0]
            w = np.ones(len(idx), dtype=np.float32) / max(1, len(idx))
        else:
            w = weights[idx]
        xc = (X[idx] * w[:, None]).sum(axis=0)
        xc = xc / (np.linalg.norm(xc) + 1e-9)
        centers[c] = xc

    return X, all_cls, selected, margin, centers

def label_embedding_generation(device, model, tokenizer, topk=12, temperature=0.07):
    X, cls_arr, selected, margin, centers = build_class_centers(
        model, tokenizer, TEXT_PROTOS, topk=12, temperature=0.07
    )

    walk_embedding      = torch.tensor(centers["walk"])      # (H,)
    still_embedding       = torch.tensor(centers["still"])
    stairsup_embedding  = torch.tensor(centers["stairsup"])
    stairsdown_embedding= torch.tensor(centers["stairsdown"])
    label_proto_n = torch.stack([walk_embedding, still_embedding, stairsup_embedding, stairsdown_embedding], dim=0)  # (4,H)
    return label_proto_n 

def build_label_prototypes(device, model, tokenizer, topk=12, temperature=0.07):

    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        label_proto_n = label_embedding_generation(
            device=device, model=model, tokenizer=tokenizer,
            topk=topk, temperature=temperature
        )  
        label_proto_n = label_proto_n.detach().to(device).float()
    if model_was_training:
        model.train()
    return label_proto_n







    
