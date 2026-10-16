'''behaviour cloning policy
'''

import csv
import json
import sys
from pathlib import Path
 
import numpy as np
import torch
from torch import nn
 
HELD_OUT = {"demo_04", "demo_09", "demo_13", "demo_17"}
MODEL_PATH = Path("models/policy.pt")
EPOCHS = 2000
RESIDUAL_SCALE = 0.05  # corrections are a few cm, scaled up to around 1 for training
WEIGHT_PENALTY = 0.1   # pushes weights towards zero, i.e. towards the baseline
 
 
def load_demos():
    demos = [json.loads(p.read_text()) for p in sorted(Path("demos").glob("demo_*.json"))]
    train = [d for d in demos if d["name"] not in HELD_OUT]
    test = [d for d in demos if d["name"] in HELD_OUT]
    return train, test
 
 
def to_input(demo):
    '''What the policy sees: where the cube starts.'''
    return np.array(demo["grasp_xy"], dtype=np.float32)
 
 
def to_output(demo):
    '''What it predicts: the carried path, flattened into one list, then the release point.'''
    return np.concatenate([np.ravel(demo["path_xy"]), demo["release_xy"]]).astype(np.float32)
 
 
def make_model(n_out):
    '''Linear model: each output is a weighted sum of the start x and y. Too simple to memorise.'''
    return nn.Linear(2, n_out)
 
 
def baseline_output(grasp_xy, mean_release, n_points):
    '''Straight line from the start to the usual drop spot, in the same layout as to_output.'''
    path = np.linspace(grasp_xy, mean_release, n_points)
    return np.concatenate([path.ravel(), mean_release]).astype(np.float32)
 
 
def train():
    torch.manual_seed(0)  # same result every run
    train_demos, _ = load_demos()
    if len(train_demos) < 3:
        raise RuntimeError("Need at least 3 training demos in demos/")
 
    n_points = len(train_demos[0]["path_xy"])
    mean_release = np.mean([d["release_xy"] for d in train_demos], axis=0)
 
    X = np.stack([to_input(d) for d in train_demos])
    Y = np.stack([to_output(d) for d in train_demos])
    B = np.stack([baseline_output(d["grasp_xy"], mean_release, n_points) for d in train_demos])
 
    # inputs scaled to mean 0, spread 1. Targets are the corrections to the baseline.
    x_mean, x_std = X.mean(0), X.std(0) + 1e-6
    X_scaled = torch.tensor((X - x_mean) / x_std, dtype=torch.float32)
    R_scaled = torch.tensor((Y - B) / RESIDUAL_SCALE, dtype=torch.float32)
 
    model = make_model(Y.shape[1])
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
 
    for epoch in range(EPOCHS):
        fit = nn.functional.mse_loss(model(X_scaled), R_scaled)
        penalty = WEIGHT_PENALTY * (model.weight ** 2).mean()
        loss = fit + penalty
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if epoch % 500 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:4d}  fit {fit.item():.5f}  penalty {penalty.item():.5f}")
 
    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "n_out": int(Y.shape[1]),
        "n_points": int(n_points),
        "mean_release": mean_release.tolist(),
        "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
    }, MODEL_PATH)
    print(f"Trained on {len(train_demos)} demos, saved {MODEL_PATH}")
 
 
def load_model():
    checkpoint = torch.load(MODEL_PATH)
    model = make_model(checkpoint["n_out"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint
 
 
def predict(model, checkpoint, grasp_xy):
    '''Returns the predicted path and release point for a cube starting at grasp_xy.'''
    grasp_xy = np.array(grasp_xy, dtype=np.float32)
    x = (grasp_xy - np.array(checkpoint["x_mean"])) / np.array(checkpoint["x_std"])
    with torch.no_grad():
        correction = model(torch.tensor(x, dtype=torch.float32)).numpy() * RESIDUAL_SCALE
    y = baseline_output(grasp_xy, np.array(checkpoint["mean_release"]), checkpoint["n_points"]) + correction
 
    path = y[:-2].reshape(-1, 2)
    release = y[-2:]
    path[0] = grasp_xy  # the path always starts where the cube actually is
    return path, release
 
 
def save_prediction(name, grasp_xy, path, release):
    demo = {
        "name": f"predicted_{name}",
        "source": "policy",
        "grasp_xy": [float(v) for v in grasp_xy],
        "release_xy": np.round(release, 4).tolist(),
        "path_xy": np.round(path, 4).tolist(),
    }
    out_path = Path("demos") / f"predicted_{name}.json"
    out_path.write_text(json.dumps(demo, indent=1))
    return out_path
 
 
def straight_line(start, end, n_points):
    return np.linspace(start, end, n_points)
 
 
def test():
    train_demos, test_demos = load_demos()
    if not test_demos:
        raise RuntimeError(f"No held-out demos found. Expected {sorted(HELD_OUT)} in demos/")
    model, checkpoint = load_model()
 
    # baselines: always release at the average training spot, carried in a straight line
    mean_release = np.mean([d["release_xy"] for d in train_demos], axis=0)
 
    rows = []
    for demo in test_demos:
        true_path = np.array(demo["path_xy"])
        true_release = np.array(demo["release_xy"])
 
        path, release = predict(model, checkpoint, demo["grasp_xy"])
        base_path = straight_line(demo["grasp_xy"], mean_release, len(true_path))
 
        rows.append({
            "demo": demo["name"],
            "release_error_policy_mm": round(float(np.linalg.norm(release - true_release)) * 1000, 1),
            "release_error_baseline_mm": round(float(np.linalg.norm(mean_release - true_release)) * 1000, 1),
            "path_error_policy_mm": round(float(np.linalg.norm(path - true_path, axis=1).mean()) * 1000, 1),
            "path_error_baseline_mm": round(float(np.linalg.norm(base_path - true_path, axis=1).mean()) * 1000, 1),
        })
        save_prediction(demo["name"], demo["grasp_xy"], path, release)
 
    print(f"{'demo':10} {'release policy':>15} {'release base':>13} {'path policy':>12} {'path base':>10}  (mm)")
    for r in rows:
        print(f"{r['demo']:10} {r['release_error_policy_mm']:15} {r['release_error_baseline_mm']:13} "
              f"{r['path_error_policy_mm']:12} {r['path_error_baseline_mm']:10}")
    for key in rows[0]:
        if key != "demo":
            print(f"mean {key}: {np.mean([r[key] for r in rows]):.1f}")
 
    Path("results").mkdir(exist_ok=True)
    with open("results/policy_test.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved results/policy_test.csv and demos/predicted_*.json")
 
 
if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
 
    if command == "train":
        train()
    elif command == "test":
        test()
    elif command == "predict" and len(sys.argv) == 4:
        start = [float(sys.argv[2]), float(sys.argv[3])]
        model, checkpoint = load_model()
        path, release = predict(model, checkpoint, start)
        print(f"Saved {save_prediction('custom', start, path, release)}, release at {release.round(3)}")
    else:
        print(__doc__)