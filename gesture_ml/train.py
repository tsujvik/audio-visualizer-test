import numpy as np
import torch
import torch.nn as nn

from gesture import LABELS, Net

X = np.load("data/X.npy")
y = np.load("data/y.npy")
print(len(X), "samples")

# shuffle then 80/20 train/val split
np.random.seed(0)
torch.manual_seed(0)
idx = np.random.permutation(len(X))
cut = int(len(X) * 0.8)
X_train = torch.tensor(X[idx[:cut]])
y_train = torch.tensor(y[idx[:cut]], dtype=torch.long)
X_val = torch.tensor(X[idx[cut:]])
y_val = torch.tensor(y[idx[cut:]], dtype=torch.long)


def augment(x):
    # randomly rotate hands a bit + add noise so it doesnt just memorize how i held my hand
    pts = x.view(-1, 21, 3).clone()
    angle = (torch.rand(len(x)) - 0.5) * 0.5  # about +-15 degrees
    c = torch.cos(angle).unsqueeze(1)
    s = torch.sin(angle).unsqueeze(1)
    xs = pts[:, :, 0].clone()
    ys = pts[:, :, 1].clone()
    pts[:, :, 0] = xs * c - ys * s
    pts[:, :, 1] = xs * s + ys * c
    pts = pts + torch.randn_like(pts) * 0.02
    return pts.view(-1, 63)


net = Net(len(LABELS))
opt = torch.optim.Adam(net.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
best = 0

for epoch in range(60):
    net.train()
    order = torch.randperm(len(X_train))
    for i in range(0, len(X_train), 64):
        batch = order[i:i + 64]
        out = net(augment(X_train[batch]))
        loss = loss_fn(out, y_train[batch])
        opt.zero_grad()
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        acc = (net(X_val).argmax(1) == y_val).float().mean().item()
    if epoch % 5 == 0:
        print(f"epoch {epoch} loss {loss.item():.3f} val acc {acc:.3f}")
    if acc > best:  # save the best one
        best = acc
        torch.save(net.state_dict(), "gesture_model.pt")

print("best val acc:", round(best, 3))

# confusion matrix to see which gestures get mixed up
net.load_state_dict(torch.load("gesture_model.pt"))
net.eval()
with torch.no_grad():
    preds = net(X_val).argmax(1)
print("rows = real, cols = guessed")
print("       " + " ".join(f"{l:>6}" for l in LABELS))
for i, name in enumerate(LABELS):
    row = [int(((y_val == i) & (preds == j)).sum()) for j in range(len(LABELS))]
    print(f"{name:>6} " + " ".join(f"{n:>6}" for n in row))