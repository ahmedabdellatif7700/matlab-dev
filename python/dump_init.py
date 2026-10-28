import numpy as np
import equalizer as eq

eq.seed_training()                      # same seed as the training run
model = eq.create_model("cpu")          # init happens here, before any training

parts = [model.fc1.weight, model.fc1.bias, model.fc2.weight, model.fc2.bias]
flat = np.concatenate([p.detach().numpy().astype(np.float32).ravel() for p in parts])
assert flat.size == 86
np.savetxt("init_params.txt", flat.astype(np.float64), fmt="%.10e")
print(flat[:4], flat[-2:])