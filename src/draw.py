import numpy as np
import matplotlib.pyplot as plt

# 加载损失数据
data = np.load("loss_data.npz")
train_losses = data["train_losses"]
val_losses = data["val_losses"]

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.show()
