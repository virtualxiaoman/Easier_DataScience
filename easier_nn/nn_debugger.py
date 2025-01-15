import torch
import torch.nn as nn


class LayerShapeTracker:
    def __init__(self, model):
        self.model = model
        self.shapes = []

    def hook_fn(self, module, input, output):
        """Hook function to capture the output shape after each layer."""
        self.shapes.append(output.shape)

    def register_hooks(self):
        """Register hooks to all layers in the model."""
        for layer in self.model.children():
            if isinstance(layer, nn.Sequential):
                for sub_layer in layer:
                    sub_layer.register_forward_hook(self.hook_fn)
            else:
                layer.register_forward_hook(self.hook_fn)

    def print_shapes(self, data):
        """Pass the data through the model and print each layer's output shape."""
        self.shapes = []  # Reset the shapes list
        self.model(data)  # Run a forward pass through the model
        for idx, shape in enumerate(self.shapes):
            print(f"Layer {idx + 1} output shape: {shape}")


# 示例模型（一个简单的MLP）
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 示例：传入数据并查看每层的输出形状
if __name__ == "__main__":
    # 模型输入特征维度
    input_dim = 10

    # 创建模型和数据
    model = SimpleNN(input_dim)
    X = torch.randn(16, input_dim)  # 假设batch_size为16

    # 创建LayerShapeTracker实例并注册钩子
    tracker = LayerShapeTracker(model)
    tracker.register_hooks()

    # 打印每一层的输出形状
    tracker.print_shapes(X)
