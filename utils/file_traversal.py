import numpy as np
import yaml

def visualize_structure(data, name="root"):
    if isinstance(data, dict):
        node = {name: {}}
        for key, value in data.items():
            node[name][key] = visualize_structure(value, key)
        return node
    elif isinstance(data, list):
        node = {name: f"List of length {len(data)}"}
        if len(data) > 0:
            node[name] += " with elements:"
            for i, item in enumerate(data):
                node[name] += f"\n  - {visualize_structure(item, f'{name}_{i}')}"
        return node
    elif isinstance(data, np.ndarray):
        return {name: f"np.array of shape {data.shape} and dtype {data.dtype}"}
    else:
        return {name: f"{type(data).__name__} value"}

# Example usage:
data = {
    "observations": {
        "state": np.array([1, 2, 3]),
        "metadata": {"id": 123, "type": "example"}
    },
    "actions": [0.5, 0.7, 0.2],
    "info": np.array([[1, 2], [3, 4]])
}

# Convert the structure to YAML format
tree = visualize_structure(data)
yaml_str = yaml.dump(tree, default_flow_style=False)

# Output the YAML string
print(yaml_str)
