#!/usr/bin/env python3
"""Check ACTPolicy signature and stats format"""

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
import inspect

# Check the __init__ signature
sig = inspect.signature(ACTPolicy.__init__)
print("ACTPolicy.__init__ parameters:")
for param_name, param in sig.parameters.items():
    print(f"  {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'no annotation'}")

# Check if there's a stats parameter
print(f"\n'stats' in parameters: {'stats' in sig.parameters}")

# Try to see what the normalize_inputs layer expects
print("\nLet's check the normalize_inputs layer:")
config = ACTConfig(
    n_obs_steps=1,
    n_action_steps=8,
    input_features={
        "observation.images.front": __import__('lerobot.configs.types', fromlist=['PolicyFeature']).PolicyFeature(
            type=__import__('lerobot.configs.types', fromlist=['FeatureType']).FeatureType.VISUAL,
            shape=(3, 480, 640),
        ),
        "observation.state": __import__('lerobot.configs.types', fromlist=['PolicyFeature']).PolicyFeature(
            type=__import__('lerobot.configs.types', fromlist=['FeatureType']).FeatureType.STATE,
            shape=(6,),
        ),
    },
    output_features={
        "action": __import__('lerobot.configs.types', fromlist=['PolicyFeature']).PolicyFeature(
            type=__import__('lerobot.configs.types', fromlist=['FeatureType']).FeatureType.ACTION,
            shape=(6,),
        ),
    },
)

try:
    model = ACTPolicy(config)
    print(f"Model created successfully")
    print(f"Has normalize_inputs: {hasattr(model, 'normalize_inputs')}")
    if hasattr(model, 'normalize_inputs'):
        norm = model.normalize_inputs
        print(f"normalize_inputs type: {type(norm)}")
        print(f"normalize_inputs attributes: {dir(norm)}")
except Exception as e:
    print(f"Error creating model: {e}")
