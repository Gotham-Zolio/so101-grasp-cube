# ACT Policy Training and Deployment Guide

This guide explains how to train ACT (Action Chunking with Transformers) policies using motion planning data and deploy them in simulation and on real robots.

## Workflow Overview

```
1. Data Collection (Motion Planning)
   → 1.1 Validate data quality (including images)
   → 2. Format Conversion (ManiSkill HDF5 → LeRobot Parquet)
   → 3. Training (LeRobot native ACT: lerobot.scripts.train)
   → 4. Evaluation (Simulation/Real Robot)
```

## Training Approaches

This guide supports **two training approaches**:

### **Approach A: Motion Planning Data Training** (Recommended for development)
- Generate synthetic demonstrations using motion planning in simulation
- Faster iteration, controlled environment
- Good for initial development and debugging

### **Approach B: Real Robot Data Training** (Recommended for deployment)
- Use pre-collected real robot demonstrations
- Train directly on real-world data
- Better sim-to-real transfer performance

## Pre-trained Checkpoints

The `checkpoint_real_robot/` directory contains pre-trained ACT policy checkpoints for the three benchmark tasks (lift, sort, stack) trained on real robot demonstrations. These checkpoints can be used directly for evaluation or deployment without retraining.

---

# Approach A: Motion Planning Data Training

## Quick Start

Complete workflow example for the lift task:

```bash
# Step 1: Collect data (50 episodes, ~5-10 minutes)
uv run python scripts/collect_motion_planning_data.py \
    --env-id LiftCubeSO101-v1 \
    --num-episodes 50 \
    --output-dir ./demos

# Step 1.1: Validate data (check if images are valid)
# Note: Files are named with timestamps, use the latest file or specify actual filename
LATEST_H5=$(ls -t demos/LiftCubeSO101-v1/motionplanning/*.h5 | head -1)
uv run python scripts/visualize_training_data_images.py \
    --hdf5-path "$LATEST_H5" \
    --max-images 20 \
    --mode static \
    --save-static ./training_data_visualization.png \
    --no-display

# Step 2: Convert data format (HDF5 → LeRobot Parquet, ~1-2 min with --no-videos)
uv run python scripts/convert_h5_to_lerobot_parquet.py \
    --input-dir ./demos/LiftCubeSO101-v1/motionplanning \
    --output-dir ./lerobot_datasets \
    --task-name lift \
    --no-videos

# Step 3: Train model with LeRobot native ACT (~1-3 hours)
CUDA_VISIBLE_DEVICES=0 uv run python -m lerobot.scripts.train \
    --dataset.repo_id lift_cube \
    --dataset.root ./lerobot_datasets/lift_cube \
    --policy.type act \
    --policy.chunk_size 100 \
    --policy.push_to_hub false \
    --output_dir ./checkpoints/lift_act \
    --steps 50000 \
    --save_freq 10000 \
    --batch_size 16 \
    --num_workers 8 \
    --policy.device cuda

# Step 4: Evaluate model (~5-10 minutes)
# --policy-path can be output_dir; eval script auto-finds latest checkpoint
uv run python scripts/eval_sim_policy.py \
    --policy-path ./checkpoints/lift_act \
    -e LiftCubeSO101-v1 \
    -n 50 \
    --save-camera \
    --verbose
```

**Expected Results:**
- Data collection: HDF5 files with valid images
- Conversion: Parquet dataset under `./lerobot_datasets/{task}_cube`
- Training: loss decreases, model converges
- Evaluation: success rate > 50% (target)

## Step 1: Data Collection

Use motion planning solutions to generate successful trajectory data. **Important**: Ensure the data contains valid camera images (sensor_data).

```bash
# Collect lift task data (recommended: 50-500 episodes)
uv run python scripts/collect_motion_planning_data.py \
    --env-id LiftCubeSO101-v1 \
    --num-episodes 50 \
    --output-dir ./demos

# Collect stack task data
uv run python scripts/collect_motion_planning_data.py \
    --env-id StackCubeSO101-v1 \
    --num-episodes 50 \
    --output-dir ./demos

# Collect sort task data
uv run python scripts/collect_motion_planning_data.py \
    --env-id SortCubeSO101-v1 \
    --num-episodes 50 \
    --output-dir ./demos
```

**Output**: Trajectory files in ManiSkill format (h5 format), saved in `./demos/{env_id}/motionplanning/`

**File Naming**:
- Files are named with timestamp format (e.g., `20260115_063449.h5`), not fixed `lift_demo.h5`
- Get the latest file using: `ls -t demos/LiftCubeSO101-v1/motionplanning/*.h5 | head -1`
- Or check the directory directly: `ls demos/LiftCubeSO101-v1/motionplanning/`

**Important Notes**:
- Data collection script automatically verifies if `sensor_data` is available
- If you see "✓ sensor_data is available in get_obs()", data collection is working correctly
- If you see "⚠️ WARNING: sensor_data NOT in get_obs()!", check environment configuration

### Step 1.1: Validate Data Quality

Always validate that collected data contains valid images before training:

```bash
# Visualize training data images (check if images are valid)
# Note: Data collection generates timestamp-formatted filenames (e.g., 20260115_063449.h5)
# Use the latest file or specify actual filename
LATEST_H5=$(ls -t demos/LiftCubeSO101-v1/motionplanning/*.h5 | head -1)
uv run python scripts/visualize_training_data_images.py \
    --hdf5-path "$LATEST_H5" \
    --max-images 20 \
    --mode static \
    --save-static ./training_images.png \
    --no-display

# Or directly specify filename (replace with actual filename)
# uv run python scripts/visualize_training_data_images.py \
#     --hdf5-path ./demos/LiftCubeSO101-v1/motionplanning/20260115_063449.h5 \
#     --max-images 20 \
#     --mode static \
#     --save-static ./training_images.png \
#     --no-display
```

**Validation Criteria**:
- ✓ obs contains sensor_data
- ✓ sensor_data contains front, left_side, right_side
- ✓ Average image brightness > 50 (not all black)
- ✓ Image shape (480, 640, 3)

If average image brightness < 5.0, images are all black and data must be recollected.

## Step 2: Format Conversion

Convert ManiSkill trajectories (HDF5) to LeRobot Dataset format (Parquet). Use `convert_h5_to_lerobot_parquet.py` so the output is compatible with LeRobot's native `lerobot.scripts.train`.

```bash
# Convert lift task data
uv run python scripts/convert_h5_to_lerobot_parquet.py \
    --input-dir ./demos/LiftCubeSO101-v1/motionplanning \
    --output-dir ./lerobot_datasets \
    --task-name lift \
    --no-videos

# Convert stack task data
uv run python scripts/convert_h5_to_lerobot_parquet.py \
    --input-dir ./demos/StackCubeSO101-v1/motionplanning \
    --output-dir ./lerobot_datasets \
    --task-name stack \
    --no-videos

# Convert sort task data
uv run python scripts/convert_h5_to_lerobot_parquet.py \
    --input-dir ./demos/SortCubeSO101-v1/motionplanning \
    --output-dir ./lerobot_datasets \
    --task-name sort \
    --no-videos

# Convert avoid_obstacle task data (custom task)
uv run python scripts/convert_h5_to_lerobot_parquet.py \
    --input-dir ./demos/AvoidObstacleSO101-v1/motionplanning \
    --output-dir ./lerobot_datasets \
    --task-name avoid_obstacle \
    --no-videos
```

**Notes**:
- `--input-dir`: directory containing `.h5` files (e.g. `./demos/LiftCubeSO101-v1/motionplanning`)
- `--output-dir`: parent dir for datasets; each task is written to `{output-dir}/{task_name}_cube` (e.g. `./lerobot_datasets/lift_cube`)
- `--task-name`: one of `lift`, `stack`, `sort`, `avoid_obstacle`
- `--no-videos`: store images as arrays (faster conversion, slightly slower training); omit to use MP4 videos
- All tasks use three cameras: `front`, `left_side`, `right_side`

## Step 3: Training

### 3.1 Install LeRobot

```bash
# Install using uv
uv add lerobot

# Or using pip
pip install lerobot
```

### 3.2 Train ACT Policy (LeRobot Native Command)

Use LeRobot's native training script `lerobot.scripts.train` with the Parquet datasets from Step 2:

```bash
# Train lift task policy
CUDA_VISIBLE_DEVICES=0 uv run python -m lerobot.scripts.train \
    --dataset.repo_id lift_cube \
    --dataset.root ./lerobot_datasets/lift_cube \
    --policy.type act \
    --policy.chunk_size 100 \
    --policy.push_to_hub false \
    --output_dir ./checkpoints/lift_act \
    --steps 50000 \
    --save_freq 10000 \
    --batch_size 16 \
    --num_workers 8 \
    --policy.device cuda

# Train stack task policy
CUDA_VISIBLE_DEVICES=0 uv run python -m lerobot.scripts.train \
    --dataset.repo_id stack_cube \
    --dataset.root ./lerobot_datasets/stack_cube \
    --policy.type act \
    --policy.chunk_size 100 \
    --policy.push_to_hub false \
    --output_dir ./checkpoints/stack_act \
    --steps 50000 \
    --save_freq 10000 \
    --batch_size 16 \
    --num_workers 8 \
    --policy.device cuda

# Train sort task policy
CUDA_VISIBLE_DEVICES=0 uv run python -m lerobot.scripts.train \
    --dataset.repo_id sort_cube \
    --dataset.root ./lerobot_datasets/sort_cube \
    --policy.type act \
    --policy.chunk_size 100 \
    --policy.push_to_hub false \
    --output_dir ./checkpoints/sort_act \
    --steps 50000 \
    --save_freq 10000 \
    --batch_size 16 \
    --num_workers 8 \
    --policy.device cuda

# Train avoid_obstacle task policy (custom task)
CUDA_VISIBLE_DEVICES=0 uv run python -m lerobot.scripts.train \
    --dataset.repo_id avoid_obstacle_cube \
    --dataset.root ./lerobot_datasets/avoid_obstacle_cube \
    --policy.type act \
    --policy.chunk_size 100 \
    --policy.push_to_hub false \
    --output_dir ./checkpoints/avoid_obstacle_act \
    --steps 50000 \
    --save_freq 10000 \
    --batch_size 16 \
    --num_workers 8 \
    --policy.device cuda
```

**Training Parameters**:
- `--dataset.repo_id`: Dataset name, must match the folder under `--dataset.root` (e.g. `lift_cube`)
- `--dataset.root`: Path to the LeRobot dataset directory (e.g. `./lerobot_datasets/lift_cube`)
- `--policy.type act`: Use ACT policy
- `--policy.chunk_size 100`: Action chunk size
- `--policy.push_to_hub false`: Do not upload to HuggingFace
- `--output_dir`: Directory for checkpoints (e.g. `./checkpoints/lift_act`)
- `--steps`: Training steps
- `--save_freq`: Save checkpoint every N steps
- `--batch_size`, `--num_workers`: Data loading
- `--policy.device cuda`: Device for policy

**Training Output**:
- Checkpoints under `{output_dir}/checkpoints/{step}/pretrained_model/` (e.g. `020000`, `040000`, `last`)
- `eval_sim_policy.py` accepts `--policy-path {output_dir}` and automatically uses the latest checkpoint.

## Step 4: Evaluation

### Simulation Evaluation

```bash
# Evaluate lift task policy (--policy-path can be output_dir; latest checkpoint is used automatically)
uv run python scripts/eval_sim_policy.py \
    --policy-path ./checkpoints/lift_act \
    -e LiftCubeSO101-v1 \
    -n 50 \
    --save-camera \
    --verbose

# Evaluate stack task policy
uv run python scripts/eval_sim_policy.py \
    --policy-path ./checkpoints/stack_act \
    -e StackCubeSO101-v1 \
    -n 50 \
    --save-camera

# Evaluate sort task policy
uv run python scripts/eval_sim_policy.py \
    --policy-path ./checkpoints/sort_act \
    -e SortCubeSO101-v1 \
    -n 50 \
    --save-camera
```

**Evaluation Parameters**:
- `--save-camera`: Save camera view videos to `./eval_results/camera_videos/`
- `--save-video`: Save rendered view videos to `./eval_results/videos/`
- `--save-failed-only`: Only save failed episode videos
- `--verbose`: Show detailed debugging information

**Target**: Success rate > 50% (c1 + c2 + c3 = 3 points)

### Real Robot Evaluation

```bash
# Evaluate policy (requires robot interface configuration)
uv run python scripts/eval_real_policy.py \
    --policy-path ./checkpoints/lift_act \
    --camera-config-path configs/so101.json \
    --task "Pick up the red cube and lift it." \
    -n 10
```

**Target**: Success rate > 30% (c1 + c2 + c3 = 3 points)

## Data Validation Tools

### Visualize Training Data Images

Strongly recommend visualizing data before training to confirm image quality:

```bash
# Static visualization (grid display of multiple images)
# Use the latest generated file
LATEST_H5=$(ls -t demos/LiftCubeSO101-v1/motionplanning/*.h5 | head -1)
uv run python scripts/visualize_training_data_images.py \
    --hdf5-path "$LATEST_H5" \
    --max-images 20 \
    --mode static \
    --save-static ./training_images.png \
    --no-display

# Animation visualization (video format, more intuitive)
LATEST_H5=$(ls -t demos/LiftCubeSO101-v1/motionplanning/*.h5 | head -1)
uv run python scripts/visualize_training_data_images.py \
    --hdf5-path "$LATEST_H5" \
    --max-images 50 \
    --mode animation \
    --save-animation ./training_images.mp4 \
    --fps 10 \
    --no-display
```

**Output Description**:
- Static visualization: Saved as PNG image, displays grid of images from multiple timesteps
- Animation visualization: Saved as MP4 video, can view image sequences
- Image statistics: Shows average brightness, max, min, and other information

**Evaluation Criteria**:
- ✓ Average image brightness > 50: Image quality is normal
- ⚠️ Average image brightness 5-50: Images are dim, may affect training
- ✗ Average image brightness < 5: Images are all black, **must recollect data**

## Key Configuration Details

### Camera Configuration

- All tasks use three fixed cameras: **front**, **left_side**, **right_side** (480×640, 50° FoV).
- Wrist cameras are not used; left_side and right_side are fixed poses.
- Data collection must use `obs_mode="rgb"` so sensor_data is saved correctly.

### Action Format

- **Format**: Joint positions (qpos), not end-effector poses
- **Dimensions**:
  - Single-arm tasks (lift, stack): 6-dim (6 joints)
  - Dual-arm tasks (sort, avoid_obstacle): 12-dim (6+6 joints)

### Task Prompts

- `LiftCubeSO101-v1`: "Pick up the red cube and lift it."
- `StackCubeSO101-v1`: "Stack the red cube on top of the green cube."
- `SortCubeSO101-v1`: "Move the red cube to the left region and the green cube to the right region."
- `AvoidObstacleSO101-v1`: "Lift the red cube over obstacles to the middle. Lift the green cube over obstacles to the middle."

## Data Augmentation and Generalization

### Overview

We implemented a `VisualAugmentation` module in `grasp_cube/utils/data_augmentation.py` that provides visual data augmentation during training. This is a meaningful difference from the baseline LeRobot ACT implementation, which typically only performs normalization without strong online visual augmentation.

### Augmentation Types

The `VisualAugmentation` module provides:

- **Color Jitter**: Random adjustments of brightness, contrast, saturation, and hue
- **Gaussian Noise**: Add random noise to improve robustness
- **Gaussian Blur**: Optional, simulates motion blur

### Usage Method 1: Modify LeRobot Source (Recommended for Training)

Since LeRobot's training loop is encapsulated, the simplest method is to apply augmentation in the model's forward pass.

**Steps:**

1. Find LeRobot installation location:
   ```bash
   python -c "import lerobot; print(lerobot.__file__)"
   ```

2. Modify `lerobot/policies/act/modeling_act.py`:
   - In the `ACTPolicy` class's `forward` or `normalize_inputs` method
   - Add augmentation after normalization, before encoding

   Example modification location (in `forward` method):
   ```python
   # After normalize_inputs
   batch = self.normalize_inputs(batch)
   
   # Add data augmentation (only during training)
   if self.training:
       from grasp_cube.utils.data_augmentation import VisualAugmentation
       if not hasattr(self, '_augmentation'):
           self._augmentation = VisualAugmentation(
               brightness_range=(0.7, 1.3),
               contrast_range=(0.7, 1.3),
               saturation_range=(0.7, 1.3),
               hue_range=(-0.1, 0.1),
               gaussian_noise_std=0.02,
               apply_prob=0.8,
               training=True,
           )
       # Extract image features
       images = {k: v for k, v in batch.items() if k.startswith("observation.images.")}
       if images:
           augmented_images = self._augmentation(images)
           batch.update(augmented_images)
   ```

3. Retrain the model (use `lerobot.scripts.train` as in Step 3.2):
   ```bash
   CUDA_VISIBLE_DEVICES=0 uv run python -m lerobot.scripts.train \
       --dataset.repo_id avoid_obstacle_cube \
       --dataset.root ./lerobot_datasets/avoid_obstacle_cube \
       --policy.type act \
       --policy.chunk_size 100 \
       --policy.push_to_hub false \
       --output_dir ./checkpoints/avoid_obstacle_act_aug \
       --steps 50000 \
       --save_freq 10000 \
       --batch_size 16 \
       --num_workers 8 \
       --policy.device cuda
   ```

### Generalization Experiments

To verify that augmented models have better generalization performance than baseline models, we conduct visual disturbance experiments.

**Experimental Setup:**
- Baseline model: Trained with standard LeRobot (no augmentation)
- Augmented model: Trained with data augmentation
- Test conditions: Apply different levels of visual disturbances during evaluation

#### Baseline Model Evaluation (No Disturbance)

```bash
uv run python scripts/eval_sim_policy.py \
    --policy-path ./checkpoints/avoid_obstacle_act_baseline \
    -e AvoidObstacleSO101-v1 \
    -n 50 \
    --output-dir ./eval_results/baseline_no_disturbance
```

#### Baseline Model Evaluation (With Disturbance)

```bash
uv run python scripts/eval_sim_policy.py \
    --policy-path ./checkpoints/avoid_obstacle_act_baseline \
    -e AvoidObstacleSO101-v1 \
    -n 50 \
    --visual-disturbance \
    --brightness-factor 0.7 \
    --contrast-factor 0.8 \
    --noise-std 0.05 \
    --output-dir ./eval_results/baseline_with_disturbance
```

#### Augmented Model Evaluation (No Disturbance)

```bash
uv run python scripts/eval_sim_policy.py \
    --policy-path ./checkpoints/avoid_obstacle_act_aug \
    -e AvoidObstacleSO101-v1 \
    -n 50 \
    --output-dir ./eval_results/augmented_no_disturbance
```

#### Augmented Model Evaluation (With Disturbance)

```bash
uv run python scripts/eval_sim_policy.py \
    --policy-path ./checkpoints/avoid_obstacle_act_aug \
    -e AvoidObstacleSO101-v1 \
    -n 50 \
    --visual-disturbance \
    --brightness-factor 0.7 \
    --contrast-factor 0.8 \
    --noise-std 0.05 \
    --output-dir ./eval_results/augmented_with_disturbance
```

### Disturbance Intensity Settings

Test different disturbance intensities:

**Light Disturbance:**
```bash
--brightness-factor 0.9 --contrast-factor 0.9 --noise-std 0.02
```

**Medium Disturbance:**
```bash
--brightness-factor 0.7 --contrast-factor 0.8 --noise-std 0.05
```

**Heavy Disturbance:**
```bash
--brightness-factor 0.5 --contrast-factor 0.6 --noise-std 0.1
```

### Results Analysis

Compare the following metrics:

1. **Success rate under no disturbance**: Verify augmentation doesn't affect normal performance
2. **Success rate under disturbance**: Verify augmentation improves generalization performance

**Expected Results:**
- Baseline model: High success rate without disturbance, significant drop with disturbance
- Augmented model: Similar success rate to baseline without disturbance, smaller drop with disturbance

**Success Criteria:**
- Augmented model's success rate under disturbance is significantly higher than baseline model
- This proves data augmentation improves model generalization capability

## Troubleshooting

### Data Collection Issues

**Issue: obs group is empty, no sensor_data**
- **Cause**: Environment configuration issue, sensor_data not saved
- **Solution**:
  1. Ensure using `obs_mode="rgb"` (script automatically sets this)
  2. Check debug output during data collection, confirm "✓ sensor_data is available"
  3. If sensor_data still missing, check ManiSkill version and environment configuration

**Issue: Images are all black (average brightness < 5.0)**
- **Cause**: Scene rendering issue or camera configuration error
- **Solution**:
  1. Check if camera configuration is correct
  2. Ensure objects and lighting exist in scene
  3. Recollect data

### Training Failures

- Ensure Parquet dataset and `meta/info.json` exist under `./lerobot_datasets/{task}_cube`.
- Ensure `--dataset.repo_id` and `--dataset.root` match (e.g. `lift_cube` and `./lerobot_datasets/lift_cube`).
- Check LeRobot: `python -c "import lerobot; print(lerobot.__version__)"`.
- Reduce `--batch_size` if out of GPU memory.

### Low Evaluation Success Rate

- **Check data quality**: Use visualization tools to confirm images are valid
- **Increase training data**: 50-500 episodes (recommended: 200+)
- **Adjust model capacity**: Increase hidden_dim, num_layers
- **Adjust hyperparameters**: learning rate, action chunk size
- **Check action quality**: Are actions smooth, are images clear

## Why Use LeRobot Native Training

1. **Format compatibility**: `convert_h5_to_lerobot_parquet.py` produces Parquet datasets that work directly with `lerobot.scripts.train`.
2. **Full pipeline**: Native script provides dataloader, optimizer, checkpointing, WandB, etc.
3. **Low maintenance**: Track LeRobot releases instead of maintaining a custom training loop.
4. **Evaluation**: `eval_sim_policy.py` accepts `--policy-path {output_dir}` and resolves to `checkpoints/{step}/pretrained_model` automatically.

## Notes on Data Augmentation

1. **Training Time**: Data augmentation increases training time (each batch requires additional processing), but impact is usually small.

2. **Augmentation Strength**: Don't set too strong augmentation (e.g., `brightness_range=(0.1, 2.0)`), this may cause training instability.

3. **Evaluation Disturbances**: Evaluation disturbance intensity should match or slightly exceed training augmentation intensity to verify generalization performance.

4. **Reproducibility**: Recommend fixing random seed to ensure reproducible results:
   ```bash
   --seed 42
   ```

---

# Approach B: Real Robot Data Training

Train ACT policies directly on pre-collected real robot demonstrations.

## Data Preparation

The project includes pre-collected real robot demonstration data. Convert existing data to LeRobot format:

```bash
# Convert existing real data to LeRobot format
uv run python scripts/convert_h5_to_lerobot_parquet.py \
    --input-dir real_data/lift \
    --output-dir datasets/lift

uv run python scripts/convert_trajectory_to_lerobot.py \
    --input-dir real_data/lift \
    --output-dir datasets/lift
```

## Train ACT Policy

Train ACT policies for each task using real demonstration data:

```bash
# Train Lift task (6D actions)
uv run python scripts/train_act_real_data.py \
    --task lift \
    --output-dir checkpoints/lift_real \
    --epochs 100 \
    --batch-size 8 \
    --learning-rate 1e-4

# Train Sort task (12D actions)
uv run python scripts/train_act_real_data.py \
    --task sort \
    --output-dir checkpoints/sort_real \
    --epochs 100 \
    --batch-size 8 \
    --learning-rate 1e-4

# Train Stack task (6D actions)
uv run python scripts/train_act_real_data.py \
    --task stack \
    --output-dir checkpoints/stack_real \
    --epochs 100 \
    --batch-size 8 \
    --learning-rate 1e-4
```

## Simulation Evaluation

Evaluate trained policies in simulation:

```bash
# Evaluate Lift policy
uv run python scripts/eval_sim_policy.py \
    --policy-path checkpoints/lift_real \
    --task lift \
    --num-episodes 50 \
    --output-dir eval_results/lift

# Evaluate Sort policy
uv run python scripts/eval_sim_policy.py \
    --policy-path checkpoints/sort_real \
    --task sort \
    --num-episodes 50 \
    --output-dir eval_results/sort

# Evaluate Stack policy
uv run python scripts/eval_sim_policy.py \
    --policy-path checkpoints/stack_real \
    --task stack \
    --num-episodes 50 \
    --output-dir eval_results/stack
```

**Expected Performance**:
- **Lift**: ~82% success rate (50 episodes)
- **Sort**: ~84% success rate (50 episodes)
- **Stack**: ~90% success rate (50 episodes)

## Real Robot Deployment

### Server Setup (Policy Server)

```bash
# Build Docker image for policy server
docker build -t so101-act-server -f docker/Dockerfile.server .

# Or run directly
uv run python grasp_cube/real/serve_act_policy.py \
    --policy-path checkpoints/lift_real \
    --host 0.0.0.0 \
    --port 8000
```

### Client Setup (Robot Environment)

```bash
# Create separate environment for robot client
uv venv robot_env
source robot_env/bin/activate  # On Windows: robot_env\Scripts\activate
uv pip install -e packages/env-client

# Test with simulated environment
uv run python grasp_cube/real/run_fake_env_client.py \
    --dataset-path datasets/lift \
    --host localhost \
    --port 8000
```

### Real Robot Evaluation

```bash
# Run real robot evaluation
uv run python scripts/eval_real_policy.py \
    --policy-server ws://robot-server:8000 \
    --task lift \
    --num-episodes 10 \
    --output-dir real_eval_results/lift
```

### Monitoring Dashboard

Access monitoring dashboard at `http://localhost:9000` to view real-time execution and control evaluation flow.

## Key Differences: Approach A vs B

| Aspect | Approach A (Motion Planning) | Approach B (Real Data) |
|--------|-----------------------------|------------------------|
| **Data Source** | Synthetic demonstrations | Real robot demonstrations |
| **Data Quality** | Perfect trajectories | Human demonstrations (may have imperfections) |
| **Training Time** | Faster (50-200 episodes) | Slower (requires more data) |
| **Sim-to-Real** | May need domain adaptation | Better transfer performance |
| **Development Speed** | Faster iteration | Slower iteration |
| **Production Use** | Good for prototyping | Recommended for deployment |

## Future Improvements

1. **Domain Randomization**: Add lighting, texture randomization during data collection.
2. **Action Chunking**: Tune chunk_size per task to balance latency and performance.
3. **Hyperparameter Tuning**: Adjust learning rate, batch size, etc. on a validation set.