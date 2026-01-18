# SO101 Cube Grasping with ACT

This repository implements a complete **Action Chunking with Transformers (ACT)** pipeline for robotic manipulation on the SO101 robot platform. The project demonstrates end-to-end deployment from simulation training to real robot execution.

## 🎯 Key Features

- **ACT Policy Implementation**: Complete 417-line ACT policy with manual normalization
- **Multi-Task Support**: Lift, Sort, and Stack tasks with varying action dimensions (6D/12D)
- **Real Robot Integration**: Server-client architecture for production deployment
- **Comprehensive Evaluation**: Simulation and real robot evaluation frameworks
- **Docker Deployment**: Containerized policy server for easy deployment

## 📋 Prerequisites

- **Python**: >= 3.10
- **OS**: Linux (recommended), Windows/macOS (limited support)
- **GPU**: NVIDIA GPU with CUDA support (recommended for training)
- **Dependencies**: uv package manager

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Gotham-Zolio/so101-grasp-cube.git
cd so101-grasp-cube

# Install dependencies
uv sync

# Install LeRobot submodule
git submodule update --init --recursive
cd external/lerobot
uv pip install -e .
cd ../..

# Install env-client package
uv pip install -e packages/env-client
```

### 2. Data Preparation

The project uses real robot demonstration data. Download the datasets:

```bash
# Create datasets directory
mkdir -p datasets

# Download datasets (replace with actual download commands)
# TODO: Add actual dataset download commands
```

Convert existing data to LeRobot format if needed:

```bash
# Convert HDF5 trajectories to LeRobot parquet format
uv run python scripts/convert_h5_to_lerobot_parquet.py --input-dir real_data/lift --output-dir datasets/lift

# Convert trajectory data
uv run python scripts/convert_trajectory_to_lerobot.py --input-dir real_data/lift --output-dir datasets/lift
```

### 3. Training ACT Policy

Train ACT policies for each task:

```bash
# Train Lift task (6D actions)
uv run python scripts/train_act_real_data.py \
    --task lift \
    --output-dir checkpoints/lift_act \
    --epochs 100 \
    --batch-size 8 \
    --learning-rate 1e-4

# Train Sort task (12D actions)
uv run python scripts/train_act_real_data.py \
    --task sort \
    --output-dir checkpoints/sort_act \
    --epochs 100 \
    --batch-size 8 \
    --learning-rate 1e-4

# Train Stack task (6D actions)
uv run python scripts/train_act_real_data.py \
    --task stack \
    --output-dir checkpoints/stack_act \
    --epochs 100 \
    --batch-size 8 \
    --learning-rate 1e-4
```

### 4. Simulation Evaluation

Evaluate trained policies in simulation:

```bash
# Evaluate Lift policy
uv run python scripts/eval_sim_policy.py \
    --policy-path checkpoints/lift_act \
    --task lift \
    --num-episodes 50 \
    --output-dir eval_results/lift

# Evaluate Sort policy
uv run python scripts/eval_sim_policy.py \
    --policy-path checkpoints/sort_act \
    --task sort \
    --num-episodes 50 \
    --output-dir eval_results/sort

# Evaluate Stack policy
uv run python scripts/eval_sim_policy.py \
    --policy-path checkpoints/stack_act \
    --task stack \
    --num-episodes 50 \
    --output-dir eval_results/stack
```

Expected performance (based on current implementation):
- **Lift**: ~82% success rate
- **Sort**: ~84% success rate
- **Stack**: ~90% success rate

## 🤖 Real Robot Deployment

### Server Setup (Policy Server)

1. **Package the policy server**:

```bash
# Build Docker image for policy server
docker build -t so101-act-server -f docker/Dockerfile.server .

# Or run directly
uv run python grasp_cube/real/serve_act_policy.py \
    --policy-path checkpoints/lift_act \
    --host 0.0.0.0 \
    --port 8000
```

### Client Setup (Robot Environment)

1. **Install dependencies**:

```bash
# Create separate environment for robot client
uv venv robot_env
source robot_env/bin/activate  # On Windows: robot_env\Scripts\activate
uv pip install -e packages/env-client
```

2. **Test with simulated environment**:

```bash
# Run fake environment client for testing
uv run python grasp_cube/real/run_fake_env_client.py \
    --dataset-path datasets/lift \
    --host localhost \
    --port 8000
```

3. **Deploy on real robot**:

```bash
# Run real robot evaluation
uv run python scripts/eval_real_policy.py \
    --policy-server ws://robot-server:8000 \
    --task lift \
    --num-episodes 10 \
    --output-dir real_eval_results/lift
```

### Monitoring

Access the monitoring dashboard at `http://localhost:9000` during evaluation to:
- View real-time policy execution
- Monitor success/failure rates
- Control evaluation flow (start/stop/reset)

## 📊 Project Structure

```
so101-grasp-cube/
├── grasp_cube/                    # Main package
│   ├── envs/tasks/               # Simulation environments
│   │   ├── lift_cube_so101.py    # Lift task (6D)
│   │   ├── sort_cube_so101.py    # Sort task (12D)
│   │   └── stack_cube_so101.py   # Stack task (6D)
│   ├── real/                     # Real robot integration
│   │   ├── act_policy.py         # ACT policy implementation (417 lines)
│   │   ├── serve_act_policy.py   # Policy server
│   │   ├── run_env_client.py     # Robot client
│   │   └── monitor_wrapper.py    # Monitoring infrastructure
│   ├── utils/                    # Utilities
│   │   └── image_distortion.py   # Camera distortion correction
│   └── motionplanning/           # Motion planning (legacy)
├── scripts/                      # Training and evaluation scripts
│   ├── train_act_real_data.py    # ACT training script
│   ├── eval_sim_policy.py        # Simulation evaluation
│   ├── eval_real_policy.py       # Real robot evaluation
│   └── convert_*.py              # Data conversion utilities
├── real_data/                    # Real robot demonstration data
│   ├── lift/                     # Lift task data
│   ├── sort/                     # Sort task data
│   └── stack/                    # Stack task data
├── packages/env-client/          # Client package for robot communication
├── external/lerobot/             # LeRobot submodule
├── checkpoints/                  # Trained model checkpoints
├── eval_results/                 # Evaluation results
└── docker/                       # Docker deployment files
```

## 🔧 Configuration

### Training Configuration

Key training parameters (in `scripts/train_act_real_data.py`):

```python
# ACT Configuration
config = ACTConfig(
    n_obs_steps=2,              # Observation history length
    n_action_steps=16,          # Action chunk size
    n_latents=256,              # Latent dimension
    n_heads=8,                  # Attention heads
    n_encoder_layers=4,         # Encoder layers
    n_decoder_layers=6,         # Decoder layers
)

# Training parameters
epochs = 100
batch_size = 8
learning_rate = 1e-4
```

### Environment Configuration

Tasks are configured in `grasp_cube/envs/tasks/`:

- **Lift**: 6D action space (gripper pose + open/close)
- **Sort**: 12D action space (dual-arm coordination)
- **Stack**: 6D action space (precise placement)

## 🐳 Docker Deployment

### Build Policy Server Image

```bash
# Build the Docker image
docker build -t so101-act-server:latest -f docker/Dockerfile.server .

# Run the container
docker run -p 8000:8000 so101-act-server:latest
```

### Build Complete Environment

```bash
# Build full environment image
docker build -t so101-grasp-cube:latest -f docker/Dockerfile .

# Run with GPU support
docker run --gpus all -p 9000:9000 so101-grasp-cube:latest
```

## 📈 Performance Benchmarks

### Simulation Results
- **Lift Task**: 82% success rate (50 episodes)
- **Sort Task**: 84% success rate (50 episodes)
- **Stack Task**: 90% success rate (50 episodes)

### Real Robot Metrics
- **Inference Latency**: <100ms per action chunk
- **Memory Usage**: <2GB GPU memory
- **Network Latency**: <50ms end-to-end

## 🔍 Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   ```bash
   # Reduce batch size
   --batch-size 4
   # Or use gradient accumulation
   ```

2. **LeRobot import errors**:
   ```bash
   # Ensure submodule is initialized
   git submodule update --init --recursive
   cd external/lerobot && pip install -e .
   ```

3. **Robot connection failed**:
   ```bash
   # Check network connectivity
   ping robot-server
   # Verify WebSocket port
   telnet robot-server 8000
   ```

### Debug Mode

Run with verbose logging:

```bash
# Training with debug output
uv run python scripts/train_act_real_data.py --task lift --debug

# Evaluation with visualization
uv run python scripts/eval_sim_policy.py --policy-path checkpoints/lift_act --vis
```

## 📝 Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{so101-grasp-cube,
  title={SO101 Cube Grasping with ACT},
  author={Gotham-Zolio},
  year={2024},
  url={https://github.com/Gotham-Zolio/so101-grasp-cube}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LeRobot**: For the ACT implementation and dataset format
- **ManiSkill**: For the simulation environment
- **SO101 Robot**: For the hardware platform
- **Action Chunking with Transformers**: Original ACT paper and implementation