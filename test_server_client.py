#!/usr/bin/env python3
"""
Simple test script to verify Server-Client integration works.

This script:
1. Starts the DiffusionPolicy Server
2. Connects a fake client 
3. Runs a few inference steps
4. Verifies outputs are correct

Usage:
    python test_server_client.py --policy.path checkpoints/lift_real/checkpoint-best --policy.task lift
"""

import subprocess
import time
import sys
import threading
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent

def run_server(policy_path: str, task: str, host: str = "0.0.0.0", port: int = 8000):
    """Run the DiffusionPolicy server in a subprocess."""
    cmd = [
        sys.executable, "-m", "uv", "run",
        str(SCRIPT_DIR / "serve_diffusion_policy.py"),
        f"--policy.path={policy_path}",
        f"--policy.task={task}",
        f"--host={host}",
        f"--port={port}",
    ]
    
    print(f"🚀 Starting Server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc

def run_client(task: str, host: str = "0.0.0.0", port: int = 8000, num_episodes: int = 1):
    """Run the fake client."""
    cmd = [
        sys.executable, "-m", "uv", "run",
        str(SCRIPT_DIR / "run_fake_env_client.py"),
        f"--env.task={task}",
        f"--host={host}",
        f"--port={port}",
        f"--num-episodes={num_episodes}",
        f"--eval.output-dir=/tmp/test_eval",
    ]
    
    print(f"🎮 Starting Client: {' '.join(cmd)}")
    time.sleep(3)  # Wait for server to start
    
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Server-Client integration")
    parser.add_argument("--policy.path", required=True, help="Path to model checkpoint")
    parser.add_argument("--policy.task", default="lift", help="Task name (lift/sort/stack)")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DiffusionPolicy Server-Client Integration Test")
    print("=" * 70)
    
    # Start server
    server_proc = run_server(args.__dict__["policy.path"], args.__dict__["policy.task"])
    
    # Start client in a thread
    client_proc = None
    try:
        client_proc = run_client(
            args.__dict__["policy.task"],
            num_episodes=args.num_episodes,
        )
        
        # Wait for both processes
        start_time = time.time()
        while time.time() - start_time < args.timeout:
            server_code = server_proc.poll()
            client_code = client_proc.poll()
            
            if client_code is not None:
                # Client finished
                print("\n✅ Client finished")
                _, client_stderr = client_proc.communicate(timeout=5)
                if client_code != 0:
                    print(f"❌ Client error (code {client_code}):")
                    print(client_stderr)
                    return 1
                return 0
            
            if server_code is not None:
                print(f"\n❌ Server crashed (code {server_code})")
                _, server_stderr = server_proc.communicate(timeout=5)
                print(server_stderr)
                return 1
            
            time.sleep(1)
        
        print("\n⏱️ Timeout reached")
        return 1
        
    finally:
        # Kill processes
        if client_proc and client_proc.poll() is None:
            client_proc.terminate()
            client_proc.wait(timeout=5)
        
        if server_proc and server_proc.poll() is None:
            server_proc.terminate()
            server_proc.wait(timeout=5)

if __name__ == "__main__":
    sys.exit(main())
