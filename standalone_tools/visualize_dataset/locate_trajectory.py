#!/usr/bin/env python3
"""
Script to locate trajectory files by trajectory ID.

Usage:
    python locate_trajectory.py <trajectory_id> [--data-root DATA_ROOT]

Example:
    python locate_trajectory.py 42
    python locate_trajectory.py 42 --data-root /path/to/data
"""

import os
import sys
import json
import argparse
from pathlib import Path

def locate_trajectory_files(trajectory_id, data_root="/home/zhangjinyu/code_repo/starVLA/data/task_3124"):
    """
    Locate all files associated with a given trajectory ID.

    Args:
        trajectory_id (int): The trajectory/episode index
        data_root (str): Root directory of the dataset

    Returns:
        dict: Dictionary containing file paths and metadata
    """
    data_root = Path(data_root)

    # Load episode metadata
    episodes_meta_path = data_root / "meta" / "episodes.jsonl"
    episode_info = None

    if episodes_meta_path.exists():
        with open(episodes_meta_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    episode = json.loads(line.strip())
                    if episode["episode_index"] == trajectory_id:
                        episode_info = episode
                        break
                except json.JSONDecodeError:
                    continue

    if episode_info is None:
        return {"error": f"Trajectory ID {trajectory_id} not found in episodes metadata"}

    # Calculate chunk information
    chunk_size = 1000  # From info.json
    chunk_index = trajectory_id // chunk_size
    episode_in_chunk = trajectory_id % chunk_size

    # File paths
    data_file = data_root / f"data/chunk-{chunk_index:03d}/episode_{trajectory_id:06d}.parquet"

    video_files = {}
    video_keys = ["observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right"]

    for video_key in video_keys:
        # Convert video key to directory name
        dir_name = video_key.replace("observation.images.", "").replace(".", "_")
        video_file = data_root / f"videos/chunk-{chunk_index:03d}/{dir_name}/episode_{trajectory_id:06d}.mp4"
        video_files[video_key] = video_file

    result = {
        "trajectory_id": trajectory_id,
        "episode_info": episode_info,
        "chunk_info": {
            "chunk_index": chunk_index,
            "chunk_size": chunk_size,
            "episode_in_chunk": episode_in_chunk
        },
        "files": {
            "data_file": str(data_file),
            "video_files": {k: str(v) for k, v in video_files.items()}
        },
        "file_exists": {
            "data_file": data_file.exists(),
            "video_files": {k: v.exists() for k, v in video_files.items()}
        }
    }

    return result

def print_trajectory_info(info):
    """Print trajectory information in a readable format."""
    if "error" in info:
        print(f"❌ Error: {info['error']}")
        return

    print(f"📍 Trajectory ID: {info['trajectory_id']}")
    print(f"📋 Task: {info['episode_info']['tasks'][0]}")
    print(f"⏱️  Length: {info['episode_info']['length']} steps")
    print()

    print("📁 Chunk Information:")
    chunk = info['chunk_info']
    print(f"   Chunk Index: {chunk['chunk_index']}")
    print(f"   Chunk Size: {chunk['chunk_size']}")
    print(f"   Position in Chunk: {chunk['episode_in_chunk']}")
    print()

    print("📂 File Locations:")
    print(f"   Data File: {info['files']['data_file']}")
    print(f"   Status: {'✅ Exists' if info['file_exists']['data_file'] else '❌ Missing'}")
    print()

    print("🎥 Video Files:")
    for key, path in info['files']['video_files'].items():
        exists = info['file_exists']['video_files'][key]
        camera_name = key.split('.')[-1]
        print(f"   {camera_name}: {path}")
        print(f"   Status: {'✅ Exists' if exists else '❌ Missing'}")
    print()

def main():
    parser = argparse.ArgumentParser(description="Locate trajectory files by trajectory ID")
    parser.add_argument("trajectory_id", type=int, help="Trajectory ID to locate")
    parser.add_argument("--data-root", type=str,
                       default="/home/zhangjinyu/code_repo/starVLA/data/task_3124",
                       help="Root directory of the dataset")

    args = parser.parse_args()

    # Validate trajectory ID
    if args.trajectory_id < 0:
        print("❌ Error: Trajectory ID must be non-negative")
        sys.exit(1)

    # Locate files
    info = locate_trajectory_files(args.trajectory_id, args.data_root)

    # Print results
    print_trajectory_info(info)

if __name__ == "__main__":
    main()
