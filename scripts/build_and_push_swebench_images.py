#!/usr/bin/env python3
"""Build SWE-bench Docker images and push to Docker Hub.

Run this on a machine WITH Docker build privileges (not RunPod).
Then on RunPod, just 'docker pull' the images.

Usage:
    # Build + push first 2 instances (for testing)
    python scripts/build_and_push_swebench_images.py \
        --docker-hub-user YOUR_USERNAME \
        --limit 2

    # Build + push all instances
    python scripts/build_and_push_swebench_images.py \
        --docker-hub-user YOUR_USERNAME

    # On RunPod, pull the images:
    docker pull YOUR_USERNAME/sweb.eval.x86_64.astropy__astropy-11693:latest
"""

import argparse
import subprocess
import sys

import docker
from datasets import load_dataset
from swebench.harness.docker_build import (
    build_env_images,
    build_instance_images,
)
from swebench.harness.test_spec.test_spec import (
    get_test_specs_from_dataset,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker-hub-user", required=True)
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_oracle")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    client = docker.from_env()

    print(f"Loading dataset {args.dataset} split={args.split}...")
    ds = load_dataset(args.dataset, split=args.split)
    instances = list(ds)
    if args.limit > 0:
        instances = instances[: args.limit]
    print(f"Building images for {len(instances)} instances")

    print("\n=== Building env images ===")
    build_env_images(
        client,
        instances,
        max_workers=args.max_workers,
        instance_image_tag="latest",
        env_image_tag="latest",
    )

    print("\n=== Building instance images ===")
    build_instance_images(
        client,
        instances,
        max_workers=args.max_workers,
        tag="latest",
        env_image_tag="latest",
    )

    print("\n=== Tagging and pushing to Docker Hub ===")
    test_specs = get_test_specs_from_dataset(instances)
    for spec in test_specs:
        local_tag = f"sweb.eval.x86_64.{spec.instance_id}:latest"
        remote_tag = f"{args.docker_hub_user}/{local_tag}"

        print(f"  Tagging {local_tag} → {remote_tag}")
        try:
            img = client.images.get(local_tag)
            img.tag(remote_tag)
            print(f"  Pushing {remote_tag}...")
            client.images.push(remote_tag)
            print(f"  ✓ {remote_tag}")
        except Exception as e:
            print(f"  ✗ {local_tag}: {e}")

    print("\n=== Done ===")
    print(f"On RunPod, pull images with:")
    print(f"  docker pull {args.docker_hub_user}/sweb.eval.x86_64.<instance_id>:latest")


if __name__ == "__main__":
    main()
