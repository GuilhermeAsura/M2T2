#!/usr/bin/env python3
"""Standalone smoke-test client for the M2T2 ZMQ server (grasping/pick mode).

Loads a sample scene the same way demo.py does (via load_rgb_xyz), sends it to
the server through the `infer` action, and prints how many grasps came back
per object, for comparison against demo.py's own output on the same scene.

Usage:
    python3 client-server/m2t2_client_test.py \
        --data_dir sample_data/real_world/00 --host localhost --port 5556
"""
import argparse

import msgpack
import msgpack_numpy
import numpy as np
import zmq

from m2t2.dataset import load_rgb_xyz

msgpack_numpy.patch()


def parse_args():
    parser = argparse.ArgumentParser(description='Smoke-test the M2T2 ZMQ server')
    parser.add_argument('--data_dir', type=str, default='sample_data/real_world/00')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5556)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--mask_thresh', type=float, default=0.4)
    return parser.parse_args()


def request(sock, payload):
    sock.send(msgpack.packb(payload, use_bin_type=True))
    response = msgpack.unpackb(sock.recv(), raw=False)
    if 'error' in response:
        raise RuntimeError(f"Server error: {response['error']}")
    return response


def main():
    args = parse_args()

    # robot_prob=1.0, world_coord=True, jitter_scale=0, grid_res=0.01: same
    # defaults config.yaml uses for eval/demo.py.
    data, _ = load_rgb_xyz(args.data_dir, 1.0, True, 0.0, 0.01)
    xyz = data['points'].numpy().astype(np.float32)
    rgb = data['inputs'][:, 3:].numpy().astype(np.float32)
    print(f'Loaded {xyz.shape[0]} points from {args.data_dir}')

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f'tcp://{args.host}:{args.port}')

    print('health:', request(sock, {'action': 'health'}))
    print('metadata:', request(sock, {'action': 'metadata'}))

    response = request(sock, {
        'action': 'infer',
        'points': xyz,
        'rgb': rgb,
        'num_runs': args.num_runs,
        'mask_thresh': args.mask_thresh,
    })

    for i, grasps in enumerate(response['grasps']):
        print(f'object_{i:02d} has {len(grasps)} grasps')


if __name__ == '__main__':
    main()
