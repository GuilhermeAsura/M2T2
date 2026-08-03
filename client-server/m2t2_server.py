#!/usr/bin/env python3
"""Headless ZMQ inference server for M2T2, grasping (pick) mode only.

Loads the checkpoint once at startup and serves inference requests over a
ZMQ REP socket. Unlike demo.py, this never touches meshcat/create_visualizer,
so it does not depend on a meshcat-server being reachable.

Usage:
    python3 client-server/m2t2_server.py --checkpoint weights/m2t2.pth --port 5556
"""
import argparse
import logging

import msgpack
import msgpack_numpy
import torch
from omegaconf import OmegaConf
import zmq

from m2t2.dataset import collate
from m2t2.dataset_utils import sample_points
from m2t2.m2t2 import M2T2
from m2t2.train_utils import to_cpu, to_gpu

msgpack_numpy.patch()

logger = logging.getLogger(__name__)


def build_data_from_arrays(xyz, rgb):
    """xyz: Nx3 tensor in world coordinates. rgb: Nx3 tensor, normalized as normalize_rgb would."""
    return {
        'inputs': torch.cat([xyz - xyz.mean(dim=0), rgb], dim=1),
        'points': xyz,
        'task': 'pick',
        'object_inputs': torch.rand(1024, 6),  # placeholder, unused in pick mode
        'ee_pose': torch.eye(4),
        'cam_pose': torch.eye(4),
        # M2T2's place-prediction branch always runs (num_place_queries > 0
        # in config.yaml, regardless of task), so bottom_center is required
        # even in pick mode. Same default load_rgb_xyz uses for pick scenes.
        'bottom_center': torch.zeros(3),
    }


def load_model(checkpoint_path, model_cfg):
    model = M2T2.from_config(model_cfg)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    return model.cuda().eval()


def infer(model, xyz, rgb, cfg, num_runs, mask_thresh):
    data = build_data_from_arrays(xyz, rgb)
    eval_cfg = OmegaConf.merge(cfg.eval, {'mask_thresh': mask_thresh})
    inputs, points = data['inputs'], data['points']
    outputs = {'grasps': [], 'grasp_confidence': [], 'grasp_contacts': []}
    for _ in range(num_runs):
        pt_idx = sample_points(points, cfg.data.num_points)
        data['inputs'] = inputs[pt_idx]
        data['points'] = points[pt_idx]
        batch = collate([data])
        to_gpu(batch)
        with torch.no_grad():
            model_outputs = model.infer(batch, eval_cfg)
        to_cpu(model_outputs)
        for key in outputs:
            outputs[key].extend(model_outputs[key][0])
    return outputs


def handle_request(model, cfg, request):
    action = request.get('action')
    if action == 'health':
        return {'status': 'ok'}
    if action == 'metadata':
        return {'gripper_name': 'franka_panda_2f', 'model_name': 'm2t2'}
    if action == 'infer':
        xyz = torch.from_numpy(request['points']).float()
        rgb = torch.from_numpy(request['rgb']).float()
        num_runs = int(request.get('num_runs', cfg.eval.num_runs))
        mask_thresh = float(request.get('mask_thresh', cfg.eval.mask_thresh))
        out = infer(model, xyz, rgb, cfg, num_runs, mask_thresh)
        return {
            'grasps': [g.numpy() for g in out['grasps']],
            'confidence': [c.numpy() for c in out['grasp_confidence']],
            'contacts': [c.numpy() for c in out['grasp_contacts']],
        }
    return {'error': f'Unknown action: {action}'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Start an M2T2 ZMQ inference server (grasping/pick mode only)'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to the M2T2 checkpoint (e.g. weights/m2t2.pth)'
    )
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to the M2T2 hydra config (default: config.yaml)'
    )
    parser.add_argument(
        '--host', type=str, default='0.0.0.0',
        help='Address to bind the ZMQ socket (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port', type=int, default=5556,
        help='Port to bind the ZMQ socket (default: 5556)'
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    args = parse_args()

    cfg = OmegaConf.load(args.config)
    logger.info('Loading M2T2 checkpoint from %s', args.checkpoint)
    model = load_model(args.checkpoint, cfg.m2t2)
    logger.info('Modelo carregado. M2T2 pronto para inferencia.')

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    bind_addr = f'tcp://{args.host}:{args.port}'
    sock.bind(bind_addr)
    logger.info('M2T2 ZMQ server listening on %s', bind_addr)

    try:
        while True:
            raw = sock.recv()
            try:
                request = msgpack.unpackb(raw, raw=False)
                response = handle_request(model, cfg, request)
            except Exception as exc:
                logger.exception('Error handling request')
                response = {'error': str(exc)}
            sock.send(msgpack.packb(response, use_bin_type=True))
    except KeyboardInterrupt:
        logger.info('Shutting down server')
    finally:
        sock.close()
        ctx.term()


if __name__ == '__main__':
    main()
