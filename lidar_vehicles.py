import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN


def transform(record):
    q = record['rotation']  # nuScenes uses w,x,y,z
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t[:3, 3] = record['translation']
    return t


def apply(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def box(points):
    center = points[:, :2].mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((points[:, :2] - center).T))
    axes = vectors[:, ::-1]
    local = (points[:, :2] - center) @ axes
    lo, hi = local.min(axis=0), local.max(axis=0)
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                        [hi[0], hi[1]], [lo[0], hi[1]]]) @ axes.T + center
    c = ((lo + hi) / 2) @ axes.T + center
    return corners, c, np.sort(hi - lo)[::-1]


def detect(raw, rng, radius):
    # Work in ego coordinates: x forward, y left, z up.
    mask = ((np.linalg.norm(raw[:, :2], axis=1) < radius)
            & (np.linalg.norm(raw[:, :2], axis=1) > 2.5)
            & (raw[:, 2] > -2.5) & (raw[:, 2] < 4))
    cropped = raw[mask]
    _, indices = np.unique(np.floor(cropped / .18).astype(int), axis=0, return_index=True)
    points = cropped[indices]
    # Fit a near-horizontal plane using low points, never arbitrary facades.
    low = points[(points[:, 2] < .35) & (points[:, 2] > -1.5)]
    best, plane = -1, (np.array([0., 0., 1.]), 0.)
    if len(low) >= 3:
        for _ in range(180):
            a, b, c = low[rng.choice(len(low), 3, replace=False)]
            normal = np.cross(b - a, c - a)
            norm = np.linalg.norm(normal)
            if norm < 1e-8:
                continue
            normal /= norm
            if normal[2] < 0:
                normal *= -1
            if normal[2] < .97:
                continue
            offset = -normal @ a
            score = np.count_nonzero(np.abs(low @ normal + offset) < .16)
            if score > best:
                best, plane = score, (normal.copy(), offset)
    normal, offset = plane
    heights = points @ normal + offset
    foreground = points[(heights > .22) & (heights < 3.5)]
    labels = (DBSCAN(eps=.85, min_samples=5).fit_predict(foreground)
              if len(foreground) else np.empty(0, dtype=int))
    candidates = []
    for label in sorted(set(labels) - {-1}):
        cluster = foreground[labels == label]
        if len(cluster) < 10:
            continue
        corners, center, dimensions = box(cluster)
        length, width = dimensions
        height = np.ptp(cluster[:, 2])
        base = np.min(cluster @ normal + offset)
        # Broad observed-size limits allow partial car returns; not a classifier.
        if 1.5 < length < 6.5 and .65 < width < 3.2 and .45 < height < 2.8 and base < 1.0:
            quality = 'strong' if (2.5 < length < 5.5 and 1.1 < width < 2.6
                                    and len(cluster) >= 30) else 'uncertain'
            candidates.append(dict(points=cluster, corners=corners, center=center,
                                   dimensions=[length, width, height], quality=quality))
    return points, foreground, labels, candidates, best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('F:/nuscenes'))
    parser.add_argument('--output', type=Path, default=Path('output'))
    parser.add_argument('--scene', default='scene-0061')
    parser.add_argument('--frames', type=int, default=40)
    parser.add_argument('--radius', type=float, default=40)
    parser.add_argument('--list-scenes', action='store_true')
    args = parser.parse_args()
    meta = args.data / 'v1.0-mini'
    def read(name):
        return json.loads((meta / (name + '.json')).read_text())
    scenes = read('scene')
    if args.list_scenes:
        for scene in scenes:
            print(scene['name'], ':', scene['description'])
        return
    scene = next((s for s in scenes if s['name'] == args.scene), None)
    if scene is None:
        parser.error('Unknown scene; use --list-scenes')
    if args.frames < 1:
        parser.error('--frames must be positive')
    tables = {name: {r['token']: r for r in read(name)} for name in
              ['sample', 'sample_data', 'calibrated_sensor', 'sensor', 'ego_pose',
               'sample_annotation', 'instance', 'category']}
    lidar = {}
    annotations_by_sample = {}
    for annotation in tables['sample_annotation'].values():
        annotations_by_sample.setdefault(annotation['sample_token'], []).append(annotation['token'])
    for record in tables['sample_data'].values():
        calibration = tables['calibrated_sensor'][record['calibrated_sensor_token']]
        sensor = tables['sensor'][calibration['sensor_token']]
        if sensor['channel'] == 'LIDAR_TOP' and record['is_key_frame']:
            lidar[record['sample_token']] = record
    out = args.output
    frames_dir = out / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    token, index, next_id = scene['first_sample_token'], 0, 1
    tracks, rows, evaluations, images = {}, [], [], []
    while token and index < args.frames:
        sample = tables['sample'][token]
        scan = lidar[token]
        calibration = transform(tables['calibrated_sensor'][scan['calibrated_sensor_token']])
        pose = transform(tables['ego_pose'][scan['ego_pose_token']])
        raw = np.fromfile(args.data / scan['filename'], dtype=np.float32).reshape(-1, 5)[:, :3]
        raw = apply(raw, calibration)
        points, foreground, labels, detections, plane_support = detect(raw, rng, args.radius)
        time = scan['timestamp'] / 1e6
        centers = [apply(np.array([[*d['center'], 0]]), pose)[0, :2] for d in detections]
        active = {k: v for k, v in tracks.items() if time - v['time'] <= 1.5}
        assigned = {}
        ids = list(active)
        if ids and centers:
            predicted = np.array([active[k]['center'] + active[k]['velocity'] *
                                  (time - active[k]['time']) for k in ids])
            costs = np.linalg.norm(predicted[:, None, :] - np.array(centers)[None, :, :], axis=2)
            rr, cc = linear_sum_assignment(costs)
            for r, c in zip(rr, cc):
                if costs[r, c] < 4:
                    assigned[c] = ids[r]
        for j, detection in enumerate(detections):
            identity = assigned.get(j)
            if identity is None:
                identity, next_id = next_id, next_id + 1
            old = tracks.get(identity)
            velocity = (centers[j] - old['center']) / (time - old['time']) if old else np.zeros(2)
            tracks[identity] = dict(center=centers[j], velocity=velocity, time=time)
            detection['id'] = identity
            rows.append(dict(frame=index, timestamp=time, track_id=identity,
                             x_m=float(detection['center'][0]), y_m=float(detection['center'][1]),
                             distance_m=float(np.linalg.norm(detection['center'])),
                             length_m=float(detection['dimensions'][0]), width_m=float(detection['dimensions'][1]),
                             height_m=float(detection['dimensions'][2]), points=len(detection['points']),
                             quality=detection['quality']))
        tracks = {k: v for k, v in tracks.items() if time - v['time'] <= 1.5}
        # Reference annotations enter only here, after detection and association.
        references = []
        inv_pose = np.linalg.inv(pose)
        for ann_token in annotations_by_sample.get(token, []):
            ann = tables['sample_annotation'][ann_token]
            instance = tables['instance'][ann['instance_token']]
            category = tables['category'][instance['category_token']]['name']
            if category != 'vehicle.car' or ann['num_lidar_pts'] < 5:
                continue
            c = apply(np.array([ann['translation']]), inv_pose)[0]
            if not 2.5 < np.linalg.norm(c[:2]) < args.radius:
                continue
            width, length, _ = ann['size']
            local = np.array([[-length/2, -width/2, 0], [length/2, -width/2, 0],
                              [length/2, width/2, 0], [-length/2, width/2, 0]])
            corners = apply(apply(local, transform(ann)), inv_pose)[:, :2]
            references.append((c[:2], corners))
        matches = 0
        if detections and references:
            distances = np.linalg.norm(np.array([d['center'] for d in detections])[:, None, :] -
                                       np.array([r[0] for r in references])[None, :, :], axis=2)
            # Maximize number of matches within 2 m before minimizing distance.
            costs = np.where(distances < 2., distances, 1e6)
            r, c = linear_sum_assignment(costs)
            matches = int(np.sum(distances[r, c] < 2.))
        evaluations.append(dict(frame=index, detections=len(detections), reference_cars=len(references),
                                matched=matches, false_positives=len(detections)-matches,
                                missed=len(references)-matches))
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.scatter(points[:, 0], points[:, 1], s=.5, c='#b4bdc8', rasterized=True)
        for _, corners in references:
            ax.add_patch(Polygon(corners, fill=False, edgecolor='#21a85b', linestyle='--', linewidth=1))
        for d in detections:
            color = plt.get_cmap('tab20')(d['id'] % 20)
            ax.scatter(d['points'][:, 0], d['points'][:, 1], s=3, color=color)
            ax.add_patch(Polygon(d['corners'], fill=False, edgecolor=color, linewidth=2))
            x, y = d['center']
            ax.text(x, y, f"ID {d['id']} | {np.hypot(x,y):.1f}m\n{d['quality']}", fontsize=6,
                    bbox=dict(facecolor='white', alpha=.75, edgecolor='none'))
        ax.plot(0, 0, '^', color='black', markersize=9)
        ax.set(xlim=(-args.radius, args.radius), ylim=(-args.radius, args.radius),
               xlabel='Forward x (m)', ylabel='Left y (m)', aspect='equal',
               title=f"{args.scene} | frame {index:03d} | {len(detections)} vehicle candidates\n"
                     'Solid: detections | dashed green: reference cars (evaluation only)')
        ax.grid(alpha=.2)
        fig.tight_layout()
        filename = frames_dir / f'frame_{index:03d}.png'
        fig.savefig(filename, dpi=110)
        plt.close(fig)
        with Image.open(filename) as im:
            images.append(im.convert('RGB').resize((700, 700)))
        if index in [0, 10, 20]:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            for axis, cloud, title, colors in zip(axes, [points, foreground, foreground],
                    ['Cropped + voxelized', 'Ground removed', 'DBSCAN clusters'],
                    ['#65758b', '#65758b', labels]):
                axis.scatter(cloud[:, 0], cloud[:, 1], s=.6, c=colors, cmap='tab20' if not isinstance(colors, str) else None)
                axis.set(xlim=(-40, 40), ylim=(-40, 40), title=title, aspect='equal', xlabel='x (m)', ylabel='y (m)')
            fig.tight_layout()
            fig.savefig(out / f'stages_{index:03d}.png', dpi=130)
            plt.close(fig)
        print(f"Frame {index:03d}: {len(detections)} candidates, {matches}/{len(references)} reference cars matched", flush=True)
        token, index = sample['next'], index + 1
    if images:
        images[0].save(out / 'detections.gif', save_all=True, append_images=images[1:], duration=500, loop=0)
    for name, records in [('detections', rows), ('evaluation', evaluations)]:
        with (out / f'{name}.csv').open('w', newline='') as f:
            fields = list(records[0]) if records else ['frame']
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    totals = {k: sum(r[k] for r in evaluations) for k in evaluations[0] if k != 'frame'}
    totals.update(precision=totals['matched']/max(totals['detections'], 1),
                  recall=totals['matched']/max(totals['reference_cars'], 1))
    summary = dict(scene=args.scene, description=scene['description'], frames=index, radius_m=args.radius,
                   evaluation='One-to-one ego-frame BEV centre matching <2 m; reference cars with >=5 LiDAR points. Not official nuScenes AP.',
                   metrics=totals, seed=42)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    best_frame = max(evaluations, key=lambda r: (r['matched'], -r['false_positives']))
    hard_frame = max(evaluations, key=lambda r: r['missed'] + r['false_positives'])
    (out / 'index.html').write_text(f'''<!doctype html><html><head><meta charset="utf-8">
    <title>LiDAR vehicle candidates</title><style>body{{font:16px system-ui;max-width:1100px;margin:30px auto;background:#f4f6fa;color:#172238}}
    img{{max-width:100%}}section{{background:white;padding:20px;margin:20px 0}}pre{{white-space:pre-wrap}}</style></head><body>
    <h1>{args.scene}: LiDAR vehicle candidates</h1><p>Solid boxes are geometric detections. Dashed green boxes are reference cars used only for evaluation.
    Strong/uncertain indicates a geometry heuristic, not a probability. IDs use ego-compensated association and may switch.</p>
    <section><h2>Sequence</h2><img src="detections.gif"></section>
    <section><h2>Frame with most matches: {best_frame['frame']}</h2><img src="frames/frame_{best_frame['frame']:03d}.png"></section>
    <section><h2>Frame with most errors: {hard_frame['frame']}</h2><img src="frames/frame_{hard_frame['frame']:03d}.png"></section>
    <section><h2>Processing stages</h2><img src="stages_000.png"></section>
    <section><h2>Evaluation</h2><pre>{json.dumps(summary, indent=2)}</pre>
    <p>Counts are per-frame observations, not unique vehicles. Matching uses a 2 m centre threshold and excludes reference cars with fewer than five LiDAR points.
    False positives can include other vehicle types because evaluation targets cars only. This is not official nuScenes AP.</p>
    <a href="detections.csv">Detections CSV</a> | <a href="evaluation.csv">Per-frame evaluation CSV</a></section></body></html>''', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
