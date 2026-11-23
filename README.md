# Finding Vehicles in LiDAR Point Clouds

Which groups of laser returns could belong to a car? This Python project answers that question using the shape, size, and position of points in a nuScenes mini scan. It then follows candidate vehicles across scans and compares their positions with annotated cars. The detector uses geometric rules and requires no model training.

![Bird’s-eye animation of detected candidates and reference cars in scene-0103](output/scene-0103/detections.gif)

**Reading the animation.** The black triangle marks the recording vehicle. The horizontal axis points forward and the vertical axis points left. A solid box encloses the visible points of a candidate. A dashed green box shows an annotated car. Each label gives a provisional track ID, horizontal distance from the recording vehicle, and a geometry-based quality flag.

[Run the project](#run-the-project) · [Follow one scan](#from-a-scan-to-vehicle-candidates) · [Read the results](#results-on-two-scenes)

## Run the project

To explore the saved runs immediately, open `output/index.html` or `output/scene-0103/index.html` locally in a browser. Each viewer includes an animation, example frames, processing stages, and evaluation counts.

To process scans yourself, download and extract **nuScenes v1.0-mini** separately. The dataset root should contain `v1.0-mini/` (metadata) and `samples/LIDAR_TOP/` (scans). From the directory containing `lidar_vehicles.py`, run these commands in PowerShell. Replace the dataset path with your own.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$data = 'F:/nuscenes'
.\.venv\Scripts\python.exe lidar_vehicles.py --data $data --list-scenes
.\.venv\Scripts\python.exe lidar_vehicles.py --data $data --scene scene-0061 --frames 40 --output runs/scene-0061
Start-Process ./runs/scene-0061/index.html
```

The frame count is an upper limit. For the second demonstration, use `--scene scene-0103 --output runs/scene-0103`. The default search radius is 40 m. To run the two dataset-independent checks for coordinate conversion and synthetic car detection, use `.\.venv\Scripts\python.exe -m unittest -v test_pipeline`.

## From a scan to vehicle candidates

![Workflow from raw measurements through detection, tracking and evaluation](docs/screenshots/workflow.png)

### 1. Read the points and locate them relative to the vehicle

One laser return gives one point on a surface. A roof-mounted LiDAR might measure points at (5.0, 1.0, −1.8), (10.0, 2.0, −0.5), and (10.2, 2.1, 0.2) metres. The last two are close together, but their coordinates alone cannot tell us which object produced them. The script reads the first three values (XYZ) of each five-float LiDAR record.

<details>
<summary>What one scan contains</summary>

![The vehicle detection question for an unlabelled point cloud](docs/screenshots/question.png)

![Three example XYZ measurements in metres](docs/screenshots/xyz.png)

</details>

The sensor and the vehicle have different origins and can face different directions. Calibration converts each point with **vehicle point = rotation × sensor point + translation**. Imagine a sideways-pointing ruler mounted on the car. Rotation aligns its axes with the car, and translation accounts for where it is mounted. In vehicle coordinates, (10, 2, 0) means 10 m forward and 2 m left at the vehicle origin’s height.

<details>
<summary>Sensor axes, vehicle axes, and the sideways-ruler example</summary>

![Sensor origin and vehicle axis conventions](docs/screenshots/coordinates.png)

![Rotation aligns axes and translation shifts the origin](docs/screenshots/rotation.png)

</details>

### 2. Crop the scan and reduce repeated measurements

The code keeps points with horizontal range strictly between 2.5 and 40 m by default and vehicle-frame height strictly between −2.5 and 4 m. Range is measured in the ground plane. Assuming the height passes, a point 12 m forward and 5 m left is 13 m away and stays. A point 45 m ahead falls outside the region. A point 1 m ahead falls inside the exclusion zone around the recording vehicle.

The remaining space is divided into cubes called **voxels**, each 18 cm on a side. If five points land in one cube, the code keeps the first measured point in that cube. Smaller cubes preserve more detail but leave more points to process. Larger cubes can erase the few returns from a distant car.

<details>
<summary>Three crop decisions, five points in one voxel, and the voxel-size tradeoff</summary>

![Range calculation for points at 13, 45, and 1 metres](docs/screenshots/crop.png)

![Five nearby points reduced to one in an occupied voxel](docs/screenshots/voxel.png)

![Detail and processing tradeoffs for voxel size](docs/screenshots/voxel-size.png)

</details>

### 3. Estimate the road and remove ground points

Road points can connect two parked cars into one continuous group. To separate them, **RANSAC** tries up to 180 planes through random triples of low points (−1.5 < z < 0.35 m). A plane’s normal is a direction perpendicular to its surface. An upward-pointing normal suggests a road, while a sideways normal suggests a wall. The code considers planes with an upward normal component of at least 0.97 and chooses the one with the most low points within 0.16 m. Rejecting a plane only discards that proposed model. Point removal happens after the road estimate is chosen.

<details>
<summary>How RANSAC samples points, tests the normal, and counts support</summary>

![RANSAC plane estimation by repeated sampling](docs/screenshots/road-plane.png)

![Selection of low points and random triples](docs/screenshots/sample-plane.png)

![Plane normals and support counts](docs/screenshots/plane-normal.png)

</details>

For each point, the code measures signed distance above the estimated plane and keeps points strictly between **0.22 and 3.5 m** above it. For a level road at z = −0.50 m, the arithmetic is simple.

| Point z | Height above road | Decision |
|---:|---:|---|
| −0.48 m | 0.02 m | Remove |
| −0.40 m | 0.10 m | Remove |
| 0.30 m | 0.80 m | Keep |
| 1.00 m | 1.50 m | Keep |

<details>
<summary>Road, body, and roof returns around the 22 cm cutoff</summary>

![Road and car returns relative to the cutoff](docs/screenshots/cutoff.png)

![Four heights above a level road and their keep or remove decisions](docs/screenshots/road-height.png)

</details>

### 4. Group nearby obstacle points

**DBSCAN** looks for dense neighbourhoods in XYZ. A point with at least five points, including itself, within 0.85 m can form the core of a cluster. Connected dense neighbourhoods grow into one group. Imagine two groups of returns with a single isolated return between them. If the groups are sufficiently separated, they form two clusters, and the isolated return is labelled noise. A cluster could still belong to a car, a tree, or a barrier.

<details>
<summary>Two groups of returns and an isolated point</summary>

![Two dense groups separated by an isolated return](docs/screenshots/clusters.png)

</details>

![Actual processing stages for frame 000 of scene-0061](output/stages_000.png)

Read the panels from left to right. The first shows the cropped and downsampled scan. The second shows what remains above the road. The third colours those same points by DBSCAN label, before any vehicle-size checks.

### 5. Decide which clusters look like vehicles

Viewed from above, a parked car can sit at any angle. **Principal component analysis (PCA)** finds the direction in which a cluster’s XY points spread the most. The code aligns a rectangle with that direction and measures its length and width. Height is the highest z minus the lowest z. The resulting box describes the visible returns, so a partially hidden car may appear too short or narrow.

| Measurement | Rule for a vehicle candidate |
|---|---|
| Downsampled points | At least 10 |
| Length | 1.5 < length < 6.5 m |
| Width | 0.65 < width < 3.2 m |
| Height | 0.45 < height < 2.8 m |
| Lowest point above estimated road | Below 1 m |

A 4.0 × 1.8 × 1.4 m cluster passes if it has enough points and sits close enough to the road. A 12 m cluster fails the length check. A 0.2 m wide cluster fails the width check. Among accepted candidates, **strong** means length strictly between 2.5 and 5.5 m, width strictly between 1.1 and 2.6 m, and at least 30 points. Other accepted candidates are **uncertain**. Both appear in the evaluation, and either can be a false alarm.

<details>
<summary>Rotated rectangles, measurement rules, and three candidate decisions</summary>

![An oriented rectangle around visible vehicle points](docs/screenshots/footprint.png)

![PCA dimensions and vehicle-size thresholds](docs/screenshots/dimensions.png)

![Examples of acceptable, too long, and too narrow clusters](docs/screenshots/candidates.png)

</details>

### 6. Follow a candidate across scans

A parked car appears closer as the recording vehicle drives towards it. The tracker uses the recording vehicle’s pose to put detected centres into a common world frame. It predicts each track’s next centre from its previous centre, velocity, and elapsed time. New tracks start with zero velocity.

Suppose ID 1 is predicted at (10, 5) m. A new centre at (10.5, 5.2) is about 0.54 m away, while one at (25, 12) is about 16.55 m away. The Hungarian algorithm assigns tracks and detections one-to-one using their distances. An assigned pair keeps its ID only if it is less than **4 m** apart. An unmatched detection gets a new ID. A track expires after more than **1.5 s** without a detection.

<details>
<summary>Why ego motion matters and how a detection inherits an ID</summary>

![Ego motion and next-position prediction](docs/screenshots/tracking.png)

![Example positions and rules for reusing an ID](docs/screenshots/association.png)

</details>

## Results on two scenes

Evaluation asks a different question from tracking. For each scan, it compares candidate centres with annotated **car** centres in bird’s-eye view. A reference car must have at least five LiDAR returns and its centre must lie inside the same radial region. The evaluator pairs centres one-to-one at distances strictly below **2 m**. Annotations enter after detection and tracking. This is a centre-distance diagnostic, separate from official nuScenes average precision.

**Worked example.** Suppose there are five eligible reference cars and four detections. If three detections match, there are three true positives, one extra detection, and two missed cars. **Precision** is 3/4 = 75% of detections matched. **Recall** is 3/5 = 60% of reference cars found.

<details>
<summary>How to count a match, an extra detection, and a miss</summary>

![One-to-one evaluation and worked example of outcome counts](docs/screenshots/evaluation.png)

</details>

| Scene | Frames | Candidates | Reference cars | Matched | Extra | Missed | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scene-0061 | 39 | 252 | 133 | 66 | 186 | 67 | 26.2% | 49.6% |
| scene-0103 | 40 | 372 | 487 | 234 | 138 | 253 | 62.9% | 48.0% |

These are **per-frame observations**. A car visible in ten frames can count ten times. Precision was higher in scene-0103, where 234 of 372 candidates matched cars. In both scenes the pipeline found roughly half the eligible car observations.

<details>
<summary>Original two-scene results table</summary>

![Precision and recall for both scenes](docs/screenshots/scores.png)

</details>

### Two frames you can check yourself

**scene-0061, frame 008.** Both reference cars were found, yet only two of the six candidates matched. That gives 100% recall and 33.3% precision for this frame. Look for solid boxes without a corresponding dashed green car outline.

![Frame 008 with six candidates, two matched reference cars and four extra detections](output/frames/frame_008.png)

**scene-0103, frame 033.** Fourteen of the 16 candidates matched, but the scan contained 23 eligible reference cars. The remaining two candidates were extra detections, and nine cars were missed. Here the dashed outlines help locate cars that the detector missed even though most of its candidates were correct.

![Frame 033 with 14 matches, two extra detections and nine missed reference cars](output/scene-0103/frames/frame_033.png)

## When the method works and where it struggles

The geometry rules favour nearby, separated cars with enough visible surface on approximately planar roads. Two adjacent cars can merge under DBSCAN. A car beside vegetation can merge with the vegetation and exceed the size limits. Distance or occlusion can leave too few points or make the visible box too small. A barrier with car-like dimensions can pass all the filters, and a real truck can be too long. Because the evaluator counts cars only, a detection on another vehicle category can count as an extra detection.

A single road plane can fit one road level poorly when another is also visible. Changes in cluster shape shift the estimated centre, and rapid motion or missing scans can break an ID association. The two scene runs illustrate these operating conditions and do not establish performance on other routes.

<details>
<summary>Operating conditions and failure cases</summary>

![Operating conditions and corresponding failure modes](docs/screenshots/limits.png)

</details>

## Output files

Each scene folder contains the same set of outputs. In the supplied project, `output/` holds scene-0061 and `output/scene-0103/` holds scene-0103.

| File | What to look for |
|---|---|
| `index.html`, `detections.gif` | Local viewer and animated sequence at two frames per second |
| `frames/frame_*.png` | Individual scans with detected boxes, IDs, and reference cars |
| `stages_*.png` | Cropping, road removal, and clustering for frames 000, 010, and 020 |
| `detections.csv` | One candidate per row, with frame, timestamp, ID, position, range, dimensions, point count, and quality |
| `evaluation.csv` | One row per frame with candidates, reference cars, matches, extras, and misses |
| `summary.json` | Scene description, settings, totals, precision, and recall |
