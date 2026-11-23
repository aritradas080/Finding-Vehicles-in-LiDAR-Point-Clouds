# Finding Vehicles in LiDAR Point Clouds

A LiDAR scan is a collection of measured points, $P = \{(x_i, y_i, z_i)\}_{i=1}^{N}$. Each point tells us where the sensor measured a surface. That surface might belong to the road, a car, a pedestrian, a tree, or a building.

**Which groups of points are likely to belong to vehicles?**

![The question we want to answer from one LiDAR scan](docs/screenshots/question.png)

## Workflow

```text
LiDAR file
    ↓
Convert sensor coordinates to vehicle coordinates
    ↓
Crop the useful region
    ↓
Voxel downsampling
    ↓
RANSAC road-plane estimation
    ↓
Remove ground points
    ↓
DBSCAN object clustering
    ↓
Estimate cluster dimensions with PCA
    ↓
Apply vehicle geometry rules
    ↓
Associate detections across frames
    ↓
Visualize and evaluate
```

## Step 1. Read the LiDAR file

Imagine a LiDAR mounted on the roof of a car. It sends laser pulses in different directions and measures their reflections. From each measurement, it calculates the location of a surface point. Together, these points form a **point cloud**.

| Point | x (m) | y (m) | z (m) |
|---|---:|---:|---:|
| 1 | 5.0 | 1.0 | −1.8 |
| 2 | 10.0 | 2.0 | −0.5 |
| 3 | 10.2 | 2.1 | 0.2 |

The file gives us the coordinates of point 2. Finding out whether it belongs to a car is our task.

![Example of three measured points in a LiDAR file](docs/screenshots/xyz.png)

## Step 2. Convert sensor coordinates to vehicle coordinates

**What does “10 metres away” mean?** We need a reference point. In sensor coordinates, the LiDAR itself is at (0, 0, 0). Its mounting position can be above or forward of the vehicle’s reference point, and its axes can point in different directions.

In vehicle coordinates, **x** means forward, **y** means left, and **z** means up from the vehicle.

![The LiDAR origin and the three vehicle axes](docs/screenshots/coordinates.png)

Imagine holding a ruler pointing sideways while the car points forward. A measurement along the ruler is sideways relative to the car. **Rotation** accounts for the different axis directions. **Translation** accounts for the different origins. Together they give

$$
p_{\text{vehicle}} = R\,p_{\text{sensor}} + t.
$$

![The sideways-ruler example and the rotation and translation equation](docs/screenshots/rotation.png)

## Step 3. Crop the useful region

Calculate the horizontal distance of each point from the vehicle using $r = \sqrt{x^2+y^2}$. Keep points with **2.5 m < r < 40 m** and **−2.5 m < z < 4 m**. The inner boundary excludes the area immediately around our own vehicle. The outer boundary keeps the focus on nearby objects.

| Example | Horizontal distance | Decision, assuming z passes |
|---|---:|---|
| A at (12, 5) m | √(12² + 5²) = 13 m | Keep |
| B at (45, 0) m | 45 m | Discard, too far away |
| C at (1, 0) m | 1 m | Discard, too close to our own vehicle |

![Crop conditions and the three point examples](docs/screenshots/crop.png)

## Step 4. Reduce densely packed points with voxels

Divide the space into small cubes called **voxels**. Each side of a cube is **0.18 m**, or **18 cm**. Imagine that five nearby points fall inside one cube. We keep one of those points. Empty cubes produce no points.

![Five points in an occupied cube become one retained point](docs/screenshots/voxel.png)

Smaller voxels preserve more detail and retain more points to process. Larger voxels usually make processing faster, but a small or distant object can lose the few points that describe it.

![How voxel size affects detail, point count, and processing](docs/screenshots/voxel-size.png)

## Step 5. Estimate the road plane with RANSAC

After cropping and downsampling, the scan still contains the road, cars, trees, and buildings. Road points could connect separate objects into one large group. Imagine laying a flat sheet of cardboard along a small patch of road. **RANSAC** looks for a plane that fits the road in that way.

![How repeated three-point samples produce candidate road planes](docs/screenshots/road-plane.png)

1. **Select low points.** Points with vehicle-frame height between −1.5 m and 0.35 m are more likely to include the road. Some still belong to other objects.
2. **Choose three different points** from this set and construct a candidate plane through them.

```python
low = points[(points[:, 2] < .35) & (points[:, 2] > -1.5)]
a, b, c = low[rng.choice(len(low), 3, replace=False)]
```

![The low-point selection and random sampling used in the code](docs/screenshots/sample-plane.png)

3. **Calculate the plane’s normal.** Imagine an arrow sticking straight out of the surface. For a horizontal road, the arrow points upward. For an upright wall, it points sideways.
4. **Check the direction.** Keep a roughly horizontal plane as a possible road model. Reject a steep plane as a road model.
5. **Count supporting points.** For each acceptable plane, count how many low points lie close to it.
6. **Repeat the sampling 180 times** and select the acceptable plane with the most support.

Rejecting a candidate plane only rejects that road estimate. We decide which points to remove in the next step.

![Horizontal and upright plane normals and the order of the RANSAC checks](docs/screenshots/plane-normal.png)

## Step 6. Remove ground points

Measure each point’s height above the chosen road plane. A point **0.03 m** above the road is removed. A car body point **0.8 m** above the road and a roof point **1.5 m** above it are kept. The lower cutoff is **0.22 m**.

![Road, car body, and roof returns compared with the 0.22 m cutoff](docs/screenshots/cutoff.png)

**Example.** Suppose the road is level at z = −0.50 m. For a point at z = 0.30 m, the height above the road is 0.30 − (−0.50) = **0.80 m**.

| Point’s z coordinate | Height above that road | Action |
|---:|---:|---|
| −0.48 m | 0.02 m | Remove |
| −0.40 m | 0.10 m | Remove |
| 0.30 m | 0.80 m | Keep |
| 1.00 m | 1.50 m | Keep |

![Four examples showing the difference between z and height above the road](docs/screenshots/road-height.png)

## Step 7. Group nearby obstacle points with DBSCAN

Imagine looking down on **Group A**, a single isolated point, and **Group B**. We want each dense group to form its own cluster and the isolated point to be marked as noise.

```python
DBSCAN(eps=0.85, min_samples=5)
```

1. Around each point, look within **0.85 m**.
2. If the neighbourhood contains **at least five points**, including the point itself, it is dense enough to help form a cluster.
3. Connect neighbouring dense regions into the same cluster.
4. Mark points that cannot join a cluster as noise.

![Two groups of points and an isolated point viewed from above](docs/screenshots/clusters.png)

## Step 8. Measure each cluster with PCA

A cluster might represent a car, tree, or wall. First, look at it from above and draw a rectangle around its points. If the car is parked at an angle, the rectangle should rotate with it.

![Top views of a straight car and a rotated car](docs/screenshots/footprint.png)

Principal component analysis (**PCA**) estimates the main direction in which the points spread. We use that direction to orient the rectangle and measure its **length** and **width**. For **height**, subtract the lowest point’s z from the highest point’s z.

## Step 9. Apply the vehicle geometry rules

| Measurement | Required value |
|---|---|
| Number of points | At least 10 |
| Length | Greater than 1.5 m and less than 6.5 m |
| Width | Greater than 0.65 m and less than 3.2 m |
| Height | Greater than 0.45 m and less than 2.8 m |
| Lowest point above the road | Less than 1 m |

![PCA measurements and the rules used to accept a candidate](docs/screenshots/dimensions.png)

| Cluster | What we measured | Decision |
|---|---|---|
| A | Length 4 m, width 1.8 m, height 1.4 m, enough points, close to the road | Accept as a possible vehicle |
| B | Length 12 m | Reject because it is too long |
| C | Width 0.2 m | Reject because it is too narrow |

![The three example clusters and their decisions](docs/screenshots/candidates.png)

## Step 10. Associate detections across frames

Suppose frame 1 contains **Car A with ID 1** and **Car B with ID 2**. In frame 2, we want Car A to keep ID 1 and Car B to keep ID 2. Each new scan gives us fresh clusters, so we need to work out which cluster corresponds to each earlier car.

1. **Account for our own vehicle’s movement.** As we drive towards a parked car, its measured distance decreases even though it stays in place. Use the recording vehicle’s position and orientation to express the detected centres in a fixed world coordinate system.
2. **Predict the next centre.** Predicted position = previous position + velocity × elapsed time. A new track starts with zero velocity.

![The two-car ID example, a parked car, and the position prediction](docs/screenshots/tracking.png)

3. **Match predictions to detections.** Suppose ID 1 is predicted at **(10, 5)**. Detection A at **(10.5, 5.2)** is nearby. Detection B at **(25, 12)** is far away. A is a plausible match for ID 1. The Hungarian algorithm considers all tracks and detections together and assigns at most one detection to each track.
4. **Keep, create, or expire the ID.** Keep an assigned match if it is less than **4 m** away. Give an unmatched detection a new ID. Expire a track after more than **1.5 seconds** without a detection.

A car that disappears behind another object can receive a new ID when it reappears. A split or merged cluster can also cause an ID change.

![The two candidate matches for ID 1 and the track expiry rules](docs/screenshots/association.png)

## Step 11. Evaluate the detections

1. **Compare detected centres with reference centres.** Imagine a reference car centred at **(10, 5)** and our detection centred at **(10.5, 5.2)**. Their centres are less than **2 m** apart, so they can match. Each detection can match one reference car, and each reference car can match one detection.
2. **Count the three outcomes.**

| Situation | Name |
|---|---|
| Our detection matches a reference car | True positive, a correct detection |
| Our detection matches no reference car | False positive, an extra detection |
| A reference car has no matching detection | False negative, a missed car |

**Example.** There are **5 reference cars**. We produce **4 detections**, and **3 match**. That gives **3 correct detections**, **1 extra detection** (4 − 3), and **2 missed cars** (5 − 3). Precision is **3/4 = 75%**. Recall is **3/5 = 60%**.

![Comparing the centres and counting correct, extra, and missed detections](docs/screenshots/evaluation.png)

### Results for the two sequences

| Sequence | Matched observations | Extra detections | Missed observations | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| scene-0061 | 66 | 186 | 67 | 26.2% | 49.6% |
| scene-0103 | 234 | 138 | 253 | 62.9% | 48.0% |

![Results for scene-0061 and scene-0103](docs/screenshots/scores.png)

## Operating conditions and limitations

The method is most suitable for nearby cars with enough measured points, separation from other objects, approximately planar roads, limited occlusion, and reasonably stable sensor calibration and ego poses.

| Condition | Why it causes problems |
|---|---|
| Distant cars | Too few points to form accepted clusters |
| Occlusion | Observed dimensions become smaller than the vehicle dimensions |
| Adjacent cars | DBSCAN can combine them |
| Cars beside barriers or vegetation | Object and clutter points can connect |
| Slopes and multiple road levels | One plane cannot describe all the ground |
| Trucks and buses | Car-oriented size limits may reject them |
| Vehicle-like clutter | Its geometry can satisfy the same rules as a car |
| Cluster shape changes | The estimated centre can jump and disrupt tracking |
| Rapid motion or missed scans | Association can fail or change IDs |

![Suitable operating conditions and the causes of detection or tracking failures](docs/screenshots/limits.png)
