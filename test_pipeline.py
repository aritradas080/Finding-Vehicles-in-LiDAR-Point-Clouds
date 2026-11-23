"""Small deterministic checks independent of dataset labels."""
import unittest
import numpy as np
from lidar_vehicles import apply, detect, transform


class PipelineChecks(unittest.TestCase):
    def test_sensor_pose_round_trip(self):
        matrix = transform(dict(rotation=[2**-.5, 0, 0, 2**-.5], translation=[3, -2, 1]))
        points = np.array([[1., 0, 0], [-4, 7, 2]])
        np.testing.assert_allclose(apply(apply(points, matrix), np.linalg.inv(matrix)), points, atol=1e-10)
        np.testing.assert_allclose(apply(points[:1], matrix), [[3, -1, 1]], atol=1e-10)

    def test_flat_ground_and_separated_car_shaped_returns(self):
        x, y = np.meshgrid(np.arange(-20, 20, .4), np.arange(-15, 15, .4))
        ground = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
        rng = np.random.default_rng(3)
        obstacle = rng.uniform([8, -1, .3], [12, 1, 1.7], (700, 3))
        _, foreground, _, detections, _ = detect(np.vstack([ground, obstacle]), np.random.default_rng(42), 40)
        self.assertGreater(len(foreground), 100)
        self.assertTrue(np.all(foreground[:, 2] > .22))
        self.assertEqual(len(detections), 1)
        self.assertLess(np.linalg.norm(detections[0]['center'] - [10, 0]), .4)


if __name__ == '__main__':
    unittest.main()
