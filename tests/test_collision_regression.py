"""The original finite pose must produce contacts near the actual wheel."""
import unittest

from scripts.check_collision_regression import check_collision


class CollisionRegressionTests(unittest.TestCase):
  def test_original_and_translated_cylinder_box_contact(self):
    result=check_collision('cpu')
    self.assertGreaterEqual(result['contacts'],2)
    self.assertLess(result['max_distance_to_wheel_m'],.1)


if __name__=='__main__':
  unittest.main()
