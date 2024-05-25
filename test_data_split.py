# test_data_splitting.py
import unittest
import numpy as np
from Helpers import split_data  # Adjust this import based on the actual file name and location


class TestDataSplitting(unittest.TestCase):
    def test_split_data(self):
        # Create a small dataset
        X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])  # 6 instances
        Y = np.array([0, 1, 0, 1, 0, 1])  # Labels

        # Set the parameters for the split_data function
        P = 34  # 50% of instances will be labeled
        W = 50  # 50% of labeled instances will be mislabeled

        # Run the split_data function
        X_l, X_u, Y_l = split_data(X, Y, P, W)

        # Assertions to check the correctness of the output
        self.assertEqual(len(X_l), 2, "Incorrect number of labeled instances")
        self.assertEqual(len(X_u), 4, "Incorrect number of unlabeled instances")
        self.assertEqual(len(Y_l), 2, "Incorrect number of labels for labeled instances")
        self.assertIn(np.sum(Y_l != Y[:3]), [1], "Incorrect number of mislabeled instances")

        # Optional: Check that modifications to Y_l do not affect Y
        self.assertFalse(np.array_equal(Y_l, Y[:3]), "Modifications to Y_l affect Y")


if __name__ == '__main__':
    unittest.main()
