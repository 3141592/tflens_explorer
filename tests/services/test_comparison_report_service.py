from tflens_explorer.services.comparison_report_service import angular_change_per_head
from tflens_explorer.core.snapshot_types import SNAPSHOT_DATA_PATH
from contextlib import redirect_stdout
from io import StringIO
import unittest

class TestComparisonReportService(unittest.TestCase):
    def test_angular_change_per_head_with_empty_file(self):
        # Arrange
        filename = "test_data_empty.txt"
        with open(SNAPSHOT_DATA_PATH / filename, "w"):
            pass

        # Act
        output = StringIO()
        with redirect_stdout(output):
            angular_change_per_head(filename)

        # Assert
        report = output.getvalue()
        self.assertIn("Angular similarity by head", report)
        self.assertIn("No data rows found.", report)

    def test_angular_change_per_head_with_nonexistent_file(self):
        # Arrange
        filename = "nonexistent.txt"

        # Act & Assert (Assuming the print statement handles the error)
        with self.assertRaises(SystemExit) as cm:
            angular_change_per_head(filename)
        self.assertEqual(cm.exception.code, 1)

if __name__ == '__main__':
    unittest.main()