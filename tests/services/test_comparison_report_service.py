from tflens_explorer.services.comparison_report_service import save_cosine_similarity_data, angular_change_per_head
from tflens_explorer.core.snapshot_types import SNAPSHOT_DATA_PATH
from contextlib import redirect_stdout
from io import StringIO
import unittest

class TestComparisonReportService(unittest.TestCase):
    def test_save_cosine_similarity_data(self):
        # Arrange
        name1 = "test_name1"
        name2 = "test_name2"
        hook1 = "hook1"
        hook2 = "hook2"
        head = 0
        sim = 0.9

        # Act
        save_cosine_similarity_data(name1, name2, hook1, hook2, head, sim)

        # Assert
        with open(SNAPSHOT_DATA_PATH / f"{name1}_vs_{name2}", 'r') as f:
            lines = f.readlines()
        self.assertIn(f"{hook1},{hook2}, {head},{sim}\n", lines)

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