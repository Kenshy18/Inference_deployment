from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from tests.helpers import write_sample_sqlite


class DevDataTests(unittest.TestCase):
    def test_write_sample_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_sample_sqlite(Path(tmp) / "sample.sqlite", frames=3)
            conn = sqlite3.connect(str(path))
            try:
                self.assertEqual(
                    conn.execute("SELECT COUNT(*) FROM masks").fetchone()[0], 3
                )
                self.assertEqual(
                    conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0], 1
                )
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
