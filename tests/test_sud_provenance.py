"""Runtime source provenance must not require unpublished local tests."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from roadmc.scripts import diagnose_sud_road


class SUDProvenanceTests(unittest.TestCase):
    def test_source_hashes_work_without_test_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reader = root / "roadmc" / "data" / "real" / "sud_road.py"
            reader.parent.mkdir(parents=True)
            reader.write_bytes(b"# reader source fixture\n")
            with patch.object(diagnose_sud_road, "ROOT", root):
                result = diagnose_sud_road._implementation_hashes()
                self.assertEqual(
                    set(result),
                    {
                        "roadmc/data/real/sud_road.py",
                        "roadmc/scripts/diagnose_sud_road.py",
                    },
                )
                self.assertEqual(
                    result["roadmc/data/real/sud_road.py"],
                    hashlib.sha256(reader.read_bytes()).hexdigest(),
                )
                self.assertEqual(
                    result["roadmc/scripts/diagnose_sud_road.py"],
                    hashlib.sha256(Path(diagnose_sud_road.__file__).read_bytes()).hexdigest(),
                )
                tests = root / "tests"
                tests.mkdir()
                (tests / "test_sud_road.py").write_bytes(b"# unrelated local tests\n")
                self.assertEqual(result, diagnose_sud_road._implementation_hashes())


if __name__ == "__main__":
    unittest.main()
