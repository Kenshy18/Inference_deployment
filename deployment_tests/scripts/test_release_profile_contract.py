from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ReleaseProfileContractTests(unittest.TestCase):
    def test_builder_records_versioned_profile(self) -> None:
        text = (ROOT / "deployment/windows/Build-Deployer.ps1").read_text()
        self.assertIn('[ValidateSet("core", "all")][string]$Profile', text)
        self.assertIn("schema_version = 3", text)
        self.assertIn("profile = $Profile", text)
        self.assertIn("default_distribution = $defaultDistribution", text)

    def test_release_forwards_and_verifies_profile(self) -> None:
        text = (ROOT / "deployment/windows/Build-Release.ps1").read_text()
        self.assertIn("-Profile $Profile", text)
        self.assertIn("$manifest.profile -ne $Profile", text)

    def test_deployer_uses_manifest_profile_for_preflight(self) -> None:
        text = (ROOT / "deployment/windows/Deploy-MaskPipeline.ps1").read_text()
        self.assertIn("$profile = [string]$manifest.profile", text)
        self.assertIn('"--profile", $profile', text)
        self.assertNotIn('"--profile", "all"', text)


if __name__ == "__main__":
    unittest.main()
