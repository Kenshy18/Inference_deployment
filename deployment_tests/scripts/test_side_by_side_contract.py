from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class SideBySideDeploymentContractTests(unittest.TestCase):
    def test_manifest_has_versioned_installation_identity(self) -> None:
        text = (ROOT / "deployment/windows/Build-Deployer.ps1").read_text()
        self.assertIn("schema_version = 3", text)
        self.assertIn('default_distribution = $defaultDistribution', text)
        self.assertIn('gui_filename = $guiFileName', text)
        self.assertIn('deployment_profile = "deployment-profile.json"', text)
        self.assertIn('$deployerFileName = "MaskPipelineDeployer-$releaseToken.exe"', text)

    def test_deployer_does_not_touch_legacy_gui_state(self) -> None:
        text = (ROOT / "deployment/windows/Deploy-MaskPipeline.ps1").read_text()
        self.assertNotIn('"mask-pipeline-studio-windows"', text)
        self.assertNotIn('Get-Process "Mask Pipeline Studio"', text)
        self.assertNotIn('"Mask Pipeline Studio.lnk"', text)
        self.assertIn('$userDataDirectory = Join-Path $InstallRoot "user-data"', text)
        self.assertIn('$guiTarget = Join-Path $guiDirectory $guiFileName', text)
        self.assertIn('New-Shortcut $guiTarget $desktopShortcut $profileArgument', text)

    def test_existing_distribution_and_install_root_are_never_overwritten(self) -> None:
        text = (ROOT / "deployment/windows/Deploy-MaskPipeline.ps1").read_text()
        self.assertIn("Existing distributions are never overwritten", text)
        self.assertIn("Release install directory already exists", text)
        self.assertNotIn("--unregister", text.split("try {", 1)[0])

    def test_failed_install_preserves_diagnostics_outside_rollback_root(self) -> None:
        text = (ROOT / "deployment/windows/Deploy-MaskPipeline.ps1").read_text()
        archive = text.index('"MaskPipeline\\deployment-logs"')
        removal = text.index("Remove-Item -LiteralPath $InstallRoot -Recurse -Force")
        self.assertLess(archive, removal)
        self.assertIn("Failure log preserved", text)

    def test_release_id_contains_source_commit(self) -> None:
        text = (ROOT / "deployment/windows/Build-Release.ps1").read_text()
        self.assertIn('$commit.Substring(0, 8)', text)
        self.assertIn("side-by-side installation contract", text)


if __name__ == "__main__":
    unittest.main()
