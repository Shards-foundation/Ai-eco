from pathlib import Path
import unittest


WORKFLOW = Path(__file__).parents[2] / ".github/workflows/ml-ci-cd-ct.yml"


class MlWorkflowDependencyTests(unittest.TestCase):
    def test_ml_jobs_use_compatible_scoped_dependencies_without_project_requirements(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        data_validation = workflow.split("  train_reproducible_model:", 1)[0]
        training = workflow.split("  train_reproducible_model:", 1)[1].split(
            "  drift_detection:", 1
        )[0]

        self.assertIn("pip install 'numpy<2' 'pandas<3'", data_validation)
        self.assertNotIn("pip install -r requirements.txt", data_validation)
        self.assertIn(
            "pip install mlflow kfp scikit-learn 'numpy<2' 'pandas<3'", training
        )
        self.assertNotIn("pip install -r requirements.txt", training)

    def test_jobs_without_checkout_do_not_enable_pip_cache(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        drift_detection = workflow.split("  drift_detection:", 1)[1].split(
            "  promote_model:", 1
        )[0]
        promote_model = workflow.split("  promote_model:", 1)[1]

        self.assertNotIn("cache: pip", drift_detection)
        self.assertNotIn("cache: pip", promote_model)


if __name__ == "__main__":
    unittest.main()
