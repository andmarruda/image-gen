import os
import unittest
from unittest.mock import patch

from src.model_config import model_defaults, model_family, runtime_model_id


class ModelConfigurationTests(unittest.TestCase):
    def test_dev_is_the_default_profile(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(model_family(), "flux2-dev")
            self.assertEqual(model_defaults()["num_inference_steps"], 28)
            self.assertEqual(runtime_model_id(), "diffusers/FLUX.2-dev-bnb-4bit")

    def test_dev_uses_quantized_runtime_model_by_default(self):
        with patch.dict(os.environ, {"MODEL_ID": "black-forest-labs/FLUX.2-dev"}, clear=True):
            self.assertEqual(model_family(), "flux2-dev")
            self.assertEqual(runtime_model_id(), "diffusers/FLUX.2-dev-bnb-4bit")

    def test_flux1_schnell_defaults_remain_supported(self):
        with patch.dict(os.environ, {"MODEL_ID": "black-forest-labs/FLUX.1-schnell"}, clear=True):
            self.assertEqual(model_family(), "flux1")
            self.assertEqual(model_defaults(), {"num_inference_steps": 4, "guidance_scale": 0.0})
if __name__ == "__main__":
    unittest.main()
