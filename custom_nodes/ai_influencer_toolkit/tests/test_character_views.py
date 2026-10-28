"""Run with venv/bin/python custom_nodes/ai_influencer_toolkit/tests/test_character_views.py."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

PACKAGE = "_toolkit_views_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "custom_nodes/ai_influencer_toolkit")]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(PACKAGE + "." + name, Path(package.__path__[0]) / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


character = load("nodes_character")
variation = load("nodes_variation")


class CharacterViewsTest(unittest.TestCase):
    def test_manual_and_view_lists(self):
        node = character.AIInfluencerCharacterViews
        manual = node.execute("manual", 42, "sitting", "85mm", manual_candidates=4).result
        self.assertEqual(manual, (["sitting"], ["85mm"], [42], [""], [""], [4], [variation.DEFAULT_ANGLES]))
        for section, count in [("head", 8), ("head360", 8), ("upper360", 8), ("full360", 8), ("all 32", 32)]:
            with self.subTest(section=section):
                outputs = node.execute(section, 2147483647, manual_candidates=4).result
                self.assertTrue(all(len(output) == count for output in outputs))
                self.assertEqual(len(set(outputs[2])), count)
                self.assertEqual(outputs[5], [1] * count)
                self.assertTrue(all(len(variation.parse_angles(text)) == 1 for text in outputs[6]))
                self.assertFalse(any("<image" in text for output in outputs[:2] for text in output))
                self.assertEqual(outputs, node.execute(section, 2147483647, manual_candidates=4).result)
        title = character.CHARACTER_VIEWS[0]["title"]
        self.assertEqual(node.execute(title, 42).result[4], [title])

    def test_placement_uses_the_presets_body_angle(self):
        args = dict(plate=torch.zeros((1, 128, 128, 3)), prompt="profile facing left", seed=42,
                    candidates=1, vary="placement + angle", pose="new angle per seed",
                    x_min=0.15, x_max=0.85, size_min=0.45, size_max=0.7, horizon=0.35)
        views = character.AIInfluencerCharacterViews.execute("full360", 42).result
        angles = views[6][2]
        self.assertEqual(variation.parse_angles(angles), [(-90.0, "")])
        output = variation.AIInfluencerPlacementVariations.execute(**args, angles=angles).result
        self.assertEqual(output[1], [args["prompt"]])
        self.assertEqual(len(output[0]), 1)
        custom = "25: custom angle"
        self.assertEqual(character.AIInfluencerCharacterViews.execute("manual", 42, manual_angles=custom).result[6], [custom])

    def test_existing_graph_wiring(self):
        for filename in ["08_krea2_local_master", "09_krea2_pose_transfer", "09b_krea2_fixed_location",
                         "09d_krea2_location_reference", "10_preview_then_4k"]:
            with self.subTest(workflow=filename):
                data = json.loads((ROOT / "user/default/workflows/ai_influencer" / (filename + ".json")).read_text())
                nodes = {n["id"]: n for n in data["nodes"]}
                links = {l[0]: l for l in data["links"]}
                self.assertEqual(len(nodes), len(data["nodes"]))
                self.assertEqual(len(links), len(data["links"]))
                for lid, src, src_slot, dst, dst_slot, type_ in links.values():
                    self.assertEqual(nodes[dst]["inputs"][dst_slot]["link"], lid)
                    self.assertIn(lid, nodes[src]["outputs"][src_slot]["links"])
                views = next(n for n in nodes.values() if n["type"] == "AIInfluencerCharacterViews")
                builder = next(n for n in nodes.values() if n["type"] == "AIInfluencerPromptBuilder")
                for name, slot in [("subject_action", 0), ("camera_and_lens", 1)]:
                    inp = next(i for i in builder["inputs"] if i["name"] == name)
                    self.assertEqual(links[inp["link"]][1:3], [views["id"], slot])
                self.assertEqual(views["widgets_values"][0], "manual")
                self.assertFalse(any(n["type"] in ["TextEncodeQwenImage21", "QwenImage21Cache"] for n in nodes.values()))
                mapper = next((n for n in nodes.values() if n["type"] in
                               ["AIInfluencerPoseAngle", "AIInfluencerPlacementVariations"]), None)
                if mapper:
                    inp = next(i for i in mapper["inputs"] if i["name"] == "candidates")
                    self.assertEqual(links[inp["link"]][1:3], [views["id"], 5])
                    if mapper["type"] == "AIInfluencerPlacementVariations":
                        inp = next(i for i in mapper["inputs"] if i["name"] == "angles")
                        self.assertEqual(links[inp["link"]][1:3], [views["id"], 6])
                if filename.startswith("10_"):
                    save = next(n for n in nodes.values() if n["type"] == "AIInfluencerSaveImage" and n["mode"] == 2)
                    self.assertIsNone(next(i for i in save["inputs"] if i["name"] == "prompt_text")["link"])


if __name__ == "__main__":
    unittest.main()
