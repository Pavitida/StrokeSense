"""Integration tests for the StrokeSense experimental backend."""

import unittest
from pathlib import Path

from src.fusion.run_pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]

IMAGE = ROOT / "data/face_landmark/test/test_face.jpg"
AUDIO = ROOT / "outputs/speech/test/test_speech.wav"
VIDEO = ROOT / (
    "data/movement/An upper limb stroke rehabilitation exercise video/"
    "Exercise/1_Lifting an Object/Complete/Test/07_01_01_09.mp4"
)

NO_FACE = ROOT / "tests/fixtures/no_face.jpg"
SILENT_AUDIO = ROOT / "tests/fixtures/silent_audio.wav"
NO_PERSON = ROOT / "tests/fixtures/no_person.mp4"


class TestStrokeSensePipeline(unittest.TestCase):

    def test_normal_pipeline(self):
        result = run_pipeline(str(IMAGE), str(AUDIO), str(VIDEO))

        self.assertEqual(result["status"], "success")
        for name in ("face", "speech", "movement"):
            self.assertEqual(result[name]["status"], "success")
            self.assertIsNotNone(result[name]["score"])

        self.assertIsNotNone(result["fusion"]["score"])

    def test_no_face(self):
        result = run_pipeline(str(NO_FACE), str(AUDIO), str(VIDEO))

        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["failed_modality"], "face")
        self.assertIsNone(result["fusion"])

    def test_silent_audio(self):
        result = run_pipeline(str(IMAGE), str(SILENT_AUDIO), str(VIDEO))

        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["failed_modality"], "speech")
        self.assertIsNone(result["fusion"])

    def test_no_person(self):
        result = run_pipeline(str(IMAGE), str(AUDIO), str(NO_PERSON))

        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["failed_modality"], "movement")
        self.assertIsNone(result["fusion"])


if __name__ == "__main__":
    unittest.main()
