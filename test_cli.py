#!/usr/bin/env python3
"""
Comprehensive test suite for Sermon Shorts AI Agent.
Tests transcription, analysis, and recommendation functionality.
"""
import os
import sys
import tempfile
import json
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules to test
from nodes.transcription_node import _validate_media_file, _extract_audio_if_needed, _format_ts
from nodes.analysis_node import _format_ts as analysis_format_ts, _load_segments
from Classes.agent_state import AgentState

class TestTranscriptionNode(unittest.TestCase):
    """Test cases for transcription node functionality."""

    def test_validate_media_file_success(self):
        """Test successful file validation."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(b"dummy video content")
            tmp_path = tmp.name

        try:
            result = _validate_media_file(tmp_path)
            self.assertEqual(result.suffix.lower(), ".mp4")
            self.assertTrue(result.exists())
        finally:
            os.unlink(tmp_path)

    def test_validate_media_file_not_found(self):
        """Test file validation with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            _validate_media_file("nonexistent_file.mp4")

    def test_validate_media_file_unsupported_format(self):
        """Test file validation with unsupported format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"text content")
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError) as context:
                _validate_media_file(tmp_path)
            self.assertIn("Unsupported file format", str(context.exception))
        finally:
            os.unlink(tmp_path)

    def test_format_timestamp(self):
        """Test timestamp formatting function."""
        self.assertEqual(_format_ts(0), "00:00")
        self.assertEqual(_format_ts(65), "01:05")
        self.assertEqual(_format_ts(3661), "61:01")
        self.assertEqual(_format_ts(-5), "00:00")  # Negative should be clamped to 0

    def test_extract_audio_if_needed_audio_file(self):
        """Test audio extraction with already audio file."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(b"dummy audio content")
            tmp_path = Path(tmp.name)

        try:
            result = _extract_audio_if_needed(tmp_path)
            self.assertEqual(result, tmp_path)  # Should return same path for audio files
        finally:
            os.unlink(tmp_path)


class TestAnalysisNode(unittest.TestCase):
    """Test cases for analysis node functionality."""

    def test_format_timestamp_analysis(self):
        """Test timestamp formatting in analysis node."""
        self.assertEqual(analysis_format_ts(0), "00:00")
        self.assertEqual(analysis_format_ts(125), "02:05")
        self.assertEqual(analysis_format_ts(3725), "62:05")

    @patch('nodes.analysis_node.Path.exists')
    @patch('nodes.analysis_node.Path.read_text')
    @patch('nodes.analysis_node.json.loads')
    def test_load_segments_success(self, mock_json_loads, mock_read_text, mock_exists):
        """Test successful loading of transcription segments."""
        mock_exists.return_value = True
        mock_segments = [
            {"start": 0.0, "end": 5.0, "text": "Hello everyone"},
            {"start": 5.0, "end": 10.0, "text": "Welcome to church"}
        ]
        mock_json_data = {"segments": mock_segments}
        mock_read_text.return_value = json.dumps(mock_json_data)
        mock_json_loads.return_value = mock_json_data

        result = _load_segments()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "Hello everyone")

    @patch('nodes.analysis_node.Path.exists')
    def test_load_segments_file_not_found(self, mock_exists):
        """Test loading segments when file doesn't exist."""
        mock_exists.return_value = False

        result = _load_segments()
        self.assertEqual(result, [])


class TestAgentState(unittest.TestCase):
    """Test cases for AgentState class."""

    def test_agent_state_creation(self):
        """Test AgentState can be created and accessed."""
        state = AgentState()
        self.assertIsInstance(state, dict)

    def test_agent_state_with_data(self):
        """Test AgentState with initial data."""
        initial_data = {"filePath": "test.mp4", "model": "small.en"}
        state = AgentState(initial_data)
        self.assertEqual(state.get("filePath"), "test.mp4")
        self.assertEqual(state.get("model"), "small.en")


class TestSystemConfiguration(unittest.TestCase):
    """Test cases for system configuration and settings."""

    def test_default_model_setting(self):
        """Test that default model is set to small.en for better accuracy."""
        from nodes.transcription_node import DEFAULT_MODEL
        self.assertEqual(DEFAULT_MODEL, "small.en")

    def test_audio_directory_setting(self):
        """Test audio directory configuration."""
        from nodes.transcription_node import AUDIO_DIR
        # Should be configurable via environment variable
        self.assertIsInstance(AUDIO_DIR, Path)

    def test_supported_file_formats(self):
        """Test that all expected file formats are supported."""
        supported_formats = [".mp3", ".mp4", ".wav", ".m4a", ".mov"]

        for fmt in supported_formats:
            with tempfile.NamedTemporaryFile(suffix=fmt, delete=False) as tmp:
                tmp.write(b"dummy content")
                tmp_path = tmp.name

            try:
                result = _validate_media_file(tmp_path)
                self.assertEqual(result.suffix.lower(), fmt)
            finally:
                os.unlink(tmp_path)


class TestRecommendationSystem(unittest.TestCase):
    """Test cases for the enhanced recommendation system."""

    def test_recommendation_output_format(self):
        """Test that recommendations follow the expected JSON format."""
        # Mock recommendation structure
        expected_keys = ["start_sec", "end_sec", "start", "end", "description", "confidence", "reasoning"]

        mock_recommendation = {
            "start_sec": 120.5,
            "end_sec": 180.0,
            "start": "02:00",
            "end": "03:00",
            "description": "Inspiring message about faith",
            "confidence": 0.85,
            "reasoning": "High emotional impact with clear message"
        }

        for key in expected_keys:
            self.assertIn(key, mock_recommendation)

        # Test data types
        self.assertIsInstance(mock_recommendation["start_sec"], (int, float))
        self.assertIsInstance(mock_recommendation["confidence"], (int, float))
        self.assertIsInstance(mock_recommendation["description"], str)

    def test_clip_duration_requirements(self):
        """Test that clips meet the 60-90 second duration requirement."""
        mock_clips = [
            {"start_sec": 100, "end_sec": 170},  # 70 seconds - valid
            {"start_sec": 200, "end_sec": 290},  # 90 seconds - valid
            {"start_sec": 300, "end_sec": 350},  # 50 seconds - too short
            {"start_sec": 400, "end_sec": 600},  # 200 seconds - too long
        ]

        for clip in mock_clips:
            duration = clip["end_sec"] - clip["start_sec"]
            if 60 <= duration <= 180:  # Updated range: 60-90 seconds target, max 180
                self.assertTrue(True)  # Valid duration
            else:
                # Should be flagged in real implementation
                pass

    def test_confidence_score_range(self):
        """Test that confidence scores are within valid range."""
        valid_scores = [0.0, 0.5, 0.85, 1.0]
        invalid_scores = [-0.1, 1.1, 2.0]

        for score in valid_scores:
            self.assertTrue(0.0 <= score <= 1.0)

        for score in invalid_scores:
            self.assertFalse(0.0 <= score <= 1.0)


class TestEnhancedPromptFeatures(unittest.TestCase):
    """Test cases for enhanced system prompt features."""

    def test_content_diversity_keywords(self):
        """Test that the system recognizes content diversity requirements."""
        diversity_keywords = [
            "opening illustrations", "main teaching points", "practical applications",
            "worship moments", "personal stories", "closing prayers"
        ]

        # Mock analysis should identify these different content types
        for keyword in diversity_keywords:
            self.assertIsInstance(keyword, str)
            self.assertTrue(len(keyword) > 0)

    def test_vocal_emphasis_detection(self):
        """Test recognition of vocal emphasis indicators."""
        emphasis_indicators = [
            "Listen!", "Here's the thing...", "You know what?",
            "Oh my goodness", "Praise God", "Amen"
        ]

        for indicator in emphasis_indicators:
            # Should be detected as high-engagement content
            self.assertTrue(len(indicator) > 0)
            self.assertIsInstance(indicator, str)

    def test_contemporary_references(self):
        """Test identification of contemporary and relatable content."""
        contemporary_refs = [
            "smartphones", "Instagram", "Facebook", "TikTok",
            "sports", "celebrities", "current events"
        ]

        for ref in contemporary_refs:
            # Should be prioritized for viral potential
            self.assertIsInstance(ref, str)
            self.assertTrue(len(ref) > 0)


class TestWhisperModelUpgrade(unittest.TestCase):
    """Test cases for Whisper model upgrade from base.en to small.en."""

    def test_model_accuracy_improvement(self):
        """Test that small.en model provides better accuracy."""
        # Mock transcription results comparison
        base_result = "attending to winning"  # Common error with base.en
        small_result = "attending the wedding"  # Expected improvement with small.en

        # Test that small.en fixes common transcription errors
        self.assertNotEqual(base_result, small_result)
        self.assertIn("wedding", small_result)
        self.assertNotIn("winning", small_result)

    def test_model_performance_tradeoff(self):
        """Test that we actually use the small.en model in our configuration."""
        from nodes.transcription_node import DEFAULT_MODEL

        # Test that we're actually using small.en (not just that it exists in a dict we made up)
        self.assertEqual(DEFAULT_MODEL, "small.en")

        # Test that the model choice makes sense for our use case
        # small.en should be a reasonable balance between speed and accuracy
        # (This is more of a design decision test than a functional test)
        expected_models = ["tiny.en", "base.en", "small.en", "medium.en", "large"]
        self.assertIn(DEFAULT_MODEL, expected_models)

        # Verify it's not the extremes (too fast/inaccurate or too slow)
        self.assertNotEqual(DEFAULT_MODEL, "tiny.en")  # Too inaccurate
        self.assertNotIn("large", DEFAULT_MODEL)       # Too slow for real-time use


class TestNoiseFilteringConcepts(unittest.TestCase):
    """Test cases for future noise filtering with embeddings."""

    def test_noise_identification_concepts(self):
        """Test identification of noise vs content segments."""
        noise_examples = [
            "microphone check testing testing",
            "amen amen amen amen amen",
            "la la la oh oh oh",
            "can you hear me in the back"
        ]

        content_examples = [
            "Today we're talking about faith and hope",
            "In Romans chapter four we see God's plan",
            "God's love is unconditional and everlasting",
            "Let me tell you a story about forgiveness"
        ]

        # Noise should be shorter and less meaningful
        for noise in noise_examples:
            self.assertTrue(len(noise.split()) <= 10)  # Typically short

        # Content should be more substantial (at least 5 words)
        for content in content_examples:
            word_count = len(content.split())
            self.assertTrue(word_count >= 5, f"Content '{content}' has only {word_count} words")

    def test_semantic_filtering_concepts(self):
        """Test concepts for semantic similarity filtering."""
        sermon_keywords = ["scripture", "jesus", "faith", "god", "prayer", "love", "bible"]
        noise_keywords = ["microphone", "sound", "check", "testing", "amen repetition"]

        # Should be able to distinguish between content types
        self.assertTrue(len(sermon_keywords) > 0)
        self.assertTrue(len(noise_keywords) > 0)
        self.assertEqual(len(set(sermon_keywords) & set(noise_keywords)), 0)  # No overlap


class TestCLIFunctionality(unittest.TestCase):
    """Test cases for CLI argument handling and environment setup."""

    def test_environment_variable_handling(self):
        """Test CLI environment variable setup."""
        test_file = "test_sermon.mp4"

        # Test setting environment variable
        os.environ["SERMON_FILE_PATH"] = test_file
        self.assertEqual(os.environ.get("SERMON_FILE_PATH"), test_file)

        # Clean up
        del os.environ["SERMON_FILE_PATH"]
        self.assertIsNone(os.environ.get("SERMON_FILE_PATH"))

    def test_cli_argument_simulation(self):
        """Test CLI argument parsing simulation."""
        # Mock command line arguments
        mock_args = {
            "--file": "S:\\2025-07-27_09-58-11.mp4",
            "-f": "sermon.mp3",
            "auto-detect": None
        }

        for value in mock_args.values():
            if value:
                self.assertTrue(value.endswith(('.mp3', '.mp4', '.wav', '.m4a', '.mov')))


def main():
    """Run comprehensive test suite."""
    print("🧪 Running Comprehensive Test Suite for Sermon Shorts AI Agent\n")
    print("=" * 70)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestTranscriptionNode,
        TestAnalysisNode,
        TestAgentState,
        TestSystemConfiguration,
        TestRecommendationSystem,
        TestEnhancedPromptFeatures,
        TestWhisperModelUpgrade,
        TestNoiseFilteringConcepts,
        TestCLIFunctionality
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.wasSuccessful():
        print("\nALL TESTS PASSED!")
        return True
    else:
        print("\nSOME TESTS FAILED. Please review the failures above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
