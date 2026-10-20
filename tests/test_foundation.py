import io
import logging
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr

from modelaudit.config import ConfigError, load_config
from modelaudit.logging import configure_logging


class ConfigTests(unittest.TestCase):
    def load_text(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(text, encoding="utf-8")
            return load_config(path)

    def test_nested_configuration(self):
        self.assertEqual(self.load_text("project: {phase: 0, enabled: true}"),
                         {"project": {"phase": 0, "enabled": True}})

    def test_rejects_empty_scalar_and_sequence_roots(self):
        for text in ("", "{}", "null", "42", "[a, b]"):
            with self.subTest(text=text), self.assertRaises(ConfigError):
                self.load_text(text)

    def test_rejects_duplicate_and_nonstring_keys(self):
        for text in ("a: 1\na: 2", "nested: {a: 1, a: 2}",
                     "1: value", "nested: {true: value}"):
            with self.subTest(text=text), self.assertRaises(ConfigError):
                self.load_text(text)

    def test_rejects_malformed_and_multiple_documents(self):
        for text in ("a: [", "a: 1\n---\nb: 2"):
            with self.subTest(text=text), self.assertRaises(ConfigError):
                self.load_text(text)

    def test_rejects_python_object_tags(self):
        with self.assertRaises(ConfigError):
            self.load_text("x: !!python/tuple [1, 2]")

    def test_missing_file_stays_a_file_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_config(Path(directory) / "missing.yaml")

    def test_configuration_does_not_modify_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("project: {name: Test}", encoding="utf-8")
            before = path.read_bytes()
            load_config(path)
            self.assertEqual(path.read_bytes(), before)


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("modelaudit")
        self.saved = (self.logger.handlers[:], self.logger.level,
                      self.logger.propagate)
        self.logger.handlers = []

    def tearDown(self):
        for handler in self.logger.handlers:
            if handler not in self.saved[0]:
                handler.close()
        self.logger.handlers, level, propagate = self.saved
        self.logger.setLevel(level)
        self.logger.propagate = propagate

    def test_repeated_setup_emits_once_and_updates_level(self):
        output = io.StringIO()
        with redirect_stderr(output):
            configure_logging(logging.WARNING)
            logger = configure_logging(logging.INFO)
            logger.info("foundation ready")
        self.assertEqual(output.getvalue(), "INFO foundation ready\n")

    def test_root_logger_and_existing_handler_are_preserved(self):
        root = logging.getLogger()
        before = (root.level, root.handlers[:])
        existing = logging.NullHandler()
        self.logger.addHandler(existing)
        configure_logging()
        self.assertIn(existing, self.logger.handlers)
        self.assertEqual((root.level, root.handlers), before)


if __name__ == "__main__":
    unittest.main()
