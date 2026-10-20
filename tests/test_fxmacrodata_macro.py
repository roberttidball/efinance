"""Offline regression tests for the FXMacroData adapter boundary."""
import importlib.util
import os
from pathlib import Path
import sys
import traceback
import types
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import requests


class FXMacroDataBoundaryTests(unittest.TestCase):
    def setUp(self):
        shared = types.ModuleType("efinance.shared")
        shared.session = Mock()
        package = types.ModuleType("efinance")
        package.__path__ = []
        macro = types.ModuleType("efinance.macro")
        macro.__path__ = []
        path = Path(__file__).resolve().parents[1] / "efinance/macro/getter.py"
        spec = importlib.util.spec_from_file_location("efinance.macro.getter", path)
        self.getter = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"efinance": package, "efinance.macro": macro, "efinance.shared": shared}):
            spec.loader.exec_module(self.getter)
        self.session = shared.session
        self.response = self.session.get.return_value
        self.response.status_code = 200
        self.response.json.return_value = {"data": []}
        env = patch.dict(os.environ, {"FXMD_API_KEY": "", "FXMACRODATA_API_KEY": ""})
        env.start()
        self.addCleanup(env.stop)

    def test_error_object_is_not_turned_into_a_successful_dataframe(self):
        for payload in ({"detail": "fixture-private-value"}, {"error": "fixture-private-value"}):
            self.response.json.return_value = payload
            with self.assertRaisesRegex(ValueError, "error response") as raised:
                self.getter.get_indicator("usd", "synthetic_release")
            self.assertNotIn("fixture-private-value", str(raised.exception))

    def test_saved_error_payload_is_rejected_by_frame_boundary(self):
        with self.assertRaisesRegex(ValueError, "error response"):
            self.getter._frame({"detail": "fixture-private-value"})

    def test_non_object_or_list_json_is_rejected(self):
        for payload in (None, "unexpected text", 1, True):
            self.response.json.return_value = payload
            with self.assertRaisesRegex(ValueError, "invalid response"):
                self.getter._request("/v1/calendar/usd")

    def test_transport_error_does_not_echo_credentials_in_traceback(self):
        private_value = "fixture-private-value"
        self.session.get.side_effect = requests.ConnectionError(private_value)
        try:
            self.getter._request("/v1/calendar/usd", api_key=private_value)
        except requests.RequestException as exc:
            self.assertNotIn("fixture-private-value", "".join(traceback.format_exception(exc)))
        else:
            self.fail("transport error was suppressed")

    def test_http_error_does_not_echo_upstream_details(self):
        for status in (400, 401, 404, 500):
            self.response.status_code = status
            self.response.raise_for_status.side_effect = requests.HTTPError("fixture-private-value")
            try:
                self.getter._request("/v1/calendar/usd")
            except requests.HTTPError as exc:
                text = "".join(traceback.format_exception(exc))
                self.assertNotIn("fixture-private-value", text)
                self.assertIn(str(status), text)
            else:
                self.fail("HTTP error was suppressed")
        self.response.json.assert_not_called()

    def test_non_json_error_does_not_echo_response_body(self):
        self.response.json.side_effect = ValueError("fixture-private-value")
        try:
            self.getter._request("/v1/calendar/usd")
        except ValueError as exc:
            self.assertNotIn("fixture-private-value", "".join(traceback.format_exception(exc)))
        else:
            self.fail("non-JSON payload was accepted")

    def test_redirects_remain_disabled_and_bodies_are_not_parsed(self):
        for status in (301, 302, 303, 307, 308):
            self.response.status_code = status
            with self.assertRaises(requests.HTTPError):
                self.getter._request("/v1/calendar/usd", api_key="fixture-key")
            self.assertFalse(self.session.get.call_args.kwargs["allow_redirects"])
        self.response.json.assert_not_called()

    def test_valid_rows_and_nulls_are_preserved(self):
        self.response.json.return_value = {"currency": "usd", "data": [{"release": "synthetic_release", "val": None}]}
        frame = self.getter.get_indicator("USD", "synthetic_release", official_only=True)
        self.assertIsInstance(frame, pd.DataFrame)
        self.assertEqual(["synthetic_release"], frame["release"].tolist())
        self.assertEqual([None], frame["val"].tolist())
        self.assertEqual("true", self.session.get.call_args.kwargs["params"]["official_only"])

    def test_empty_rows_are_still_empty(self):
        self.assertTrue(self.getter.get_calendar().empty)

    def test_list_payload_is_preserved(self):
        self.response.json.return_value = [{"release": "synthetic_release"}]
        self.assertEqual(["synthetic_release"], self.getter.get_calendar()["release"].tolist())

    def test_successful_catalogue_metadata_is_not_rejected(self):
        self.response.json.return_value = {"currency": "usd", "indicators": {"synthetic_release": {"unit": "index"}}}
        self.assertEqual("usd", self.getter.get_data_catalogue()["currency"].iloc[0])

    def test_malformed_key_is_rejected_before_transport(self):
        with self.assertRaises(ValueError):
            self.getter._request("/v1/calendar/usd", api_key="fixture key")
        self.session.get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
