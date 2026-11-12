from __future__ import annotations

import html
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.alerts.service import render_alert
from src.vision.colors import _pixel_name
from src.alerts.cooldown import CooldownStore
from src.core.models import IdentityStatus, PersonRecord, SurveillanceEvent
from src.alerts.render import EmailRenderer


def person(track, status, name="Not identified", **kwargs):
    return PersonRecord(str(track), status, name=name, **kwargs)


class RendererTests(unittest.TestCase):
    def event(self, people=()):
        return SurveillanceEvent(
            event_id="EV-001", event_type="Person detected",
            date_time_with_timezone="2026-11-09 12:00:00 +05:30",
            location="Main Gate", source_id="CAM-01", people=tuple(people),
            specific_observed_reason="Person entered monitored area",
            first_detected_time="2026-11-09 12:00:00 +05:30",
            observed_duration="10 seconds", status="Active",
        )

    def test_one_known(self):
        output = EmailRenderer().render(self.event([person("1", IdentityStatus.KNOWN, "Ayush")]))
        self.assertIn("Name: Ayush", output.text)
        self.assertIn("Known people: 1", output.text)

    def test_one_unknown(self):
        output = EmailRenderer().render(self.event([person("2", IdentityStatus.UNKNOWN)]))
        self.assertIn("Unknown person 1", output.text)

    def test_multiple_sorted_and_counted(self):
        people = [person("10", IdentityStatus.UNKNOWN), person("2", IdentityStatus.KNOWN, "Ayush")]
        output = EmailRenderer().render(self.event(people))
        self.assertLess(output.text.index("Track ID: 2"), output.text.index("Track ID: 10"))
        self.assertIn("Total people visible: 2", output.text)

    def test_unresolved(self):
        output = EmailRenderer().render(self.event([person("3", IdentityStatus.UNRESOLVED)]))
        self.assertIn("Identity status: Unresolved", output.text)
        self.assertIn("identity unresolved", output.text)

    def test_missing_openpar_attributes(self):
        missing = person("1", IdentityStatus.UNKNOWN)
        output = EmailRenderer().render(self.event([missing]))
        self.assertIn("Upper clothing: upper clothing not determined", output.text)
        self.assertIn("Activity: performing an undetermined activity", output.text)
        self.assertEqual(missing.description(1), "Unknown person 1 is visible.")

    def test_description_omits_non_visible_fields(self):
        record = person(
            "1", IdentityStatus.KNOWN, "Nicole Kidman",
            upper_clothing="casual upper clothing",
        )
        description = record.description(1)
        self.assertIn("casual upper clothing", description)
        self.assertNotIn("lower clothing", description)
        self.assertNotIn("items", description)
        self.assertNotIn("activity", description)

    def test_no_people(self):
        output = EmailRenderer().render(self.event())
        self.assertIn("Total people visible: 0", output.text)
        self.assertIn("PERSON DETAILS\nNone", output.text)

    def test_html_escapes_dynamic_content(self):
        output = EmailRenderer().render(self.event([person("1", IdentityStatus.KNOWN, "<Ayush & Co>")]))
        self.assertIn(html.escape("<Ayush & Co>"), output.html)
        self.assertNotIn("Name: <Ayush & Co>", output.html)

    def test_cooldown(self):
        with tempfile.TemporaryDirectory() as folder:
            store = CooldownStore(Path(folder) / "state.json", cooldown_seconds=60)
            self.assertTrue(store.may_send("EV-1", now=100))
            store.record("EV-1", now=100)
            self.assertFalse(store.may_send("EV-1", now=120))
            self.assertTrue(store.may_send("EV-1", now=161))

    def test_alert_service_writes_one_canonical_artifact_set(self):
        with tempfile.TemporaryDirectory() as folder:
            rendered, artifacts = render_alert(self.event(), folder)
            self.assertEqual(rendered.subject, artifacts.subject.read_text(encoding="utf-8"))
            self.assertEqual(rendered.text, artifacts.plain_text.read_text(encoding="utf-8"))
            self.assertTrue(artifacts.html.is_file())
            self.assertTrue(artifacts.email_message.is_file())

    def test_extended_colour_vocabulary(self):
        examples = {
            "black": (10, 10, 10),
            "white": (250, 250, 250),
            "maroon": (90, 15, 25),
            "beige": (235, 220, 180),
            "olive": (105, 105, 35),
            "teal": (20, 120, 115),
            "cyan": (20, 210, 220),
            "navy": (15, 35, 100),
            "sky blue": (130, 190, 235),
        }
        for expected, rgb in examples.items():
            with self.subTest(expected=expected):
                self.assertEqual(_pixel_name(np.asarray(rgb)), expected)


if __name__ == "__main__":
    unittest.main()
