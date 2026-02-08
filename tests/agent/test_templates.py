"""Tests for persona templates."""

from roshni.agent.templates import AVAILABLE_TONES, get_identity_template, get_soul_template, get_user_template


class TestTemplates:
    def test_available_tones(self):
        assert "friendly" in AVAILABLE_TONES
        assert "professional" in AVAILABLE_TONES
        assert "warm" in AVAILABLE_TONES
        assert "witty" in AVAILABLE_TONES

    def test_identity_templates_load(self):
        for tone in AVAILABLE_TONES:
            template = get_identity_template(tone)
            assert "{bot_name}" in template
            assert len(template) > 100

    def test_soul_template(self):
        template = get_soul_template()
        assert "{user_name}" in template
        assert "Values" in template or "Principles" in template

    def test_user_template(self):
        template = get_user_template()
        assert "{user_name}" in template

    def test_placeholder_replacement(self):
        template = get_identity_template("friendly")
        filled = template.replace("{bot_name}", "Roshni").replace("{user_name}", "Alice")
        assert "Roshni" in filled
        assert "{bot_name}" not in filled
