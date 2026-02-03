"""
Tests for the AI analysis module.

These tests cover:
- Prompt building functions
- AIAnalyzer class methods
- Cross-experiment analysis functionality
"""

from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from spkmc.analysis.prompts import CROSS_EXPERIMENT_SYSTEM_PROMPT, build_cross_experiment_prompt


class TestBuildCrossExperimentPrompt:
    """Tests for the build_cross_experiment_prompt function."""

    def test_empty_list_returns_prompt_with_no_analyses(self):
        """An empty list should still return a valid prompt structure."""
        result = build_cross_experiment_prompt([])
        assert "Individual Experiment Analyses" in result
        assert "## Task" in result

    def test_single_analysis(self):
        """A single analysis should be included in the prompt."""
        analyses = [{"name": "Test Experiment", "analysis": "This is a test analysis."}]
        result = build_cross_experiment_prompt(analyses)

        assert "### Test Experiment" in result
        assert "This is a test analysis." in result
        assert "## Task" in result

    def test_multiple_analyses(self):
        """Multiple analyses should all be included in the prompt."""
        analyses = [
            {"name": "Experiment A", "analysis": "Analysis content A."},
            {"name": "Experiment B", "analysis": "Analysis content B."},
            {"name": "Experiment C", "analysis": "Analysis content C."},
        ]
        result = build_cross_experiment_prompt(analyses)

        assert "### Experiment A" in result
        assert "### Experiment B" in result
        assert "### Experiment C" in result
        assert "Analysis content A." in result
        assert "Analysis content B." in result
        assert "Analysis content C." in result

    def test_analyses_separated_by_dividers(self):
        """Analyses should be separated by dividers."""
        analyses = [
            {"name": "Exp 1", "analysis": "Content 1"},
            {"name": "Exp 2", "analysis": "Content 2"},
        ]
        result = build_cross_experiment_prompt(analyses)

        # Each analysis block should end with ---
        assert result.count("---") >= 2


class TestCrossExperimentSystemPrompt:
    """Tests for the cross-experiment system prompt constant."""

    def test_prompt_exists_and_is_non_empty(self):
        """The system prompt should exist and contain content."""
        assert CROSS_EXPERIMENT_SYSTEM_PROMPT
        assert len(CROSS_EXPERIMENT_SYSTEM_PROMPT) > 100

    def test_prompt_mentions_meta_analysis(self):
        """The prompt should reference meta-analysis."""
        assert "meta-analysis" in CROSS_EXPERIMENT_SYSTEM_PROMPT.lower()

    def test_prompt_mentions_required_sections(self):
        """The prompt should mention required output sections."""
        prompt = CROSS_EXPERIMENT_SYSTEM_PROMPT.lower()
        assert "executive summary" in prompt
        assert "cross-experiment patterns" in prompt
        assert "unified conclusions" in prompt
        assert "implications" in prompt


class TestAIAnalyzerCrossExperiment:
    """Tests for AIAnalyzer.generate_cross_experiment_analysis method."""

    @pytest.fixture
    def mock_openai_response(self):
        """Create a mock OpenAI response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated cross-experiment analysis content."
        return mock_response

    @pytest.fixture
    def sample_analyses(self) -> List[Dict[str, str]]:
        """Sample experiment analyses for testing."""
        return [
            {"name": "Network Topology", "analysis": "# Analysis\n\nNetwork effects observed."},
            {"name": "Distribution Impact", "analysis": "# Analysis\n\nDistribution matters."},
        ]

    def test_skips_if_file_exists_and_not_force(self, tmp_path, sample_analyses):
        """Should skip generation if file exists and force=False."""
        from spkmc.analysis.ai_analyzer import AIAnalyzer

        output_path = tmp_path / "cross_experiment_analysis.md"
        output_path.write_text("Existing content")

        analyzer = AIAnalyzer()
        result = analyzer.generate_cross_experiment_analysis(
            sample_analyses, output_path, force=False
        )

        assert result is None
        assert output_path.read_text() == "Existing content"

    def test_regenerates_if_force_true(self, tmp_path, sample_analyses, mock_openai_response):
        """Should regenerate if force=True even if file exists."""
        from spkmc.analysis.ai_analyzer import AIAnalyzer

        output_path = tmp_path / "cross_experiment_analysis.md"
        output_path.write_text("Existing content")

        analyzer = AIAnalyzer()

        with patch.object(analyzer, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_openai_response
            result = analyzer.generate_cross_experiment_analysis(
                sample_analyses, output_path, force=True
            )

        assert result == str(output_path)
        assert "Generated cross-experiment analysis" in output_path.read_text()

    def test_returns_none_for_empty_analyses(self, tmp_path):
        """Should return None if no analyses provided."""
        from spkmc.analysis.ai_analyzer import AIAnalyzer

        output_path = tmp_path / "cross_experiment_analysis.md"
        analyzer = AIAnalyzer()
        result = analyzer.generate_cross_experiment_analysis([], output_path, force=False)

        assert result is None
        assert not output_path.exists()

    def test_creates_file_with_correct_structure(
        self, tmp_path, sample_analyses, mock_openai_response
    ):
        """Generated file should have correct header structure."""
        from spkmc.analysis.ai_analyzer import AIAnalyzer

        output_path = tmp_path / "cross_experiment_analysis.md"
        analyzer = AIAnalyzer()

        with patch.object(analyzer, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_openai_response
            result = analyzer.generate_cross_experiment_analysis(
                sample_analyses, output_path, force=False
            )

        assert result == str(output_path)
        content = output_path.read_text()

        assert "# Cross-Experiment Analysis" in content
        assert "**Experiments Analyzed:**" in content
        assert "Network Topology" in content
        assert "Distribution Impact" in content
        assert "Generated by AI meta-analysis" in content

    def test_uses_cross_experiment_system_prompt(
        self, tmp_path, sample_analyses, mock_openai_response
    ):
        """Should use CROSS_EXPERIMENT_SYSTEM_PROMPT for API call."""
        from spkmc.analysis.ai_analyzer import AIAnalyzer

        output_path = tmp_path / "cross_experiment_analysis.md"
        analyzer = AIAnalyzer()

        with patch.object(analyzer, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_openai_response
            analyzer.generate_cross_experiment_analysis(sample_analyses, output_path, force=False)

            # Verify the API was called with correct system prompt
            call_args = mock_client.return_value.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            system_message = messages[0]

            assert system_message["role"] == "system"
            assert "meta-analysis" in system_message["content"].lower()


class TestExperimentsCommandAnalyzeFlag:
    """Tests for the --analyze flag on the experiments command."""

    def test_experiments_help_shows_analyze_flag(self, runner):
        """The experiments command help should show --analyze flag."""
        from spkmc.cli.commands import cli

        result = runner.invoke(cli, ["experiments", "--help"])
        assert "--analyze" in result.output
        assert "AI analysis" in result.output

    def test_analyze_flag_requires_api_key(self, runner, monkeypatch):
        """Using --analyze without OPENAI_API_KEY should fail early."""
        from spkmc.cli.commands import cli

        # Ensure API key is not set
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        result = runner.invoke(cli, ["experiments", "--analyze", "--all"])

        assert result.exit_code != 0
        assert "OPENAI_API_KEY" in result.output


@pytest.fixture
def runner():
    """Create a Click test runner."""
    from click.testing import CliRunner

    return CliRunner()
