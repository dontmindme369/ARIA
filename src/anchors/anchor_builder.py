#!/usr/bin/env python3
"""
Anchor Builder - Tool to Expand Anchors to Monolithic Scale

This tool helps expand current 200-300 line anchors into 3,000-5,000 line
monolithic anchors with comprehensive examples, constraints, and anti-patterns.

Usage:
    python anchor_builder.py --anchor technical --output technical_expanded.md
    python anchor_builder.py --all --output-dir ./expanded_anchors/
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass


@dataclass
class AnchorSection:
    """A section to include in expanded anchor"""

    title: str
    content: str
    size_target: int  # Target number of lines


class AnchorBuilder:
    """Build monolithic anchors from templates"""

    # Standard sections for all anchors
    STANDARD_SECTIONS = [
        "Core Identity",
        "Context Interpretation",
        "Synthesis Protocols",
        "Response Structure",
        "Quality Standards",
        "Common Pitfalls",
        "Edge Cases",
        "Reasoning Model Guidelines",
        "Prevention Strategies",
        "Real Examples",
    ]

    # Example counts for each section type
    EXAMPLE_TARGETS = {
        "positive_examples": 500,  # "DO THIS" examples
        "negative_examples": 500,  # "DON'T DO THIS" examples
        "edge_cases": 200,  # Unusual scenarios
        "correction_examples": 300,  # Before/after corrections
        "reasoning_constraints": 400,  # Specific constraints for reasoning models
        "quality_checks": 200,  # Verification checklists
        "anti_patterns": 300,  # Common mistakes to avoid
        "query_patterns": 400,  # Query types that trigger this anchor
    }

    def __init__(self, anchor_name: str, current_anchor_path: Optional[Path] = None):
        """
        Initialize builder for specific anchor

        Args:
            anchor_name: Name of anchor (formal, casual, technical, etc.)
            current_anchor_path: Path to current anchor file to expand from
        """
        self.anchor_name = anchor_name
        self.current_content = (
            self._load_current(current_anchor_path) if current_anchor_path else ""
        )
        self.sections = []
        self.total_lines = 0

    def _load_current(self, path: Path) -> str:
        """Load current anchor content"""
        if path.exists():
            return path.read_text()
        return ""

    def add_section(self, title: str, content: str, size_target: int = 100):
        """Add a section to the anchor"""
        section = AnchorSection(title=title, content=content, size_target=size_target)
        self.sections.append(section)
        self.total_lines += len(content.split("\n"))

    def build_comprehensive_examples_section(
        self, example_type: str, count_target: int = 500
    ) -> str:
        """
        Build a section with many examples

        Args:
            example_type: Type of examples (positive, negative, edge_case, etc.)
            count_target: Target number of examples

        Returns:
            Formatted section content
        """
        section_content = [
            f"## {example_type.replace('_', ' ').title()} ({count_target} Examples)",
            "",
            f"This section provides {count_target} examples of {example_type.replace('_', ' ')}.",
            "",
        ]

        # Template for expansion
        template = self._get_example_template(example_type)

        section_content.extend(
            [
                "### TEMPLATE FOR EXPANSION",
                "",
                "```",
                template,
                "```",
                "",
                f"### Instructions:",
                f"- Create {count_target} examples following this template",
                f"- Cover diverse scenarios within {self.anchor_name} anchor context",
                "- Each example should be unique and instructive",
                "- Group by theme/topic for easy reference",
                "",
                "### Example Groups to Cover:",
                *self._get_example_groups(example_type),
                "",
                f"### [PLACEHOLDER: {count_target} examples to be added]",
                "",
            ]
        )

        return "\n".join(section_content)

    def _get_example_template(self, example_type: str) -> str:
        """Get template for specific example type"""
        templates = {
            "positive_examples": """
**[ID] Example Title**

Scenario: [Description of situation]
Query: "[User query]"
Response: "[Correct response showing best practices for this anchor]"
Why it works: [Explanation of what makes this good]
Key elements: [Bullet points of important aspects]
""",
            "negative_examples": """
**[ID] Anti-Pattern: Example Title**

Scenario: [Description of situation]
Query: "[User query]"
Wrong response: "[Bad response that violates this anchor's principles]"
Why it fails: [Explanation of what's wrong]
Correct approach: "[How to fix it]"
""",
            "edge_cases": """
**[ID] Edge Case: Example Title**

Unusual scenario: [Description]
Query: "[User query with edge case]"
Challenge: [What makes this tricky]
Correct response: "[How to handle it]"
Key insight: [What to remember for similar cases]
""",
            "correction_examples": """
**[ID] Correction: Example Title**

Original query: "[User query]"
Wrong response: "[Initial wrong response]"
User feedback: "[What user said was wrong]"
Corrected response: "[Fixed response]"
Lesson learned: [What to avoid in future]
""",
        }
        return templates.get(example_type, "Generic template")

    def _get_example_groups(self, example_type: str) -> List[str]:
        """Get list of example groups to cover for given type"""
        base_groups = [
            "- Basic scenarios",
            "- Intermediate complexity",
            "- Advanced/complex cases",
            "- Multi-step interaction scenarios",
            "- Edge cases and unusual situations",
            "- Common user patterns",
            "- Reasoning model specific cases",
            "- Error-prone scenarios",
        ]

        # Add anchor-specific groups
        anchor_specific = {
            "technical": [
                "- Code examples across languages",
                "- Debugging scenarios",
                "- Architecture questions",
                "- Implementation details",
                "- Performance optimization",
            ],
            "educational": [
                "- Concept explanations",
                "- Progressive complexity",
                "- Analogy usage",
                "- Student misconceptions",
                "- Learning progressions",
            ],
            "formal": [
                "- Research citations",
                "- Academic rigor",
                "- Scholarly analysis",
                "- Literature reviews",
                "- Methodological discussions",
            ],
            # Add more anchor-specific groups
        }

        return base_groups + anchor_specific.get(self.anchor_name, [])

    def build_reasoning_model_section(self) -> str:
        """Build comprehensive reasoning model guidelines"""
        content = [
            "## Reasoning Model Specific Guidelines (400 Constraints)",
            "",
            "### For QwQ / O1 / R1 Models",
            "",
            "Reasoning models require EXPLICIT, PRESCRIPTIVE constraints.",
            "They will optimize for 'best explanation' unless told otherwise.",
            "",
            "[PLACEHOLDER: 400 specific constraints for reasoning models]",
            "",
            "### Examples of Good Constraints:",
            "",
            "- DO NOT write long explanations before the answer",
            "- Start with the direct answer, then explain",
            "- Keep responses under 500 words unless asked for more",
            "- Use code blocks for all code examples",
            "- Cite sources when making factual claims",
            "",
            "[PLACEHOLDER: 395 more reasoning model constraints]",
            "",
        ]

        return "\n".join(content)

    def build_quality_checklist_section(self) -> str:
        """Build comprehensive quality checklist"""
        content = [
            "## Quality Verification Checklist (200 Checks)",
            "",
            "### Level 1: Basic Quality (50 checks)",
            "",
            "[PLACEHOLDER: 50 basic quality checks]",
            "",
            "### Level 2: Content Quality (50 checks)",
            "",
            "- Response directly answers the query",
            "- Information is accurate",
            "- Examples are relevant",
            "- Tone matches anchor mode",
            "- Clarity meets anchor standards",
            "",
            "[PLACEHOLDER: 45 more content quality checks]",
            "",
            "### Level 3: Anchor-Specific (50 checks)",
            "",
            f"[PLACEHOLDER: 50 checks specific to {self.anchor_name} anchor]",
            "",
            "### Level 4: Edge Case Handling (50 checks)",
            "",
            "[PLACEHOLDER: 50 edge case checks]",
            "",
        ]

        return "\n".join(content)

    def build_anti_patterns_section(self) -> str:
        """Build section documenting common mistakes"""
        content = [
            "## Common Anti-Patterns to Avoid (300 Examples)",
            "",
            f"These are mistakes frequently made in {self.anchor_name} mode:",
            "",
            "### Category 1: Level Mismatches (100 examples)",
            "",
            "[PLACEHOLDER: 100 level mismatch anti-patterns]",
            "",
            "### Category 2: Scope Issues (100 examples)",
            "",
            "[PLACEHOLDER: 100 scope issue anti-patterns]",
            "",
            "### Category 3: Structural Problems (100 examples)",
            "",
            "[PLACEHOLDER: 100 structural anti-patterns]",
            "",
        ]

        return "\n".join(content)

    def generate_expansion_plan(self) -> Dict[str, Any]:
        """Generate plan showing what needs to be added"""
        plan: Dict[str, Any] = {
            "current_lines": (
                len(self.current_content.split("\n")) if self.current_content else 0
            ),
            "target_lines": 3000,
            "sections_planned": {},
        }

        for section_name, target_count in self.EXAMPLE_TARGETS.items():
            # Estimate lines per example (avg 8 lines)
            estimated_lines = target_count * 8
            plan["sections_planned"][section_name] = {
                "examples": target_count,
                "estimated_lines": estimated_lines,
            }

        total_planned = sum(
            s["estimated_lines"] for s in plan["sections_planned"].values()
        )
        plan["total_planned_lines"] = plan["current_lines"] + total_planned

        return plan

    def build_skeleton(self) -> str:
        """Build skeleton anchor with placeholders for expansion"""
        sections = [
            f"# {self.anchor_name.title()} Anchor - Monolithic Version",
            "",
            f"## Status: EXPANSION IN PROGRESS",
            f"Current size: {len(self.current_content.split(chr(10)))} lines",
            f"Target size: 3,000+ lines",
            "",
            "---",
            "",
            "## SECTION 1: Core Identity",
            "",
            (
                self.current_content
                if self.current_content
                else f"[Current {self.anchor_name} anchor content]"
            ),
            "",
            "---",
            "",
            "## SECTION 2: Comprehensive Examples",
            "",
            self.build_comprehensive_examples_section("positive_examples", 500),
            "",
            self.build_comprehensive_examples_section("negative_examples", 500),
            "",
            self.build_comprehensive_examples_section("edge_cases", 200),
            "",
            self.build_comprehensive_examples_section("correction_examples", 300),
            "",
            "---",
            "",
            "## SECTION 3: Reasoning Model Guidelines",
            "",
            self.build_reasoning_model_section(),
            "",
            "---",
            "",
            "## SECTION 4: Quality Standards",
            "",
            self.build_quality_checklist_section(),
            "",
            "---",
            "",
            "## SECTION 5: Anti-Patterns",
            "",
            self.build_anti_patterns_section(),
            "",
            "---",
            "",
            "## SECTION 6: Expansion Notes",
            "",
            "### How to Complete This Anchor:",
            "",
            "1. **Replace all [PLACEHOLDER] sections** with actual content",
            "2. **Follow the templates provided** for each example type",
            "3. **Maintain consistency** across all examples",
            "4. **Test examples** with reasoning models to verify they work",
            "5. **Organize examples** by theme/category for easy reference",
            "",
            "### Expansion Checklist:",
            "",
            "- [ ] 500 positive examples added",
            "- [ ] 500 negative examples added",
            "- [ ] 200 edge cases added",
            "- [ ] 300 correction examples added",
            "- [ ] 400 reasoning constraints added",
            "- [ ] 200 quality checks added",
            "- [ ] 300 anti-patterns added",
            "- [ ] Total line count > 3,000",
            "",
            "---",
            "",
            f"**END OF {self.anchor_name.upper()} ANCHOR SKELETON**",
        ]

        return "\n".join(sections)

    def save(self, output_path: Path):
        """Save built anchor to file"""
        skeleton = self.build_skeleton()
        output_path.write_text(skeleton)
        print(f"✅ Saved {self.anchor_name} anchor skeleton to {output_path}")
        print(f"   Current size: {len(skeleton.split(chr(10)))} lines")
        print(f"   Target size: 3,000+ lines")
        print(f"   Status: Ready for expansion")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Build monolithic anchor files")
    parser.add_argument(
        "--anchor",
        type=str,
        choices=[
            "formal",
            "casual",
            "technical",
            "educational",
            "philosophical",
            "analytical",
            "factual",
            "creative",
            "feedback_correction",
        ],
        help="Anchor to build/expand",
    )
    parser.add_argument("--all", action="store_true", help="Build all anchors")
    parser.add_argument(
        "--current", type=Path, help="Path to current anchor file to expand from"
    )
    parser.add_argument("--output", type=Path, help="Output path for built anchor")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./expanded_anchors"),
        help="Output directory for all anchors (when using --all)",
    )
    parser.add_argument(
        "--plan", action="store_true", help="Just show expansion plan without building"
    )

    args = parser.parse_args()

    if not args.anchor and not args.all:
        parser.error("Must specify --anchor or --all")

    # Build single anchor
    if args.anchor:
        builder = AnchorBuilder(args.anchor, args.current)

        if args.plan:
            plan = builder.generate_expansion_plan()
            print("\n" + "=" * 70)
            print(f"EXPANSION PLAN: {args.anchor} anchor")
            print("=" * 70)
            print(f"\nCurrent size: {plan['current_lines']} lines")
            print(f"Target size: {plan['target_lines']} lines")
            print(f"\nSections to add:")
            for section, details in plan["sections_planned"].items():
                print(
                    f"  - {section}: {details['examples']} examples (~{details['estimated_lines']} lines)"
                )
            print(f"\nEstimated final size: {plan['total_planned_lines']} lines")
            print("=" * 70)
        else:
            output = args.output or Path(f"{args.anchor}_expanded.md")
            builder.save(output)

    # Build all anchors
    elif args.all:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        all_anchors = [
            "formal",
            "casual",
            "technical",
            "educational",
            "philosophical",
            "analytical",
            "factual",
            "creative",
            "feedback_correction",
        ]

        for anchor_name in all_anchors:
            builder = AnchorBuilder(anchor_name)
            output = args.output_dir / f"{anchor_name}_expanded.md"
            builder.save(output)

        print(f"\n✅ All anchor skeletons created in {args.output_dir}")
        print("   Next step: Expand placeholders with actual content")


if __name__ == "__main__":
    main()
