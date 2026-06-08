import html
import os
import re


WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "workspace")


def _parse_skill_frontmatter(content: str) -> dict:
    parts = re.split(r"^---\s*$", content, maxsplit=2, flags=re.MULTILINE)
    if len(parts) < 3:
        return {}
    meta = {}
    for line in parts[1].splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


class SkillRegistry:
    def __init__(self, workspace_dir: str = WORKSPACE_DIR):
        self._workspace_dir = workspace_dir
        self._skills: list[dict] | None = None

    def load_skills(self) -> list[dict]:
        """Read workspace/skills/, parse each SKILL.md, and cache the results."""
        if self._skills is not None:
            return self._skills

        skills_dir = os.path.join(self._workspace_dir, "skills")
        try:
            entries = sorted(os.listdir(skills_dir))
        except OSError:
            self._skills = []
            return self._skills

        skills = []
        for name in entries:
            dir_path = os.path.join(skills_dir, name)
            if not os.path.isdir(dir_path):
                continue
            skill_file = os.path.join(dir_path, "SKILL.md")
            try:
                with open(skill_file, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue
            meta = _parse_skill_frontmatter(content)
            skills.append({
                "name": meta.get("name", name),
                "description": meta.get("description", ""),
                "location": os.path.join(skills_dir, name, "SKILL.md"),
                "directory": os.path.join(skills_dir, name),
            })

        self._skills = skills
        return self._skills

    def get_skills_index(self) -> str:
        """Return a formatted XML skills index string for use in the system prompt."""
        skills = self.load_skills()
        if not skills:
            return ""

        xml_entries = "\n".join(
            f"  <skill>\n"
            f"    <name>{html.escape(s['name'])}</name>\n"
            f"    <description>{html.escape(s['description'])}</description>\n"
            f"    <location>{html.escape(s['location'])}</location>\n"
            f"    <directory>{html.escape(s['directory'])}</directory>\n"
            f"  </skill>"
            for s in skills
        )
        return (
            "When a task matches one of the skills below, use the `read_file` tool to "
            "load the SKILL.md at the listed location for detailed instructions.\n\n"
            "All scripts and paths referenced inside a SKILL.md are relative to that "
            "skill's <directory>. For example, if a skill says `uv run ./scripts/foo.py`, "
            "the full path is <directory>/scripts/foo.py. Always prefix script paths with "
            "the skill's <directory> when calling run_command.\n\n"
            f"<available_skills>\n{xml_entries}\n</available_skills>"
        )
