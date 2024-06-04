"""
Minimal utilities for parsing Secondary files.

(Secondary is my personal note-taking software.)
"""

from typing import Tuple, List, Dict

import re
import string


TAG_PATTERN = re.compile('(?s)\\{\\{(.*?)\\}\\}')
DATE_PATTERN = re.compile('^\\*\\*([0-9]{4}-[0-9]{2}-[0-9]{2})\\*\\*')
HEADING_PATTERN = re.compile('^#+\s+(.*)')


def parse_secondary_file(content: str) -> List[Dict]:
    """parse a secondary file into a list of items"""

    items = []

    lines = content.split("\n")

    # 0 - before start of item
    # 1 - in header
    # 2 - in notes
    state = 0
    item = None

    for line in lines:
        if line.startswith('!'):
            if item is not None:
                items.append(item)
            item = {}
            state = 1
        elif state == 1:
            if line.strip() == "":
                state = 2
            else:
                line_split = re.split(':\\s+', line)
                item[line_split[0]] = ": ".join(line_split[1:])
        elif state == 2:
            notes = item.get('notes', '')
            item["notes"] = notes + line + "\n"

    if item is not None and len(item) > 0:
        items.append(item)

    for item in items:
        if item.get('id') is None and item.get('name') is not None:
            item_id = item.get('name')
            item_id = item_id.lower()
            for c in string.punctuation:
                item_id = item_id.replace(c, '')
            item_id = item_id.replace(' ', '_')
            item['id'] = item_id

    return items


def parse_sections(notes: str, pattern: re.Pattern) -> List[Tuple[str, str]]:
    """
    Parse the notes of a secondary items into sections
    using heading conventions defined in `pattern`.
    """

    lines = notes.split('\n')

    sections = []
    sec_lines = []
    sec_name = None

    for line in lines:
        match = pattern.match(line)
        if match:
            if sec_name is not None:
                sections.append((sec_name, '\n'.join(sec_lines).strip()))
            sec_name = match.group(1)
            sec_lines = []
        else:
            sec_lines.append(line)

    if sec_name is not None:
        sections.append((
            sec_name,
            '\n'.join(sec_lines).strip()
        ))

    return sections


def split_paragraphs(text: str) -> List[str]:
    """Split markdown into paragraphs."""
    res = re.split('\n\n+', text)
    # remove any empty results (from two blank lines at end of section)
    res = [x for x in res if x and x != '&nbsp;']
    return res


def remove_tags(text: str) -> str:
    """Remove secondary tags in their entirety from text."""
    return TAG_PATTERN.sub('', text)


def remove_tag_brackets(text: str) -> str:
    """Remove curly brackets (replace tag with capturing group)."""
    return TAG_PATTERN.sub('\\g<1>', text)
