'''
Parsing the Mcgill Billboard dataset, and returning a list.

For extracting chords from the whole dataset directory,
use parse_directory()

The structure of the list:
[       ] <- Whole dataset
[    ] <- Song
[ ] <- Section (Verse/Chorus)
'''

import re
import os
from typing import *

class McGillParser():
    def parse_directory(self, dirname: str) -> Iterator[List[Tuple[str, str]]]:
        for root, _, files in os.walk(dirname):
            for filename in files:
                # only parse the file if the extension is txt
                if os.path.splitext(filename)[1] == '.txt':
                    filename = os.path.join(root, filename) 
                    for chords in self.parse_file(filename):
                        yield chords

    def parse_file(self, filename: str) -> Iterator[List[Tuple[str, str]]]:
        '''Parsing the file'''
        with open(filename, 'r') as lines:
            section = []
            for idx, line in enumerate(lines):
                # 0~3 meta data, 4 blank line, 5 silence
                if idx <= 5:
                    continue
                # If the line is the start of a new section
                # process the lines of the previous section
                else:
                    if self._is_section_start_point(line):
                        chords = self._extract_chords(section)
                        # only export chords when it's not empty
                        if chords:
                            yield chords
                        section = [] # clean section list
                    elif self._is_transposition_start_point(line):
                        chords = self._extract_chords(section)
                        # only export chords when it's not empty
                        if chords:
                            yield chords
                        section = [] # clean section list
                        continue # skip the current line
                section.append(line)


    def _is_section_start_point(self, line: str) -> bool:
        '''If the line is the start of a new section
        Arg:
        - line: a line of text
        Return:
        - A bool value
        '''
        if re.search(r'^[0-9]+\.[0-9]+(\s+|\t+)[A-Z]', line):
            return True
        else:
            return False

    def _is_transposition_start_point(self, line: str) -> bool:
        '''If the line is the start of a new key 
        Arg:
        - line: a line of text
        Return:
        - A bool value
        '''
        if re.search(r'^#', line):
            return True
        else:
            return False

    def _extract_chords(self, section: List[str]) -> List[Tuple[str, str]]:
        '''Extract chords from a section
        Arg:
        - section: A list of several lines of verse or chorus.
        Return:
        - chords: A list of chords
        '''
        def substitute_attribute(chord: Tuple[str, str]) -> Tuple[str, str]:
            '''chord example: ('A', 'maj')'''
            attr = chord[1]
            if attr != 'maj' and attr != 'min':
                return (chord[0], 'maj')
            else:
                return chord
        chords = []
        for line in section:
            chords_in_line = re.findall(r'([A-G])b{0,1}\#{0,1}\:(maj|min|.|)', line)
            chords_in_line = list(map(substitute_attribute, chords_in_line))
            chords = chords + chords_in_line
        return chords
