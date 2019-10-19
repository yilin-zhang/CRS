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

class McGillParser():
    def parse_directory(self, dirname):
        dir_chords = []
        for root, _, files in os.walk(dirname):
            for filename in files:
                # only parse the file if the extension is txt
                if os.path.splitext(filename)[1] == '.txt':
                    filename = os.path.join(root, filename) 
                    dir_chords.append(self.parse_file(filename))
        return dir_chords

    def parse_file(self, filename):
        '''Parsing the file'''
        with open(filename, 'r') as lines:
            section = []
            chord_sequences = []
            for idx, line in enumerate(lines):
                # 0~3 meta data, 4 blank line, 5 silence
                if idx <= 5:
                    continue
                # If the line is the start of a new section
                # process the lines of the previous section
                if self._is_section_start_point(line) and idx > 6:
                    chord_sequences.append(self._extract_chords(section))
                    section = [] # clean section list
                section.append(line)
            chord_sequences.append(self._extract_chords(section))
        return chord_sequences


    def _is_section_start_point(self, line):
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

    def _extract_chords(self, section):
        '''Extract chords from a section
        Arg:
        - section: A list of several lines of verse or chorus.
        Return:
        - chords: A list of chords
        '''
        def substitute_attribute(chord):
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

if __name__ == '__main__':
    parser = McGillParser()
    chords = parser.parse_directory('McGill-Billboard')
    print(chords)