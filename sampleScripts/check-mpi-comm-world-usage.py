import sys
import os
import subprocess
import re

def get_changed_files(start_commit, end_commit):
    """Get a dictionary of files and their changed lines between two commits where MPI_COMM_WORLD was added."""
    cmd = ["git", "diff", "-U0", "--ignore-all-space", start_commit, end_commit]
    result = subprocess.check_output(cmd).decode('utf-8')

    # Regex to capture filename and the line numbers of the changes
    file_pattern = re.compile(r'^\+\+\+ b/(.*?)$', re.MULTILINE)
    line_pattern = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', re.MULTILINE)

    files = {}
    for match in file_pattern.finditer(result):
        file_name = match.group(1)

        # Filtering for C/C++ files and excluding certain directories
        if file_name.endswith(('.c', '.cpp', '.h', '.hpp')) and all(
            excluded not in file_name for excluded in ['test/', 'tests/', 'unit_test', 'example/', 'examples/']
        ):
            # Find the lines that changed for this file
            lines_start_at = match.end()
            next_file_match = file_pattern.search(result, pos=match.span(0)[1])

            # Slice out the part of the diff that pertains to this file
            file_diff = result[lines_start_at:next_file_match.span(0)[0] if next_file_match else None]

            # Extract line numbers of the changes
            changed_lines = []
            for line_match in line_pattern.finditer(file_diff):
                start_line = int(line_match.group(1))
                num_lines = int(line_match.group(2) or 1)

                # The start and end positions for this chunk of diff
                chunk_start = line_match.end()
                next_chunk = line_pattern.search(file_diff, pos=line_match.span(0)[1])
                chunk_diff = file_diff[chunk_start:next_chunk.span(0)[0] if next_chunk else None]

                # Only include if "MPI_COMM_WORLD" is added and "CHECK: ALLOW MPI_COMM_WORLD" isn't present
                if "MPI_COMM_WORLD" in chunk_diff and "CHECK: ALLOW MPI_COMM_WORLD" not in chunk_diff:
                    changed_lines.extend(range(start_line, start_line + num_lines))

            if changed_lines:
                files[file_name] = changed_lines

    return files


if __name__ == "__main__":
    # In a GitHub action, the base commit and head commit can be specified as:
    start_commit = os.environ.get("BASE_REF")
    print(f"Start commit: {start_commit}")
    end_commit = os.environ.get("HEAD_REF")
    print(f"End commit: {end_commit}")

    changed_files = get_changed_files(start_commit, end_commit)

    if changed_files:
        print("Detected MPI_COMM_WORLD in the following files:")
        for file_name, lines in changed_files.items():
            print(f"File: {file_name}")
            print("Changed Lines:", ', '.join(map(str, lines)))
            print("-----")
        sys.exit(1)  # Exit with an error code to fail the GitHub Action
    else:
        print("No addition of MPI_COMM_WORLD detected.")
        sys.exit(0)
