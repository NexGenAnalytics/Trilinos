import sys
import subprocess
import os

def get_changed_files(start_commit, end_commit):
    """Get list of files changed between two commits."""
    cmd = ["git", "diff", "--name-only", start_commit, end_commit]
    result = subprocess.check_output(cmd).decode('utf-8')
    files = result.splitlines()

    # Filtering for C/C++ files and excluding ones with test/ or example/ anywhere in their paths
    c_cpp_files = [
        f for f in files
        if f.endswith(('.c', '.cpp', '.h', '.hpp'))
        and 'test/' not in f
        and 'tests/' not in f
        and 'unit_test' not in f
        and 'example/' not in f
        and 'examples/' not in f
    ]

    return c_cpp_files


def search_for_string_in_files(files, search_string):
    """Search for a string in the list of files."""
    matched_files = []

    for file in files:
        try:
            with open(file, 'r') as f:
                content = f.read()
                if search_string in content:
                    matched_files.append(file)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    return matched_files

if __name__ == "__main__":
    # In a GitHub action, the base commit and head commit can be specified as:
    start_commit = os.environ.get("BASE_REF")
    print(f"Start commit: {start_commit}")
    end_commit = os.environ.get("HEAD_REF")
    print(f"End commit: {end_commit}")

    changed_files = get_changed_files(start_commit, end_commit)
    matched_files = search_for_string_in_files(changed_files, "MPI_COMM_WORLD")

    if matched_files:
        print("Detected MPI_COMM_WORLD in the following files:")
        for file in matched_files:
            print(file)
        sys.exit(1)  # Exit with an error code to fail the GitHub Action
    else:
        print("No addition of MPI_COMM_WORLD detected.")
        sys.exit(0)
