import subprocess


def count_lines_fast(filename):
    print(f"Counting lines in {filename}...")
    result = subprocess.run(["wc", "-l", filename], capture_output=True, text=True)
    result = int(result.stdout.split()[0])
    print(f"Found {result} lines.")
    return result
