import multiprocessing as mp
import os

formatPathStr = lambda path: path.replace("\\\\", "/").replace("\\", "/")

dll_path = input("Enter libsm64 path: ")
if dll_path.endswith("dll"):
    dll_path = os.path.dirname(dll_path)
dll_path_str = formatPathStr(dll_path)

found_any_version = False
for version in ("us", "jp", "sh", "eu"):
    lib_full_path = os.path.join(dll_path, f"sm64_{version}.dll")
    if os.path.isfile(lib_full_path):
        print(f"Detected version: {version}")
        found_any_version = True
if not found_any_version:
    raise ValueError("Could not find sm64_XX.dll in path")

system_threads = mp.cpu_count()
print(f"""
Total system threads: {system_threads}
Recommended to use around half of this, as performance typically
is bottlenecked by memory usage from state loading anyway.
FPS will vary depending on how frequently the state is reloaded.
Try experimenting with different values, and check fps
and cpu usage to see what is best for you.""")
num_threads = int(input("Enter default number of threads: "))
print()

core_path = os.path.join(os.path.dirname(__file__), "core")
core_path_str = formatPathStr(core_path)
scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
for filename in os.listdir(scripts_path):
    if not filename.endswith(".py"):
        continue
    file_path = os.path.join(scripts_path, filename)
    if not os.path.isfile(file_path):
        continue

    with open(file_path, "rt") as fp:
        text = fp.read()

    lines = text.split("\n")
    set_dll_path = False
    set_num_threads = False
    set_core_path = False
    for i in range(len(lines)):
        line = lines[i]
        if (not set_dll_path and line.startswith("DLL_PATH = ")):
            lines[i] = f"DLL_PATH = \"{dll_path_str}\""
            set_dll_path = True
        elif (not set_num_threads and line.startswith("NUM_THREADS = ")):
            lines[i] = f"NUM_THREADS = {num_threads}"
            set_num_threads = True
        elif (not set_core_path and line.startswith("sys.path.append(")):
            lines[i] = f"sys.path.append(\"{core_path_str}\")"
            set_core_path = True
        if (set_dll_path and set_num_threads and set_core_path):
            break

    if (set_dll_path or set_num_threads or set_core_path):
        text = "\n".join(lines)
        with open(file_path, "wt") as fp:
            fp.write(text)
        print(f"Updated \"{filename}\"")
        
