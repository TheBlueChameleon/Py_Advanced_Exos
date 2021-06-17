#!/usr/bin/python3

# ============================================================================ #
# dependencies

import os
import sys
import glob
import ctypes                                   # see "Additional Feature" block

# ============================================================================ #
# problem 1: File System Scan

# ---------------------------------------------------------------------------- #
# Additional Feature
# Detection of hidden files
# (This was neither in the lecture nor asked for on the problem paper.)
#
# Files can be hidden from the end user to protect them from meddling.
# Of course, the user can make them visible, but this takes deliberate action
# and should warn the user to only change hidden files if they really know what
# they're doing.
#
# Unixoid systems (linux, mac) follow the convention that the file names of
# hidden files begin with a dot ('.').
# Windows realizes this with attributes that are stored within the inode.
# 
# Source of this code snippet:
# https://stackoverflow.com/questions/284115/cross-platform-hidden-file-detection

def is_hidden(filepath):
    name = os.path.basename(os.path.abspath(filepath))
    return name.startswith('.') or has_hidden_attribute(filepath)

def has_hidden_attribute(filepath):
    try:
        attrs = ctypes.windll.kernel32.GetFileAttributesW(filepath)
        assert attrs != -1
        result = bool(attrs & 2)
    except (AttributeError, AssertionError):
        result = False
    return result

# ---------------------------------------------------------------------------- #
# Determine whether to use CWD or user specified directory as root directory
# Include additional feature: if 2nd command line argument is 'includeHidden',
# also scan for hidden files.
#
# Note that there is a module dedicated to parsing command line arguments:
# https://docs.python.org/3/library/argparse.html

root = ""
if len(sys.argv) == 1 :
    root = os.getcwd()
    print("No Command Line Argument -- exploring CWD:")
    print("  ", root)
else :
    root = sys.argv[1]
    print("Specified root directory:")
    print("  ", root)
    
    if os.path.isdir(root) :
        print("is a directory and will be used as root directory.")
    else :
        print("is not a directory -- defaulting to CWD:")
        root = os.getcwd()
        print("  ", root)

includeHidden = False
if len(sys.argv) >= 3 :
    if sys.argv[2] == "includeHidden" : includeHidden = True

print("Hidden filesa are" +\
      "" if includeHidden else "not",
      "included in the scan.")
print()

# ---------------------------------------------------------------------------- #
# helper class: dict with default values
# if a read access with a non-existing key is made, the defaultDict instantiates
# the key with a default value

class defaultDict (dict) :
    def __init__(self, default, *args, **kwargs) :
        self.default = default
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, key) :
        if not key in self.keys() :
            self[key] = self.default
        
        return super().__getitem__(key)

# ---------------------------------------------------------------------------- #
# recursively traverse the file system and sum up the sizes, with no regard of
# the files in subdirectories

sizesLocal = defaultDict(0)

directories = glob.glob(os.path.join(root, "**/"), recursive=True)

for currentDirectory in directories :
    
    with os.scandir(currentDirectory) as it:
        for entry in it:
            if (includeHidden or not is_hidden(entry)) and entry.is_file() :
                sizesLocal[os.path.relpath(currentDirectory, root)] += entry.stat().st_size

# ---------------------------------------------------------------------------- #
# sum up file sizes for parent directories

sizesAccumulated = defaultDict(0)

for currentDirectory in sizesLocal :
    pathElements = currentDirectory.split('/')
    
    for chainlength in range( 1, len(pathElements) + 1 ) :
        addToDir = os.path.join(
            *(currentDirectory.split('/')[:chainlength])
        )
        sizesAccumulated[addToDir] += sizesLocal[currentDirectory]

# the root directory is represented as '.' in this -- handle this manually has 
# to be done in a second loop so that the accumulation of the sizes of the 
# subdirectories is complete.
# one version would be using the recorded data:
#for currentDirectory in sizesLocal :
    #if not ('/' in currentDirectory) and not ('.' in currentDirectory) :
        #sizesAccumulated['.'] += sizesAccumulated[currentDirectory]
# this can be very slow when dealing with long lists from scanning a large
# file system
    
# alternatively, we can simply re-scan the root directory and use the
# collected size data

with os.scandir(root) as it:
    for entry in it:
        if (includeHidden or not is_hidden(entry)) and entry.is_dir() :
            sizesAccumulated['.'] += sizesAccumulated[entry.name]


# ---------------------------------------------------------------------------- #
# some nice output

for path, size in sorted( sizesAccumulated.items() ) :
    print(f"{size:12,} bytes under '{path:35}', thereof {sizesLocal[path]:12,} directly in the directory.")

# ============================================================================ #
# remote controlling GROMOCKS

for opt in ("3\n", "5\n", "7\n") :
    process = os.popen("python3 gromocks.py settings.ini >> output.txt", "w")
    process.write(opt)
    process.close()
    
# ---------------------------------------------------------------------------- #
