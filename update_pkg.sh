#!/bin/bash

VERSION_FILE="./src/bio_lib/version.py"
# Use awk instead of grep -oP for macOS compatibility
CURRENT_VERSION=$(awk -F'"' '/^__version__/{print $2}' $VERSION_FILE)

# Split version into major, minor, patch
IFS='.' read -r major minor patch <<< "$CURRENT_VERSION"

# Convert to integers for arithmetic
major=$((10#$major))
minor=$((10#$minor))
patch=$((10#$patch))

# Increment with rollover logic
if [ $patch -eq 9 ]; then
    patch=0
    if [ $minor -eq 9 ]; then
        minor=0
        major=$((major + 1))
    else
        minor=$((minor + 1))
    fi
else
    patch=$((patch + 1))
fi

NEW_VERSION="${major}.${minor}.${patch}"

# Update version.py using perl instead of sed for macOS compatibility
perl -pi -e "s/__version__ = \".*\"/__version__ = \"${NEW_VERSION}\"/" $VERSION_FILE

echo "Updated version from ${CURRENT_VERSION} to ${NEW_VERSION}"

# Clean and build
rm -rf dist/
python -m build

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*