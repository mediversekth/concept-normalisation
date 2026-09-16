#!/bin/sh

# Convert Windows backslashes to forward slashes
PATH_TO_NEO4J_OUTPUT=$(printf '%s' "$PATH_TO_NEO4J_OUTPUT" | tr '\\' '/')

case "$PATH_TO_NEO4J_OUTPUT" in
  /*)
    # Linux / macOS
    TARGET="$PATH_TO_NEO4J_OUTPUT"
    ;;
  *)
    # Windows, e.g. C:/repos/...
    TARGET="/$PATH_TO_NEO4J_OUTPUT"
    ;;
esac

mkdir -p "$(dirname "$TARGET")"
ln -sfn /neo4j-output "$TARGET"