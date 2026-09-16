#!/bin/sh

case "$NEO4J_OUTPUT_PATH" in
  /*)
    # Linux / macOS
    TARGET="$NEO4J_OUTPUT_PATH"
    ;;
  *)
    # Windows, e.g. C:/repos/...
    TARGET="/$NEO4J_OUTPUT_PATH"
    ;;
esac

mkdir -p "$(dirname "$TARGET")"
ln -sfn /neo4j-output "$TARGET"