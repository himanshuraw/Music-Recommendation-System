#!/bin/bash
set -e

# If running as root, fix permissions and re-exec as appuser
if [ "$(id -u)" = "0" ]; then
    # Ensure directories exist and set ownership
    mkdir -p /app/data /app/models
    chown -R appuser:appuser /app/data /app/models
    # Execute this script as appuser
    exec gosu appuser "$0" "$@"
fi

# Running as appuser from here
echo ">>> ENTRYPOINT.SH starting (user=$(whoami))"
echo ">>> build_data contains:"
ls -al /app/build_data
echo ">>> /app/data before copy:"
ls -al /app/data

# Initialize data volume if empty
if [ -z "$(ls -A /app/data)" ]; then
    echo "Initializing data volume from build context..."
    cp -r /app/build_data/* /app/data/
fi

# Initialize models volume if empty
if [ -z "$(ls -A /app/models)" ]; then
    echo "Initializing models volume from build context..."
    cp -r /app/build_models/* /app/models/
fi

exec "$@"