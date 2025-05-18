#!/bin/bash
set -e
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