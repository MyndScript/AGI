#!/bin/sh
set -e
file /app/memory_server
ls -l /app/memory_server
ldd /app/memory_server || true
chmod +x /app/memory_server
exec /app/memory_server
