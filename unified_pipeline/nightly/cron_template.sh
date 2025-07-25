#!/bin/bash
"""
Cron Template for Unified Pipeline Monitoring

This script provides a template for setting up automated monitoring
of the unified pipeline. It should be adapted to your specific environment
and scheduling needs.

Usage:
1. Copy this file to your desired location
2. Modify paths and configuration for your setup
3. Add to crontab with: crontab -e
4. Add line: 0 2 * * * /path/to/this/script.sh

The default schedule runs daily at 2 AM.
"""

# Configuration
PIPELINE_DIR="/path/to/unified_pipeline"
CONFIG_FILE="$PIPELINE_DIR/configs/full.yaml"
LOG_DIR="$PIPELINE_DIR/logs"
PYTHON_ENV="/path/to/python/env"  # Optional: path to Python environment

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Activate Python environment if specified
if [ ! -z "$PYTHON_ENV" ]; then
    source "$PYTHON_ENV/bin/activate"
fi

# Set up logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/drift_monitor_$TIMESTAMP.log"

echo "Starting drift monitoring at $(date)" >> "$LOG_FILE"

# Change to pipeline directory
cd "$PIPELINE_DIR"

# Run drift monitoring
echo "Checking for drift..." >> "$LOG_FILE"
python nightly/drift_monitor.py \
    --config "$CONFIG_FILE" \
    --action check_drift \
    >> "$LOG_FILE" 2>&1

DRIFT_EXIT_CODE=$?

# Generate weekly report on Sundays
if [ $(date +%u) -eq 7 ]; then
    echo "Generating weekly report..." >> "$LOG_FILE"
    python nightly/drift_monitor.py \
        --config "$CONFIG_FILE" \
        --action generate_report \
        --days_back 7 \
        >> "$LOG_FILE" 2>&1
fi

# Log completion
echo "Drift monitoring completed at $(date) with exit code $DRIFT_EXIT_CODE" >> "$LOG_FILE"

# Optional: Clean up old log files (keep last 30 days)
find "$LOG_DIR" -name "drift_monitor_*.log" -mtime +30 -delete

# Optional: Send summary email if monitoring failed
if [ $DRIFT_EXIT_CODE -ne 0 ]; then
    echo "Drift monitoring failed. Check logs at $LOG_FILE" | \
    mail -s "Unified Pipeline Monitor Alert" admin@yourcompany.com
fi

exit $DRIFT_EXIT_CODE