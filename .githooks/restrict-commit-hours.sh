#!/bin/bash

# Get current hour in 24-hour format (0-23)
HOUR=$(date +%H)

# Get current day of week (1-7, where 1 is Monday)
DAY=$(date +%u)

# Check if it's a weekday (1-5 are Monday to Friday)
if [ $DAY -ge 1 ] && [ $DAY -le 5 ]; then
  # Check if current hour is between 9 and 17 (9 AM to 5 PM)
  if [ $HOUR -ge 9 ] && [ $HOUR -lt 17 ]; then
    echo "ERROR: Commits are restricted during these hours (9 AM - 5 PM on weekdays)."
    echo "Current time: $(date +"%A, %H:%M")"
    exit 1
  fi
fi

# If we get here, the commit is allowed
exit 0