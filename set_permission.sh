#!/bin/bash

# === Jetson CPU Frequency and hwmon Permission Setting Script ===
# 사용법:
#   sudo ./set_permission.sh

chmod 666 /sys/devices/system/cpu/cpufreq/policy*/scaling_governor
chmod 666 /sys/devices/system/cpu/cpufreq/policy*/scaling_setspeed
chmod 666 /sys/devices/system/cpu/cpufreq/policy*/scaling_min_freq
chmod 666 /sys/devices/system/cpu/cpufreq/policy*/scaling_max_freq

chmod a+r /sys/class/hwmon/hwmon4/*_input

echo "CPU frequency and hwmon permissions set."
