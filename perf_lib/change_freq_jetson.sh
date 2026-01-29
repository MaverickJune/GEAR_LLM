#!/bin/bash
# === Jetson CPU Frequency Control Script (with arguments) ===
# 사용법:
#   sudo ./change_freq_jetson.sh <CPU_FREQ_KHZ>
#
# 예:
#   sudo ./change_freq_jetson.sh 1984000

set -e

# ---------- 입력값 검증 ----------
if [ $# -ne 1 ]; then
    echo "Usage: sudo ./change_freq_jetson.sh <CPU_FREQ_KHZ>"
    echo "예: sudo ./change_freq_jetson.sh 1984000"
    exit 1
fi

# root 체크 (Jetson은 보통 su 대신 sudo)
if [ "$(id -u)" != "0" ]; then
    echo "[!] root 권한 필요. 예: sudo ./change_freq_jetson.sh $*"
    exit 1
fi

CPU_FREQ_KHZ="$1"

echo "[*] 입력값: CPU=${CPU_FREQ_KHZ} kHz"

# ---------- CPU 설정 ----------
# Jetson: cpu0 정책이 0~3, cpu4 정책이 4~7 컨트롤한다고 가정
set_cpu_policy() {
    CPU_POL="$1"  # 0 or 4
    BASE="/sys/devices/system/cpu/cpu${CPU_POL}/cpufreq"

    GOV="${BASE}/scaling_governor"
    MAXF="${BASE}/scaling_max_freq"
    SETS="${BASE}/scaling_setspeed"
    CURF="${BASE}/scaling_cur_freq"

    if [ ! -d "$BASE" ]; then
        echo "[!] ${BASE} 없음 (cpufreq 경로 확인 필요)"
        return 1
    fi
    if [ ! -f "$GOV" ] || [ ! -f "$MAXF" ] || [ ! -f "$SETS" ]; then
        echo "[!] ${BASE} 아래 필요한 노드가 없음 (governor/max/setspeed)"
        return 1
    fi

    echo "[*] cpu${CPU_POL} policy 설정: governor=userspace, max=${CPU_FREQ_KHZ}, setspeed=${CPU_FREQ_KHZ}"
    echo userspace > "$GOV"
    echo 1984000 > "$MAXF"
    echo "$CPU_FREQ_KHZ" > "$SETS"

    if [ -f "$CURF" ]; then
        echo -n "    cur_freq: "
        cat "$CURF" || true
    fi
}

echo "[*] CPU policy cpu0 (0~3) 고정"
set_cpu_policy 0

echo "[*] CPU policy cpu4 (4~7) 고정"
set_cpu_policy 4

echo "[*] 모든 설정 완료!"