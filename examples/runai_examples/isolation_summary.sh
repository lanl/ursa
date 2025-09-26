#!/usr/bin/env bash
# container_isolation_summary.sh
# Skimmable isolation summary with colorized OK/WARN/INFO markers.

set -euo pipefail

# ---------- colors ----------
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1; then
  BOLD="$(tput bold)"; DIM="$(tput dim)"; RESET="$(tput sgr0)"
  GREEN="$(tput setaf 2)"; YELLOW="$(tput setaf 3)"; RED="$(tput setaf 1)"; CYAN="$(tput setaf 6)"
else
  BOLD=""; DIM=""; RESET=""; GREEN=""; YELLOW=""; RED=""; CYAN=""
fi
ok()   { echo -e "${GREEN}[OK]${RESET}   $*"; }
warn() { echo -e "${YELLOW}[WARN]${RESET} $*"; }
err()  { echo -e "${RED}[ERR]${RESET}  $*"; }
inf()  { echo -e "${CYAN}[INFO]${RESET}  $*"; }

TS="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
HOST="$(hostname 2>/dev/null || echo unknown)"
KERNEL="$(uname -srmo 2>/dev/null || echo unknown)"

echo -e "${BOLD}Container Isolation Summary${RESET}  ${DIM}$TS UTC${RESET}"
echo -e "${DIM}Host:${RESET} $HOST   ${DIM}Kernel:${RESET} $KERNEL"
echo

# ---------- helpers ----------
have(){ command -v "$1" >/dev/null 2>&1; }
readfile(){ [[ -r "$1" ]] && cat "$1" || echo ""; }
first_line(){ head -n1 2>/dev/null || true; }

# ---------- PID / processes ----------
PID1_CMD="$(ps -p 1 -o cmd= 2>/dev/null || echo "")"
NS_PID1="$(ls -l /proc/1/ns/pid 2>/dev/null | awk '{print $NF}' | sed 's/.*\[\(.*\)\]/\1/')"
NS_SELF="$(ls -l /proc/self/ns/pid 2>/dev/null | awk '{print $NF}' | sed 's/.*\[\(.*\)\]/\1/')"
PROC_COUNT="$(ps -e 2>/dev/null | wc -l || echo 0)"

echo -e "${BOLD}PID / Processes${RESET}"
if [[ -n "$NS_PID1" && -n "$NS_SELF" && "$NS_PID1" = "$NS_SELF" ]]; then
  ok "PID namespace present (pid ns: $NS_SELF); PID 1 cmd: ${PID1_CMD:-unknown}; processes: $PROC_COUNT"
else
  warn "PID namespace ambiguous (pid1:$NS_PID1 self:$NS_SELF); PID 1 cmd: ${PID1_CMD:-unknown}; processes: $PROC_COUNT"
fi
echo

# ---------- Filesystem / mounts ----------
MOUNTS="$( (have findmnt && findmnt -A -n -o TARGET,PROPAGATION,FSTYPE,SOURCE | head -n 200) || mount | head -n 200 )"
ROOT_STAT="$( (have stat && stat -f -c 'type=%T, total=%s, free=%a' /) || echo "" )"
OVERLAY_HINT="$(echo "$MOUNTS" | grep -E ' overlay | fuse-overlayfs | aufs ' -i || true)"

echo -e "${BOLD}Filesystem / Mounts${RESET}"
if [[ -n "$OVERLAY_HINT" ]]; then
  ok "Root filesystem appears layered (overlay/aufs)"
else
  inf "No overlay hint found in first 200 mounts (may still be containerized)"
fi
[[ -n "$ROOT_STAT" ]] && inf "Root fs stat: $ROOT_STAT"

check_mount() {
  local p="$1"; [[ -d "$p" ]] || { inf "$p: not present"; return; }
  local line
  if have findmnt; then
    line="$(findmnt -n "$p" 2>/dev/null || true)"
  else
    line="$(echo "$MOUNTS" | awk -v p="$p" '$1==p{print}')"
  fi
  if [[ -n "$line" ]]; then
    ok "$p is a mount point → ${DIM}${line}${RESET}"
  else
    inf "$p exists but not a distinct mount point"
  fi
  if [[ -w "$p" ]]; then ok "$p is writable"; else warn "$p not writable"; fi
}

for p in /mnt /mnt/data /workspace /vllm-workspace /tmp "$HOME"; do check_mount "$p"; done
echo

# ---------- Network ----------
IF_SUMMARY="$( (have ip && ip -o -4 addr show 2>/dev/null) || ifconfig -a 2>/dev/null || echo "" )"
ROUTE="$( (have ip && ip route 2>/dev/null) || route -n 2>/dev/null || echo "" )"
IF_COUNT="$(echo "$IF_SUMMARY" | grep -v ' lo ' | grep -c 'inet ' || echo 0)"
DEFAULT_GW="$(echo "$ROUTE" | awk '/^default|^0.0.0.0/ {print $0; exit}')"

echo -e "${BOLD}Network${RESET}"
if [[ "$IF_COUNT" -ge 1 ]]; then
  ok "Interfaces with IPv4: $IF_COUNT"
else
  warn "No non-loopback IPv4 interfaces detected"
fi
[[ -n "$DEFAULT_GW" ]] && inf "Default route: $DEFAULT_GW"
echo

# ---------- Cgroups / resource limits ----------
CG_LINE="$(grep -E '^[0-9]+::' /proc/self/cgroup 2>/dev/null || true)"
if mount | grep -q 'type cgroup2'; then CG_VER="v2"; else CG_VER="v1"; fi
CG_PATH="${CG_LINE##*::}"; [[ -z "$CG_PATH" ]] && CG_PATH="/"
BASE_V2="/sys/fs/cgroup$CG_PATH"

echo -e "${BOLD}Cgroups / Limits${RESET}"
inf "cgroup version: $CG_VER   path: $CG_PATH"
if [[ "$CG_VER" = "v2" ]]; then
  MEM_MAX="$(readfile "$BASE_V2/memory.max")"
  CPU_MAX="$(readfile "$BASE_V2/cpu.max")"
  PIDS_MAX="$(readfile "$BASE_V2/pids.max")"
  [[ -n "$MEM_MAX" ]] && { [[ "$MEM_MAX" = "max" ]] && warn "Memory: unlimited" || ok "Memory max: $MEM_MAX bytes"; }
  [[ -n "$CPU_MAX" ]] && ok "CPU quota/period: $CPU_MAX"
  [[ -n "$PIDS_MAX" ]] && ok "PIDs max: $PIDS_MAX"
else
  V1_MEM="$(readfile /sys/fs/cgroup/memory/memory.limit_in_bytes)"
  V1_QUO="$(readfile /sys/fs/cgroup/cpu/cpu.cfs_quota_us)"
  V1_PER="$(readfile /sys/fs/cgroup/cpu/cpu.cfs_period_us)"
  V1_PID="$(readfile /sys/fs/cgroup/pids/pids.max)"
  [[ -n "$V1_MEM" ]] && { [[ "$V1_MEM" = "max" || "$V1_MEM" = "-1" ]] && warn "Memory: unlimited" || ok "Memory limit: $V1_MEM bytes"; }
  [[ -n "$V1_QUO$V1_PER" ]] && ok "CPU quota/period: ${V1_QUO:-?}/${V1_PER:-?}"
  [[ -n "$V1_PID" ]] && ok "PIDs max: $V1_PID"
fi
echo

# ---------- Capabilities / privileges ----------
CAPS=""
if have capsh; then
  CAPS="$(capsh --print 2>/dev/null | awk '/Current:/{$1=""; print substr($0,2)}' | tr -d '\r')"
else
  CAPS="$(grep -E '^(CapEff|CapPrm|CapBnd)' /proc/self/status 2>/dev/null | tr -d '\r')"
fi
IS_ROOT="$(test "$(id -u)" -eq 0 && echo yes || echo no)"

echo -e "${BOLD}Privileges / Caps${RESET}"
if [[ "$IS_ROOT" = "yes" ]]; then
  warn "Running as root (uid=0)"
else
  ok "Running as non-root (uid=$(id -u))"
fi
[[ -n "$CAPS" ]] && inf "Capabilities: $(echo "$CAPS" | head -n1)"
echo

# ---------- Devices ----------
GPU_DEVS="$(ls /dev/nvidia* 2>/dev/null || true)"
if [[ -n "$GPU_DEVS" ]]; then
  ok "NVIDIA devices present: $(echo "$GPU_DEVS" | wc -w) device(s)"
  if have nvidia-smi; then
    inf "nvidia-smi: $(nvidia-smi --query-gpu=name,uuid --format=csv,noheader 2>/dev/null | head -n1)"
  else
    inf "nvidia-smi not found"
  fi
else
  inf "No NVIDIA device nodes found"
fi
echo

# ---------- Writable checks ----------
echo -e "${BOLD}Writable Locations${RESET}"
for d in / /tmp "$HOME" /mnt /mnt/data /workspace /vllm-workspace; do
  [[ -d "$d" ]] || { inf "$d: not present"; continue; }
  if [[ -w "$d" ]]; then ok "$d writable"; else warn "$d NOT writable"; fi
done
echo

# ---------- Container clues ----------
CLUES=()
[[ -f /.dockerenv ]] && CLUES+=("/.dockerenv")
[[ -f /run/.containerenv ]] && CLUES+=("/run/.containerenv")
[[ -n "${KUBERNETES_SERVICE_HOST:-}" ]] && CLUES+=("KUBERNETES_SERVICE_HOST set")
[[ -n "${RUNAI_*:-}" ]] && CLUES+=("RUNAI_* env present")
[[ -n "$(grep -E 'docker|containerd|kubepods|runai' /proc/self/cgroup 2>/dev/null)" ]] && CLUES+=("cgroup path mentions docker/containerd/kubepods/runai")

echo -e "${BOLD}Container Clues${RESET}"
if [[ ${#CLUES[@]} -gt 0 ]]; then
  ok "Indicators: ${CLUES[*]}"
else
  inf "No strong container markers detected (not definitive)"
fi
echo

# ---------- Summary verdict ----------
echo -e "${BOLD}Quick Verdict${RESET}"
SCORE=0
[[ -n "$OVERLAY_HINT" ]] && ((SCORE+=1))
[[ "$IF_COUNT" -ge 1 ]] && ((SCORE+=1))
grep -qE 'docker|containerd|kubepods|runai' /proc/self/cgroup 2>/dev/null && ((SCORE+=1))
[[ "$IS_ROOT" = "no" ]] && ((SCORE+=1))
[[ -n "$DEFAULT_GW" ]] && ((SCORE+=1))
[[ -n "$GPU_DEVS" ]] && ((SCORE+=1))
[[ "$CG_VER" = "v2" ]] && ((SCORE+=1))

case $SCORE in
  0|1|2)   warn "Low evidence of containerization/isolation (SCORE=$SCORE). Environment may be host-like." ;;
  3|4)     ok   "Moderate isolation (SCORE=$SCORE). Looks containerized with typical defaults." ;;
  5|6|7)   ok   "Strong isolation indicators (SCORE=$SCORE). Containerized with resource and namespace separation." ;;
esac
