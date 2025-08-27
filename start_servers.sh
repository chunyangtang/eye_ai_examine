#!/usr/bin/env bash
set -euo pipefail

# --- Colors ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== AI Eye Clinic 辅助诊断系统 启动脚本 ===${NC}\n"

# --- Helpers ---
port_in_use() {
  local port="$1"
  lsof -iTCP:"$port" -sTCP:LISTEN -t >/dev/null 2>&1
}

pick_free_port() {
  local candidates=("$@")
  for p in "${candidates[@]}"; do
    if ! port_in_use "$p"; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

# --- Detect project root ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Detect local IP (macOS-friendly) ---
LOCAL_IP="$(ipconfig getifaddr en0 2>/dev/null || true)"
if [ -z "${LOCAL_IP}" ]; then
  LOCAL_IP="$(ipconfig getifaddr en1 2>/dev/null || true)"
fi
if [ -z "${LOCAL_IP}" ]; then
  LOCAL_IP="$(ifconfig | awk '/inet / && $2!="127.0.0.1"{print $2; exit}')"
fi
if [ -z "${LOCAL_IP}" ]; then
  LOCAL_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi
if [ -z "${LOCAL_IP}" ]; then
  LOCAL_IP="127.0.0.1"
fi

echo -e "${GREEN}检测到本机IP地址: ${LOCAL_IP}${NC}\n"

# --- Ports (auto-fallback if busy) ---
BACKEND_PORT_DEFAULT="${BACKEND_PORT:-8000}"
FRONTEND_PORT_DEFAULT="${FRONTEND_PORT:-3000}"

BACKEND_PORT="$(pick_free_port "$BACKEND_PORT_DEFAULT" 8001 8002 || echo "$BACKEND_PORT_DEFAULT")"
FRONTEND_PORT="$(pick_free_port "$FRONTEND_PORT_DEFAULT" 3001 3002 || echo "$FRONTEND_PORT_DEFAULT")"

if port_in_use "$BACKEND_PORT_DEFAULT"; then
  echo -e "${YELLOW}后端端口 ${BACKEND_PORT_DEFAULT} 被占用，改用 ${BACKEND_PORT}${NC}"
fi
if port_in_use "$FRONTEND_PORT_DEFAULT"; then
  echo -e "${YELLOW}前端端口 ${FRONTEND_PORT_DEFAULT} 被占用，改用 ${FRONTEND_PORT}${NC}"
fi

echo -e "${YELLOW}启动说明:${NC}
1. 后端服务器: http://${LOCAL_IP}:${BACKEND_PORT}
2. 前端开发服务器: http://${LOCAL_IP}:${FRONTEND_PORT}
3. 内网访问: 在同一局域网内使用上述地址
4. 访问特定检查: http://${LOCAL_IP}:${FRONTEND_PORT}/?ris_exam_id=<exam_id>\n"

# --- Check tools ---
if ! command -v python3 >/dev/null 2>&1; then
  echo -e "${RED}未检测到 python3，请先安装。${NC}"; exit 1
fi
if ! python3 -c "import uvicorn" >/dev/null 2>&1; then
  if ! command -v uvicorn >/dev/null 2>&1; then
    echo -e "${YELLOW}未检测到 uvicorn，将尝试使用 python3 -m uvicorn（请确保已安装 uvicorn）。${NC}"
  fi
fi
if ! command -v npm >/dev/null 2>&1; then
  echo -e "${RED}未检测到 npm，请先安装 Node.js/npm。${NC}"; exit 1
fi

# --- Start backend (FastAPI via uvicorn) ---
echo -e "${BLUE}正在启动后端服务器...${NC}"
pushd backend >/dev/null

# Prefer python module invocation for portability
BACKEND_LOG="${SCRIPT_DIR}/.backend.log"
( python3 -m uvicorn main:app --reload --host 0.0.0.0 --port "${BACKEND_PORT}" >"${BACKEND_LOG}" 2>&1 ) &
BACKEND_PID=$!

sleep 2
if ! kill -0 "${BACKEND_PID}" >/dev/null 2>&1; then
  echo -e "${RED}后端启动失败，日志如下:${NC}"
  tail -n +1 "${BACKEND_LOG}" || true
  popd >/dev/null
  exit 1
fi
popd >/dev/null

# --- Start frontend (CRA dev server, network-accessible) ---
echo -e "${BLUE}正在启动前端开发服务器...${NC}"
pushd frontend/eye-frontend >/dev/null

# Install deps if needed
if [ ! -d "node_modules" ]; then
  echo -e "${YELLOW}检测到缺少依赖，正在执行 npm install...${NC}"
  npm install
fi

FRONTEND_LOG="${SCRIPT_DIR}/.frontend.log"
# For CRA: HOST to 0.0.0.0, PORT to selected port, prevent auto-open
( HOST=0.0.0.0 PORT="${FRONTEND_PORT}" BROWSER=none npm start >"${FRONTEND_LOG}" 2>&1 ) &
FRONTEND_PID=$!

# Give it a moment to bind
sleep 4
if ! kill -0 "${FRONTEND_PID}" >/dev/null 2>&1; then
  echo -e "${RED}前端启动失败，日志如下:${NC}"
  tail -n +1 "${FRONTEND_LOG}" || true
  popd >/dev/null
  # Cleanup backend if frontend failed
  kill "${BACKEND_PID}" 2>/dev/null || true
  exit 1
fi
popd >/dev/null

echo ""
echo -e "${GREEN}服务器启动完成！${NC}\n"
echo -e "${YELLOW}访问地址:${NC}"
echo "• 本机访问:       http://localhost:${FRONTEND_PORT}"
echo "• 内网其他设备:   http://${LOCAL_IP}:${FRONTEND_PORT}"
echo "• 后端API:        http://${LOCAL_IP}:${BACKEND_PORT}"
echo "• 指定检查样例:   http://${LOCAL_IP}:${FRONTEND_PORT}/?ris_exam_id=11330226"
echo ""
echo -e "${YELLOW}日志文件:${NC}"
echo "• 后端: ${BACKEND_LOG}"
echo "• 前端: ${FRONTEND_LOG}"
echo ""
echo -e "${YELLOW}停止服务器:${NC} 按 Ctrl+C 停止，脚本会自动清理进程。"

cleanup() {
  echo -e "\n${YELLOW}正在停止服务器...${NC}"
  kill "${BACKEND_PID}" 2>/dev/null || true
  kill "${FRONTEND_PID}" 2>/dev/null || true
  # Extra: try to kill children of npm start (macOS)
  pkill -P "${FRONTEND_PID}" 2>/dev/null || true
  echo -e "${GREEN}已停止。${NC}"
  exit 0
}

trap cleanup SIGINT SIGTERM
wait