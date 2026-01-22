# 创建一个简单的脚本来处理
PORT=8000
PID=$(lsof -t -i:$PORT)
if [ -z "$PID" ]; then
    echo "Port $PORT is not in use"
else
    echo "Killing process $PID using port $PORT"
    kill -9 $PID
fi