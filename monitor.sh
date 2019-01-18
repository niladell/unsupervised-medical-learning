if [ $(awk '/^MemAvailable:/ { print $2; }' /proc/meminfo) -lt 1000000 ]; then
    sh to_run2.sh
fi