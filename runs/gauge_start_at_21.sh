#!/usr/bin/env bash
# start the GAUGE driver at 21:00 local (user: GPU runs overnight)
now=$(date +%s); tgt=$(date -d "21:00" +%s); [ "$tgt" -gt "$now" ] && sleep $((tgt - now))
cd /home/prashr/mapformer && DRV_GPUS="0 1" bash run_gauge.sh
