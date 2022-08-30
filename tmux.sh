#!/bin/sh

#SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

SCRIPT=`realpath $0`
SCRIPT_DIR=`dirname $SCRIPT`

tmux new-session -d -c ${SCRIPT_DIR}/src/kapi
tmux split-window -h -c ${SCRIPT_DIR}/src
tmux select-pane -t 0
tmux split-window -v 'dmesg -w'
tmux select-pane -t 0
tmux send-keys 'sudo ./load.sh'
tmux -2 attach-session -d
