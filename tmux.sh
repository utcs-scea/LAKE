#!/bin/sh

#SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

SCRIPT=`realpath $0`
SCRIPT_DIR=`dirname $SCRIPT`

tmux new-session -d -c ${SCRIPT_DIR}/kava/scripts
tmux split-window -h -c ${SCRIPT_DIR}/kava/driver -l 60
tmux select-pane -t 0
tmux split-window -v 'dmesg -w'
tmux select-pane -t 0
tmux send-keys './load_all.sh'
tmux -2 attach-session -d
