#!/bin/sh

tmux new-session -d -c ~/HACK/kava/scripts
tmux split-window -h -c ~/HACK/kava/driver -l 60
tmux select-pane -t 0
tmux split-window -v 'dmesg -w'
tmux select-pane -t 0
tmux send-keys './load_all.sh'
tmux -2 attach-session -d