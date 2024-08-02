#!/usr/bin/env bash

cd $HOME/PythonProjects/shm_broadcast/example
source ../.venv/bin/activate
echo "================================================================================"
python ../tests/test_shm_broadcast.py 1 1 64
echo "================================================================================"
python ../tests/test_shm_broadcast.py 1 4 64
echo "================================================================================"
python ../tests/test_shm_broadcast.py 1 8 64
echo "================================================================================"
sleep 5
python ../tests/test_shm_broadcast.py 8 1 64
echo "================================================================================"
python ../tests/test_shm_broadcast.py 8 4 64
echo "================================================================================"
python ../tests/test_shm_broadcast.py 8 8 64
echo "================================================================================"
sleep 5
python ../tests/test_shm_broadcast.py 16 1 64
echo "================================================================================"
python ../tests/test_shm_broadcast.py 16 4 64
echo "================================================================================"
python ../tests/test_shm_broadcast.py 16 8 64
echo "================================================================================"
sleep 5
python ../tests/test_shm_broadcast.py 32 1 64
echo "================================================================================"
python ../tests/test_shm_broadcast.py 32 4 64
echo "================================================================================"
python ../tests/test_shm_broadcast.py 32 8 64
echo "================================================================================"
sleep 5
python ../tests/test_shm_broadcast.py 64 1 64
echo "================================================================================"
python ../tests/test_shm_broadcast.py 64 4 64
echo "================================================================================"
python ../tests/test_shm_broadcast.py 64 8 64
echo "================================================================================"
sleep 5
python ../tests/test_shm_broadcast.py 128 1 64
echo "================================================================================"
python ../tests/test_shm_broadcast.py 128 4 64
echo "================================================================================"
python ../tests/test_shm_broadcast.py 128 8 64
echo "================================================================================"
sleep 5
python ../tests/test_shm_broadcast.py 256 1 64
echo "================================================================================"
python ../tests/test_shm_broadcast.py 256 4 64
echo "================================================================================"
python ../tests/test_shm_broadcast.py 256 8 64
echo "================================================================================"
sleep 5
python ../tests/test_shm_broadcast.py 512 1 64
echo "================================================================================"
python ../tests/test_shm_broadcast.py 512 4 64
echo "================================================================================"
python ../tests/test_shm_broadcast.py 512 8 64
echo "================================================================================"
sleep 5
python ../tests/test_shm_broadcast.py 1024 1 64
echo "================================================================================"
python ../tests/test_shm_broadcast.py 1024 4 64
echo "================================================================================"
python ../tests/test_shm_broadcast.py 1024 8 64
echo "================================================================================"
