#!/usr/bin/env python

from kale.services.worker import get_kale_id, spawn_worker, KaleWorkerClient
from kale.services.manager import KaleManagerClient

import argparse
import importlib
import requests
import sys
import time

def launch_ipengine(*args, **kwargs):
    import ipyparallel.apps.ipengineapp as ipe

    if args:
        ipe.launch_new_instance(argv=args)
    else:
        ipe.launch_new_instance()

def launch_ipcontroller(*args, **kwargs):
    import ipyparallel.apps.ipcontrollerapp as ipc

    print(args)

    if args:
        ipc.launch_new_instance(argv=args)
    else:
        ipc.launch_new_instance()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mhost", help="Kale Manager Host IP")
    parser.add_argument("--mport", help="Kale Manager Port", type=int)
    launchers = parser.add_mutually_exclusive_group(required=True)
    launchers.add_argument("--ipcontroller", help="Launch ipcontroller", action='store_true')    
    launchers.add_argument("--ipengine", help="Launch ipengine", action='store_true')
    parser.add_argument("--task_args", help="Task command line args string")

    args = parser.parse_args()

    _mhost = "127.0.0.1"
    if args.mhost:
        _mhost = args.mhost

    _mport = 8099
    if args.mport:
        _mport = args.mport

    _task = None
    if args.ipcontroller:
        _task = launch_ipcontroller
    elif args.ipengine:
        _task = launch_ipengine

    _task_args = []
    if args.task_args:
        _task_args = args.task_args.split()

    mgr = KaleManagerClient(_mhost, _mport)

    # spawn kale worker
    print("Starting Kale Worker...")
    kale_id = get_kale_id()
    w = spawn_worker(kale_id, whost="0.0.0.0", mhost=_mhost, mport=_mport)

    while 1:
        print("Waiting for worker connection info...")
        try:
            info = mgr.get_worker(kale_id)
            print(info)
            break
        except requests.HTTPError:
            time.sleep(0.1)

    kale_worker = KaleWorkerClient(info["host"], info["port"])

    while 1:
        print("Waiting for worker to start listening...")
        try:
            kale_worker.get_service_status()
            break
        except requests.ConnectionError:
            time.sleep(0.1)

    # launch engine
    print("Starting Kale Task with args {}...".format(_task_args))
    kale_task = kale_worker.register_function_task(
        _task,
        _task_args)
    kale_worker.start_task(kale_task)
