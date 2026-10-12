"""
Helper tool to execute commands and transfer files to Raspberry Pi 5 using SSH key.
"""
import sys
import argparse
from pathlib import Path
import paramiko
from scp import SCPClient

import os

KEY_FILE = r"C:\Users\Administrator\.ssh\id_rsa"
PI_HOST = os.environ.get("PI_HOST", "172.16.10.145")
PI_USER = os.environ.get("PI_USER", "cytron")

def get_client():
    k = paramiko.RSAKey.from_private_key_file(KEY_FILE)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(PI_HOST, username=PI_USER, pkey=k, timeout=15)
    return c

def run_remote_command(cmd: str) -> int:
    client = get_client()
    print(f"[PI EXEC] {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd)
    
    # Stream stdout safely avoiding Windows cp1252 charmap errors
    for line in iter(stdout.readline, ""):
        sys.stdout.buffer.write(line.encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()
    err = stderr.read().decode("utf-8", errors="replace")
    if err:
        sys.stderr.buffer.write(f"[PI STDERR] {err}".encode("utf-8", errors="replace"))
        sys.stderr.buffer.flush()
        
    exit_code = stdout.channel.recv_exit_status()
    client.close()
    return exit_code

def copy_to_pi(local_path: str, remote_path: str, recursive: bool = False):
    client = get_client()
    print(f"[SCP PUT] {local_path} -> {remote_path}")
    with SCPClient(client.get_transport()) as scp:
        scp.put(local_path, remote_path, recursive=recursive)
    client.close()
    print("[SCP DONE]")

def copy_from_pi(remote_path: str, local_path: str, recursive: bool = False):
    client = get_client()
    print(f"[SCP GET] {remote_path} -> {local_path}")
    with SCPClient(client.get_transport()) as scp:
        scp.get(remote_path, local_path, recursive=recursive)
    client.close()
    print("[SCP DONE]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")
    
    run_parser = subparsers.add_parser("exec")
    run_parser.add_argument("command", type=str)
    
    put_parser = subparsers.add_parser("put")
    put_parser.add_argument("local", type=str)
    put_parser.add_argument("remote", type=str)
    put_parser.add_argument("-r", "--recursive", action="store_true")

    get_parser = subparsers.add_parser("get")
    get_parser.add_argument("remote", type=str)
    get_parser.add_argument("local", type=str)
    get_parser.add_argument("-r", "--recursive", action="store_true")

    args = parser.parse_args()
    if args.action == "exec":
        ret = run_remote_command(args.command)
        sys.exit(ret)
    elif args.action == "put":
        copy_to_pi(args.local, args.remote, recursive=args.recursive)
    elif args.action == "get":
        copy_from_pi(args.remote, args.local, recursive=args.recursive)
    else:
        parser.print_help()
