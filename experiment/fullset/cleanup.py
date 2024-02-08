#!/usr/bin/python3
import os, subprocess

def cleanup():
    root = os.path.split(os.path.abspath(__file__))[0]
    subprocess.run("rm -rf ./*.png ./results.json ./generated/* ./outputs/*", shell=True, cwd=root)

if __name__ == '__main__':
    cleanup()
