#! /usr/bin/env nu
extern-wrapped main [...args] {
    overlay use ./venv/bin/activate.nu
    python3 quickdif.py $args
}
