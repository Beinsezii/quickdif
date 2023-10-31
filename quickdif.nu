#! /usr/bin/env nu
extern-wrapped main [...args] {
    overlay use ./virtualenv/bin/activate.nu
    python3 quickdif.py $args
}
