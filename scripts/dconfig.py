#!/usr/bin/env python3

import os
import argparse
import glob


def get_argument_parser():
    """Creates the argument parser."""

    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        description="dconfig", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--runs_on",
        required=True,
        type=str,
        help="Comma-separated IP addresses of machines",
    )

    parser.add_argument(
        "--config_script",
        required=True,
        type=str,
        help="Path to bash script with configuration",
    )

    parser.add_argument(
        "--ssh_cmd",
        required=False,
        type=str,
        default="ssh -f {addr}",
        help="SSH command. Useful to utilize a pre-shared key with 'ssh -i mykey.pem -f ubuntu@{addr}'. "
        "The option `-f` should be used for non-interactive session",
    )

    return parser


def get_args():
    """Creates the argument parser and parses any input arguments."""

    parser = get_argument_parser()
    args = parser.parse_args()

    return args


def wrap_single(text):
    return f"'{text}'"


def wrap_single_nested(text, quote=r"'\''"):
    return f"{quote}{text}{quote}"


if __name__ == "__main__":
    args = get_args()

    all_addresses = args.runs_on.split(",")
    print(f"Running on addresses {all_addresses}")

    remote_config_script = f"{args.config_script}.distributed"
    for it, addr in enumerate(all_addresses):
        scp_cmd = (
            args.ssh_cmd.replace("ssh ", "scp ")
            .replace("-f", args.config_script)
            .format(addr=addr)
        )
        print(f"SCP command {scp_cmd}:{remote_config_script}")
        os.system(f"{scp_cmd}:{remote_config_script}")

        screen_name = f"allenact_config_machine{it}"
        bash_command = wrap_single_nested(
            f"source {remote_config_script} &>> log_allenact_distributed_config"
        )
        screen_command = wrap_single(
            f"screen -S {screen_name} -dm bash -c < {bash_command}"
        )

        ssh_command = f"{args.ssh_cmd.format(addr=addr)} {screen_command}"

        print(f"SSH command {ssh_command}")
        os.system(ssh_command)
        print(f"{addr} {screen_name}")

    print("DONE")
