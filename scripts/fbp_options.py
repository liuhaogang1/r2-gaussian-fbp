"""Argument boundary for opt-in FBP features (standard library only)."""
import argparse


def parse_svoxel_option(value):
    if isinstance(value, str):
        value = value.split()
    items = list(value)
    if len(items) == 1 and str(items[0]).lower() == "auto":
        return "auto"
    if len(items) != 3:
        raise ValueError("sVoxel requires X Y Z, or the single value 'auto'")
    # Numeric options keep the original float conversion/validation policy.
    return [float(item) for item in items]


class SVoxelAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            setattr(namespace, self.dest, parse_svoxel_option(values))
        except (TypeError, ValueError) as error:
            raise argparse.ArgumentError(self, str(error)) from error
